import os
import math
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import argparse

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import math
import re


import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree

# Optional: OSMnx and Pyrosm
import osmnx as ox
from pyrosm import OSM

# ------------------------- CONFIG -------------------------
OSM_PATH = "map.osm"      # <-- put your .osm/.xml or .pbf filename here
OUT_DIR  = "out"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
CRS_METRIC = "EPSG:5186"  # Korea Transverse Mercator (meters). Change if needed.

# Simulation / model
ALTITUDE_M = 20.0                # fixed altitude for all drones (m)
COMM_RANGE_M = 300.0             # max comms link distance (m)
SAMPLE_EVERY_M = 25.0            # roadmap sampling interval along streets (m)
RANDOM_SEED = 42

# -------- Formation (GRID, stationary + auto-wide) --------
GRID_ROWS = 4
GRID_COLS = 5                    # 4×5 => 20 drones
AUTO_WIDE_GRID = True            # auto-fit grid to the map extent
ENSURE_MULTIHOP = True           # adapt spacing so farthest pair has a multi-hop path (not direct)
MULTIHOP_MAX_ITERS = 12          # attempts to tweak spacing/center
SPACING_SHRINK = 0.9             # factor to shrink spacing when trying to form a path
CENTER_JITTER_M = 0.0            # set >0 to try nudging the grid center (optional)
GRID_COVERAGE = 0.9              # portion of bbox spanned by the grid (0–1)
MIN_GRID_SPACING_M = 60.0        # lower bound so it gets WIDE even on small maps
STRETCH_ASPECT = True            # anisotropic spacing so it truly fills bbox
GRID_SPACING_M = 50.0            # used only if AUTO_WIDE_GRID = False
GRID_CENTER_XY: Optional[Tuple[float, float]] = None  # None => use bbox centroid

# ----------------------------------------------------------

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

# ------------------ DATA LOADING ------------------

def load_osm_layers(osm_path: str, crs_metric: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return (buildings_gdf, roads_gdf) in metric CRS."""
    ext = os.path.splitext(osm_path)[1].lower()

    # if ext == ".pbf":
    #     osm = OSM(osm_path)
    #     buildings = osm.get_data_by_custom_criteria(
    #         custom_filter={"building": True},
    #         osm_keys_to_keep=["building", "height", "levels", "name"]
    #     )
    #     if buildings is None or len(buildings) == 0:
    #         raise ValueError("No buildings found in PBF.")
    #     roads = osm.get_network(network_type="drive")
    #     if roads is None or len(roads) == 0:
    #         raise ValueError("No road network found in PBF.")
    #     buildings = buildings.to_crs(crs_metric)
    #     roads = roads.to_crs(crs_metric)
    #     return buildings.set_geometry("geometry"), roads.set_geometry("geometry")

    # if ext in {".osm", ".xml"}:
    if ext in {".osm"}:
        # Buildings
        buildings = ox.features_from_xml(osm_path, tags={"building": True})
        if buildings is None or len(buildings) == 0:
            raise ValueError("No buildings found in OSM.")
        buildings = buildings[~buildings.geometry.is_empty].copy().to_crs(crs_metric)
        # Roads (edges from a graph or highway features fallback)
        try:
            G = ox.graph_from_xml(osm_path, simplify=True)
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            edges = edges[~edges.geometry.is_empty].copy().to_crs(crs_metric)
        except Exception:
            edges = ox.features_from_xml(osm_path, tags={"highway": True})
            edges = edges[edges.geometry.type.isin(["LineString", "MultiLineString"])].copy().to_crs(crs_metric)
        roads = gpd.GeoDataFrame(edges[["geometry"]].copy(), geometry="geometry", crs=edges.crs)
        return buildings.set_geometry("geometry"), roads

    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .osm")

# Try to load a routable OSMnx graph (projected to CRS_METRIC) for road shortest paths

def try_load_osmnx_graph(osm_path: str):
    ext = os.path.splitext(osm_path)[1].lower()
    # if ext not in {".osm", ".xml"}:
    if ext not in {".osm"}:
        return None
    try:
        G_ll = ox.graph_from_xml(osm_path, simplify=True)
        G_m = ox.projection.project_graph(G_ll, to_crs=CRS_METRIC)
        return G_m
    except Exception:
        return None

# ------------------ PREP ------------------


M_PER_FT = 0.3048

HEIGHT_KEYS = ["height", "height:mean", "building:height", "height_m"]
LEVEL_KEYS  = ["building:levels", "levels"]  # OSM often uses "building:levels"

def _parse_height_m(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip().lower().replace(",", "")
    if s in ("", "nan", "none", "null"):
        return None

    # feet-inches like 10'6" or 10ft or 10 ft 6 in
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*'\s*(\d+(?:\.\d+)?)\s*\"?\s*", s)
    if m:
        ft = float(m.group(1)); inch = float(m.group(2))
        return (ft + inch/12.0) * M_PER_FT

    # pure feet
    if "ft" in s or s.endswith("'"):
        num = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
        if num:
            return float(num[0]) * M_PER_FT

    # meters (default unit for OSM height)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if nums:
        # handle ranges like "10-12 m": take the first number
        return float(nums[0])

    # give up
    return None

def estimate_height_m(row) -> float:
    height_levels = 100.0  # default fallback height in meters
    # 1) try explicit height in meters/feet
    for key in HEIGHT_KEYS:
        if key in row:
            h = row.get(key)
            hm = _parse_height_m(h)
            if hm is not None and np.isfinite(hm) and hm > 0:
                return float(hm)

    # 2) fallback to levels × storey height
    for key in LEVEL_KEYS:
        if key in row:
            v = row.get(key)
            try:
                if v is not None and not pd.isna(v):
                    levels = float(str(v).strip())
                    if levels > 0:
                        height_levels = 30 * levels
                        print(height_levels)
                        # return 3.0 * levels # tweak if you prefer 3.0
                        return float(height_levels)
                    
            except Exception:
                pass  # try next key / fallback

    # 3) final fallback
    return height_levels

# old broken height estimation
# def estimate_height_m(row) -> float:
#     h = row.get("height"); v = row.get("levels")
    
#     try:
#         if h is not None and not pd.isna(h):
#             s = str(h).lower().replace("m", "").strip(); return float(s)
            
#     except Exception: pass
#     try:
#         if v is not None and not pd.isna(v):
#             print(v)
#             return 3.3 * float(v)
        
#     except Exception: pass
    
#     return 12.0

def prepare_buildings(buildings: gpd.GeoDataFrame, altitude_m: float) -> Tuple[gpd.GeoDataFrame, gpd.GeoSeries, STRtree]:
    b = buildings.copy()
    b["h_m"] = b.apply(estimate_height_m, axis=1)
    occluders = b[b["h_m"] >= altitude_m].geometry
    tree = STRtree(list(occluders.values))
    return b, occluders, tree


def roads_to_centerlines(roads: gpd.GeoDataFrame) -> gpd.GeoSeries:
    merged = linemerge(unary_union(roads.geometry))
    if isinstance(merged, LineString):
        return gpd.GeoSeries([merged], crs=roads.crs)
    elif hasattr(merged, "geoms"):
        return gpd.GeoSeries(list(merged.geoms), crs=roads.crs)
    else:
        return gpd.GeoSeries([], crs=roads.crs)


def sample_lines(lines: gpd.GeoSeries, every_m: float) -> gpd.GeoDataFrame:
    pts = []
    for ln in lines:
        if ln.length < 1.0: continue
        d = 0.0
        while d <= ln.length:
            pts.append(Point(ln.interpolate(d)))
            d += every_m
    gdf = gpd.GeoDataFrame(geometry=pts, crs=lines.crs)
    gdf["x"], gdf["y"] = gdf.geometry.x, gdf.geometry.y
    return gdf


def build_roadmap(points: gpd.GeoDataFrame, k_nn: int = 6) -> nx.Graph:
    from sklearn.neighbors import NearestNeighbors
    coords = np.vstack([points["x"].values, points["y"].values]).T
    nbrs = NearestNeighbors(n_neighbors=min(k_nn, len(points)), algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    G = nx.Graph()
    for i, (x, y) in enumerate(coords):
        G.add_node(i, x=float(x), y=float(y))
    for i, row in enumerate(indices):
        for j_idx, j in enumerate(row):
            if j == i: continue
            d = float(distances[i][j_idx])
            G.add_edge(i, j, weight=d)
    return G

# Remove nodes over tall buildings (>= altitude) and any nodes inside buildings at all

def prune_nodes_over_tall_buildings(G: nx.Graph, points: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, altitude_m: float) -> nx.Graph:
    tall = buildings[buildings["h_m"] >= altitude_m]
    if len(tall) == 0: return G
    tall_union = unary_union(tall.geometry)
    to_remove = []
    for n, data in G.nodes(data=True):
        p = Point(data["x"], data["y"])
        if tall_union.contains(p): to_remove.append(n)
    G.remove_nodes_from(to_remove); return G

def prune_nodes_over_any_buildings(G: nx.Graph, buildings_all: gpd.GeoDataFrame) -> nx.Graph:
    if len(buildings_all) == 0: return G
    uni = unary_union(buildings_all.geometry)
    to_remove = []
    for n, d in G.nodes(data=True):
        if uni.contains(Point(d["x"], d["y"])): to_remove.append(n)
    G.remove_nodes_from(to_remove); return G

# NEW: remove roadmap edges that cross any building polygon (ensures routes avoid buildings)

def prune_edges_crossing_buildings(G, buildings_all):
    """
    Remove roadmap edges that cross (or run through) any building polygon.
    Works with Shapely 1.x (returns geometries) and 2.x (returns indices).
    """
    if len(buildings_all) == 0:
        return G

    # Keep a list in the same positional order used to build the tree
    b_geoms = [g for g in buildings_all.geometry.values if g is not None and not g.is_empty]
    if not b_geoms:
        return G

    tree = STRtree(b_geoms)
    to_remove = []

    for u, v in list(G.edges()):
        p1 = Point(G.nodes[u]["x"], G.nodes[u]["y"])
        p2 = Point(G.nodes[v]["x"], G.nodes[v]["y"])
        seg = LineString([p1, p2])

        # Narrow candidates with a cheap bbox hit; Shapely 2.x returns indices here
        cand = tree.query(seg)  # optional: predicate="intersects" to prefilter more

        # Normalize candidates to actual polygon geometries
        if len(cand) == 0:
            continue
        if isinstance(cand[0], (int, np.integer)):
            polys = [b_geoms[i] for i in np.asarray(cand).tolist()]
        else:  # Shapely 1.x behavior: geometries directly
            polys = cand

        # Block if segment truly penetrates a polygon (touch-only is OK)
        hit = False
        for poly in polys:
            if seg.crosses(poly) or seg.within(poly) or (seg.intersects(poly) and not seg.touches(poly)):
                hit = True
                break
        if hit:
            to_remove.append((u, v))

    G.remove_edges_from(to_remove)
    return G

# ------------------ COMMS ------------------

def los_ok(pxy, qxy, occluder_tree, occluders_gs):
    seg = LineString([pxy, qxy])
    cand = occluder_tree.query(seg)  # optional: predicate="intersects"

    if len(cand) == 0:
        return True

    # Build the list used to create the tree, in positional order
    occ_geoms = [g for g in occluders_gs.values if g is not None and not g.is_empty]

    # Map indices -> geometries for Shapely 2.x; pass-through for 1.x
    try:
        first = cand[0]
    except Exception:
        return True

    if isinstance(first, (int, np.integer)):
        polys = [occ_geoms[i] for i in np.asarray(cand).tolist()]
    else:
        polys = cand

    for poly in polys:
        if seg.crosses(poly) or seg.within(poly) or seg.touches(poly):
            return False
    return True


def comm_graph_from_positions(positions_xy: List[Tuple[float, float]], occluder_tree: STRtree, occluders: gpd.GeoSeries, R: float) -> nx.Graph:
    Gc = nx.Graph(); n = len(positions_xy); Gc.add_nodes_from(range(n))
    for i in range(n):
        xi, yi = positions_xy[i]
        for j in range(i + 1, n):
            xj, yj = positions_xy[j]
            if math.hypot(xi - xj, yi - yj) <= R and los_ok((xi, yi), (xj, yj), occluder_tree, occluders):
                Gc.add_edge(i, j)
    return Gc

# ------------------ GRID PLACEMENT ------------------

def bbox_and_center_from_graph(G: nx.Graph) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    cx, cy = (xmin+xmax)/2.0, (ymin+ymax)/2.0
    return (xmin, ymin, xmax, ymax), (cx, cy)


def compute_grid_spacing(xmin, ymin, xmax, ymax, rows, cols, coverage, min_spacing, stretch):
    W = max(xmax - xmin, 1.0)
    H = max(ymax - ymin, 1.0)
    sx = max(min_spacing, coverage * W / max(cols-1, 1))
    sy = max(min_spacing, coverage * H / max(rows-1, 1))
    if stretch: return sx, sy
    return min(sx, sy), min(sx, sy)


def make_grid_points(center_xy: Tuple[float, float], rows: int, cols: int, spacing_x: float, spacing_y: float) -> List[Tuple[float, float]]:
    cx, cy = center_xy
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = cx + (c - (cols - 1) / 2.0) * spacing_x
            y = cy + ((rows - 1) / 2.0 - r) * spacing_y
            pts.append((x, y))
    return pts


def try_build_multihop_grid(G_space: nx.Graph,
                            buildings: gpd.GeoDataFrame,
                            occluder_tree: STRtree,
                            occluders: gpd.GeoSeries,
                            comm_R: float,
                            rows: int,
                            cols: int,
                            base_center: Tuple[float, float],
                            sx_init: float,
                            sy_init: float,
                            max_iters: int = MULTIHOP_MAX_ITERS,
                            shrink: float = SPACING_SHRINK,
                            jitter: float = CENTER_JITTER_M) -> Tuple[List[Tuple[float,float]], List[int], float, float, Tuple[float,float], nx.Graph, List[int]]:
    """
    Iteratively adjust spacing (and optionally center) so the farthest pair has a multi-hop path.
    Returns (drones_xy, start_nodes, sx, sy, center, Gc, comm_path_nodes).
    """
    rng = np.random.default_rng(RANDOM_SEED)

    sx, sy = sx_init, sy_init
    center = base_center
    best = None

    for it in range(max_iters):
        # Optionally jitter the center a bit to escape occlusions
        if jitter > 0:
            cx, cy = base_center
            jx = rng.uniform(-jitter, jitter)
            jy = rng.uniform(-jitter, jitter)
            center = (cx + jx, cy + jy)

        # Build grid and snap
        targets_xy = make_grid_points(center, rows, cols, sx, sy)
        start_nodes = snap_unique_nodes_free(G_space, targets_xy, buildings)
        drones_xy = [(G_space.nodes[n]["x"], G_space.nodes[n]["y"]) for n in start_nodes]

        # Comm graph
        Gc = comm_graph_from_positions(drones_xy, occluder_tree, occluders, comm_R)

        # Farthest pair
        far_i = far_j = -1; far_dist = -1.0
        for i in range(len(drones_xy)):
            xi, yi = drones_xy[i]
            for j in range(i+1, len(drones_xy)):
                xj, yj = drones_xy[j]
                d = math.hypot(xi - xj, yi - yj)
                if d > far_dist:
                    far_i, far_j, far_dist = i, j, d

        # If the farthest pair is directly connected, expand spacing a touch to force multi-hop
        if Gc.has_edge(far_i, far_j):
            sx *= 1.05; sy *= 1.05
            # But keep neighbors within R to preserve local connectivity
            sx = min(sx, comm_R * 0.95)
            sy = min(sy, comm_R * 0.95)
            continue

        # If there's a multi-hop path, we are done
        if nx.has_path(Gc, far_i, far_j):
            comm_path_nodes = nx.shortest_path(Gc, far_i, far_j)
            return drones_xy, start_nodes, sx, sy, center, Gc, comm_path_nodes

        # Otherwise shrink spacing to pull neighbors closer and try again
        sx *= shrink; sy *= shrink

    # Fall back: return the last attempt even if no path
    comm_path_nodes = []
    return drones_xy, start_nodes, sx, sy, center, Gc, comm_path_nodes


def snap_unique_nodes_free(G: nx.Graph, targets_xy: List[Tuple[float, float]], buildings_all: gpd.GeoDataFrame) -> List[int]:
    node_ids = list(G.nodes())
    coords = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_ids])
    forbidden = unary_union(buildings_all.geometry) if len(buildings_all) > 0 else None

    taken = set(); chosen = []
    for tx, ty in targets_xy:
        d2 = np.sum((coords - np.array([tx, ty]))**2, axis=1)
        order = np.argsort(d2); pick = None
        for idx in order:
            cand = node_ids[idx]
            if cand in taken: continue
            if forbidden is not None and forbidden.contains(Point(G.nodes[cand]["x"], G.nodes[cand]["y"])):
                continue
            pick = cand; taken.add(cand); break
        if pick is None:
            for idx in order:
                cand = node_ids[idx]
                if forbidden is None or not forbidden.contains(Point(G.nodes[cand]["x"], G.nodes[cand]["y"])):
                    pick = cand; break
        chosen.append(pick)
    return chosen

# ------------------ EXPORTS ------------------

def export_graph_geojson(G: nx.Graph, crs, out_nodes: str, out_edges: str):
    node_rows = [{"id": n, "geometry": Point(d["x"], d["y"])} for n, d in G.nodes(data=True)]
    gpd.GeoDataFrame(node_rows, geometry="geometry", crs=crs).to_file(out_nodes, driver="GeoJSON")

    edges = []
    for u, v, d in G.edges(data=True):
        p1 = Point(G.nodes[u]["x"], G.nodes[u]["y"]) 
        p2 = Point(G.nodes[v]["x"], G.nodes[v]["y"]) 
        edges.append({"u": u, "v": v, "weight": d.get("weight", 0.0), "geometry": LineString([p1, p2])})
    gpd.GeoDataFrame(edges, geometry="geometry", crs=crs).to_file(out_edges, driver="GeoJSON")


def export_comm_edges(xy: List[Tuple[float, float]], Gc: nx.Graph, crs, out_path: str):
    rows = []
    for u, v in Gc.edges():
        p1 = Point(xy[u][0], xy[u][1]); p2 = Point(xy[v][0], xy[v][1])
        rows.append({"u": int(u+1), "v": int(v+1), "geometry": LineString([p1, p2])})
    gpd.GeoDataFrame(rows, geometry="geometry", crs=crs).to_file(out_path, driver="GeoJSON")


def export_path_as_line(coords: List[Tuple[float, float]], crs, out_path: str, props: Optional[Dict]=None):
    if props is None:
        props = {}
    props = {**props, "hop_count": max(0, len(coords)-1)}
    gdf = gpd.GeoDataFrame([{**props, "geometry": LineString([Point(x,y) for x,y in coords])}], geometry="geometry", crs=crs)
    gdf.to_file(out_path, driver="GeoJSON")


def export_geopackage(buildings, roads, G_space, drones_xy, comm_graph, ox_path_coords, comm_path_coords, free_path_coords, pts_crs, out_path):
    print(f"Creating GeoPackage: {out_path}")

    # Buildings
    b = buildings[["h_m", "geometry"]].copy(); b.to_file(out_path, layer="buildings", driver="GPKG")

    # Roads
    roads[["geometry"]].copy().to_file(out_path, layer="roads", driver="GPKG")

    # Roadmap nodes/edges
    node_rows = [{"id": n, "geometry": Point(d["x"], d["y"]) } for n, d in G_space.nodes(data=True)]
    gpd.GeoDataFrame(node_rows, geometry="geometry", crs=pts_crs).to_file(out_path, layer="roadmap_nodes", driver="GPKG")

    edge_rows = []
    for u, v, d in G_space.edges(data=True):
        p1 = Point(G_space.nodes[u]["x"], G_space.nodes[u]["y"]) 
        p2 = Point(G_space.nodes[v]["x"], G_space.nodes[v]["y"]) 
        edge_rows.append({"u": u, "v": v, "weight": d.get("weight", 0.0), "geometry": LineString([p1, p2])})
    gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=pts_crs).to_file(out_path, layer="roadmap_edges", driver="GPKG")

    # Drone points (stationary) + comm edges
    pts_rows = [{"drone_id": i+1, "geometry": Point(x, y)} for i, (x, y) in enumerate(drones_xy)]
    gpd.GeoDataFrame(pts_rows, geometry="geometry", crs=pts_crs).to_file(out_path, layer="drone_points", driver="GPKG")

    comm_rows = []
    for u, v in comm_graph.edges():
        comm_rows.append({"u": u+1, "v": v+1, "geometry": LineString([Point(*drones_xy[u]), Point(*drones_xy[v])])})
    gpd.GeoDataFrame(comm_rows, geometry="geometry", crs=pts_crs).to_file(out_path, layer="comm_edges", driver="GPKG")

    # Paths
    if ox_path_coords is not None:
        gpd.GeoDataFrame([{"geometry": LineString([Point(x,y) for x,y in ox_path_coords])}], geometry="geometry", crs=pts_crs).to_file(out_path, layer="ox_shortest_path", driver="GPKG")
    if comm_path_coords is not None:
        gpd.GeoDataFrame([{"geometry": LineString([Point(x,y) for x,y in comm_path_coords])}], geometry="geometry", crs=pts_crs).to_file(out_path, layer="comm_shortest_path", driver="GPKG")
    if free_path_coords is not None:
        gpd.GeoDataFrame([{"geometry": LineString([Point(x,y) for x,y in free_path_coords])}], geometry="geometry", crs=pts_crs).to_file(out_path, layer="free_shortest_path", driver="GPKG")

    print("GeoPackage export complete.")

# ------------------ UTIL: shortest comm path export ------------------

# Global caches filled in main() so convenience functions can take only drone IDs
LAST_GC: Optional[nx.Graph] = None
LAST_DRONES_XY: Optional[List[Tuple[float, float]]] = None

def _write_comm_path_csv(path_nodes: List[int], drones_xy: List[Tuple[float, float]], out_csv: str):
    rows = []
    for k, nid in enumerate(path_nodes):
        x, y = drones_xy[nid]
        rows.append({"order": k, "drone_id": int(nid), "x": float(x), "y": float(y)})
    df = pd.DataFrame(rows)
    df["hop_count"] = max(0, len(path_nodes) - 1)
    df.to_csv(out_csv, index=False)


# def comm_shortest_path_to_csv(Gc: nx.Graph,
#                               drones_xy: List[Tuple[float, float]],
#                               src_id: int,
#                               dst_id: int,
#                               out_csv: str,
#                               prefer_multihop: bool = True) -> List[int]:
#     """
#     Compute the unweighted shortest communication path (minimum hops) between two drones
#     and export the result to CSV. If `prefer_multihop=True` and the pair are directly
#     connected, try to find an alternative multi-hop path by temporarily removing the
#     direct edge (without mutating the original graph). Falls back to the 1-hop path
#     if no multi-hop alternative exists.
#     """
#     import numpy as np

#     n = len(drones_xy)
#     if not isinstance(src_id, (int, np.integer)) or not isinstance(dst_id, (int, np.integer)):
#         raise TypeError("Drone IDs must be integers.")
#     if not (0 <= src_id < n and 0 <= dst_id < n):
#         raise ValueError(f"Drone IDs must be in [0, {n-1}]. Got {src_id}, {dst_id}.")

#     if src_id == dst_id:
#         _write_comm_path_csv([src_id], drones_xy, out_csv)
#         print(f"[OK] Source equals destination; wrote single-node path to {out_csv}")
#         return [src_id]

#     if not nx.has_path(Gc, src_id, dst_id):
#         pd.DataFrame([], columns=["order", "drone_id", "x", "y", "hop_count"]).to_csv(out_csv, index=False)
#         print(f"[INFO] No communication path between {src_id} and {dst_id}. Wrote empty CSV: {out_csv}")
#         return []

#     # If they are directly connected and we prefer multi-hop, try removing that edge on a copy
#     if prefer_multihop and Gc.has_edge(src_id, dst_id):
#         Gtmp = Gc.copy()
#         Gtmp.remove_edge(src_id, dst_id)
#         if nx.has_path(Gtmp, src_id, dst_id):
#             path_nodes = nx.shortest_path(Gtmp, src_id, dst_id)
#             _write_comm_path_csv(path_nodes, drones_xy, out_csv)
#             print(f"[OK] Saved multi-hop comm path ({len(path_nodes)-1} hops) {src_id}->{dst_id} to {out_csv}")
#             return path_nodes
#         # else: fall back to the direct 1-hop path

#     # Default: standard shortest (may be 1 hop)
#     path_nodes = nx.shortest_path(Gc, src_id, dst_id)
#     _write_comm_path_csv(path_nodes, drones_xy, out_csv)
#     print(f"[OK] Saved comm path ({len(path_nodes)-1} hops) {src_id}->{dst_id} to {out_csv}")
#     return path_nodes


def comm_shortest_path_to_csv(
    Gc: nx.Graph,
    drones_xy: List[Tuple[float, float]],
    src_id_1: int,               # 1-based
    dst_id_1: int,               # 1-based
    out_csv: str,
    *,
    prefer_multihop: bool = True,
    gpkg_path: Optional[str] = None,
    gpkg_layer: str = "comm_shortest_path",
    crs=None
) -> List[int]:
    """
    Compute the shortest-hop path ONLY on the communication graph Gc.
    If Gc has no edges, or no path exists, write an empty CSV and DO NOT write a layer.
    IDs are 1-based on input/output. Optionally writes a line to GPKG.
    """
    n = len(drones_xy)
    if not (1 <= src_id_1 <= n and 1 <= dst_id_1 <= n):
        raise ValueError(f"Drone IDs must be in [1, {n}]. Got {src_id_1}, {dst_id_1}.")

    s0, d0 = src_id_1 - 1, dst_id_1 - 1  # 0-based

    # If comm graph has NO edges, never produce a path
    if Gc.number_of_edges() == 0:
        pd.DataFrame([], columns=["order","drone_id","x","y","hop_count"]).to_csv(out_csv, index=False)
        print("[INFO] comm_edges is empty; no comm path written.")
        return []

    # Trivial
    if s0 == d0:
        x, y = drones_xy[s0]
        pd.DataFrame([{"order":0,"drone_id":src_id_1,"x":float(x),"y":float(y),"hop_count":0}]).to_csv(out_csv, index=False)
        if gpkg_path and crs is not None:
            gpd.GeoDataFrame(
                [{"src_id": src_id_1, "dst_id": dst_id_1, "hop_count": 0,
                  "geometry": LineString([Point(x,y), Point(x,y)])}],
                geometry="geometry", crs=crs
            ).to_file(gpkg_path, layer=gpkg_layer, driver="GPKG")
        return [src_id_1]

    # Must be connected in Gc
    if not nx.has_path(Gc, s0, d0):
        pd.DataFrame([], columns=["order","drone_id","x","y","hop_count"]).to_csv(out_csv, index=False)
        print(f"[INFO] No comm path in Gc between {src_id_1} and {dst_id_1}.")
        return []

    # Prefer multi-hop if a direct link exists (work on a copy)
    Guse = Gc
    if prefer_multihop and Gc.has_edge(s0, d0):
        Gtmp = Gc.copy()
        Gtmp.remove_edge(s0, d0)
        if nx.has_path(Gtmp, s0, d0):
            Guse = Gtmp

    # Shortest-hop path strictly on comm edges
    path0 = nx.shortest_path(Guse, s0, d0)

    # Validate: every hop is an edge in the ORIGINAL comm graph
    for u, v in zip(path0[:-1], path0[1:]):
        if not Gc.has_edge(u, v):
            raise ValueError(f"Path step {(u+1)}->{(v+1)} is not present in comm_edges.")

    # Write CSV (1-based IDs)
    path1 = [p+1 for p in path0]
    rows = []
    for k, nid1 in enumerate(path1):
        x, y = drones_xy[nid1-1]
        rows.append({"order":k, "drone_id":nid1, "x":float(x), "y":float(y)})
    df = pd.DataFrame(rows); df["hop_count"] = len(path1)-1
    df.to_csv(out_csv, index=False)

    # Optional: write line into GPKG
    if gpkg_path and crs is not None:
        line = LineString([Point(*drones_xy[i]) for i in path0])
        gpd.GeoDataFrame(
            [{"src_id": src_id_1, "dst_id": dst_id_1,
              "hop_count": len(path1)-1, "geometry": line}],
            geometry="geometry", crs=crs
        ).to_file(gpkg_path, layer=gpkg_layer, driver="GPKG")

    print(f"[OK] Saved comm path ({len(path1)-1} hops) {src_id_1}->{dst_id_1} to {out_csv}")
    return path1




def main():
    ensure_dirs()

    print("Loading OSM layers…")
    buildings, roads = load_osm_layers(OSM_PATH, CRS_METRIC)

    print("Preparing buildings & occluders…")
    buildings, occluders, occ_tree = prepare_buildings(buildings, ALTITUDE_M)

    print("Building centerlines & sampling roadmap…")
    lines = roads_to_centerlines(roads)
    pts = sample_lines(lines, SAMPLE_EVERY_M)
    G_space = build_roadmap(pts, k_nn=6)
    G_space = prune_nodes_over_tall_buildings(G_space, pts, buildings, ALTITUDE_M)
    G_space = prune_nodes_over_any_buildings(G_space, buildings)  # ensure free-space nodes
    G_space = prune_edges_crossing_buildings(G_space, buildings)  # ensure free-space edges

    print(f"Roadmap nodes: {G_space.number_of_nodes()}, edges: {G_space.number_of_edges()}")

    # ---- Stationary WIDE GRID placement ----
    print("Placing WIDE stationary grid…")
    (xmin, ymin, xmax, ymax), center_bbox = bbox_and_center_from_graph(G_space)
    center = GRID_CENTER_XY if GRID_CENTER_XY is not None else center_bbox

    if AUTO_WIDE_GRID:
        sx, sy = compute_grid_spacing(xmin, ymin, xmax, ymax, GRID_ROWS, GRID_COLS, GRID_COVERAGE, MIN_GRID_SPACING_M, STRETCH_ASPECT)
    else:
        sx = sy = GRID_SPACING_M

    # Clamp to comm range so neighbors can link (facilitates multi-hop chains)
    sx = min(sx, COMM_RANGE_M * 0.95)
    sy = min(sy, COMM_RANGE_M * 0.95)

    if ENSURE_MULTIHOP:
        drones_xy, start_nodes, sx, sy, center, Gc, comm_path_nodes = try_build_multihop_grid(
            G_space, buildings, occ_tree, occluders, COMM_RANGE_M, GRID_ROWS, GRID_COLS, center, sx, sy
        )
    else:
        targets_xy = make_grid_points(center, GRID_ROWS, GRID_COLS, sx, sy)
        start_nodes = snap_unique_nodes_free(G_space, targets_xy, buildings)
        drones_xy = [(G_space.nodes[n]["x"], G_space.nodes[n]["y"]) for n in start_nodes]
        Gc = comm_graph_from_positions(drones_xy, occ_tree, occluders, COMM_RANGE_M)

    # Cache for convenience wrappers
        # Cache for convenience wrappers (optional)
        # Cache (optional)
    global LAST_GC, LAST_DRONES_XY
    LAST_GC = Gc
    LAST_DRONES_XY = drones_xy

    # ------------ SELECTED PAIR (1-based) ------------
    parser = argparse.ArgumentParser(description="Export comm shortest path between two drones by 1-based ID")
    parser.add_argument("--src-id", type=int, required=True)
    parser.add_argument("--dst-id", type=int, required=True)
    parser.add_argument("--no-multihop", action="store_true",
                        help="Do not prefer multi-hop when a direct link exists")
    args = parser.parse_args()

    src_id = int(args.src_id)    # 1-based
    dst_id = int(args.dst_id)    # 1-based
    prefer_multihop = True and (not args.no_multihop)

    n = len(drones_xy)
    if not (1 <= src_id <= n and 1 <= dst_id <= n):
        raise ValueError(f"Drone IDs must be in [1,{n}], got src={src_id}, dst={dst_id}")

    # Communication path between chosen 1-based IDs (returns 1-based node IDs)
    # comm_csv = os.path.join(OUT_DIR, f"comm_path_{src_id}_{dst_id}.csv")
    # comm_path_nodes_1 = comm_shortest_path_to_csv(Gc, drones_xy, src_id, dst_id, comm_csv,
    #                                               prefer_multihop=prefer_multihop)
    # comm_path_coords = [drones_xy[k - 1] for k in comm_path_nodes_1] if comm_path_nodes_1 else None

    comm_csv = os.path.join(OUT_DIR, f"comm_path_{src_id}_{dst_id}.csv")
    gpkg_path = os.path.join(OUT_DIR, "geoai_stationary_wide.gpkg")

    path_ids_1 = comm_shortest_path_to_csv(
        Gc, drones_xy, src_id, dst_id, comm_csv,
        prefer_multihop=(not args.no_multihop),
        gpkg_path=gpkg_path, gpkg_layer="comm_shortest_path", crs=pts.crs
    )
    comm_path_coords = [drones_xy[i - 1] for i in path_ids_1] if path_ids_1 else None
    if Gc.number_of_edges() == 0:
        comm_path_coords = None
    # Optional: OSMnx road path between the same 1-based IDs
    ox_path_coords = None
    G_ox = try_load_osmnx_graph(OSM_PATH)
    if G_ox is not None:
        try:
            s0, d0 = src_id - 1, dst_id - 1
            ni = ox.distance.nearest_nodes(G_ox, drones_xy[s0][0], drones_xy[s0][1])
            nj = ox.distance.nearest_nodes(G_ox, drones_xy[d0][0], drones_xy[d0][1])
            try:
                sp_nodes = ox.graph.shortest_path(G_ox, ni, nj, weight="length")
            except AttributeError:
                sp_nodes = ox.shortest_path(G_ox, ni, nj, weight="length")
            xs = [G_ox.nodes[n]["x"] for n in sp_nodes]
            ys = [G_ox.nodes[n]["y"] for n in sp_nodes]
            ox_path_coords = list(zip(xs, ys))
        except Exception as e:
            print(f"[WARN] OSMnx shortest path failed: {e}")
    else:
        print("[INFO] Skipping OSMnx shortest path (need .osm/.xml input).")

    # Optional: free-space roadmap path between the same IDs
    free_path_coords = None
    s_node = start_nodes[src_id - 1]
    t_node = start_nodes[dst_id - 1]
    try:
        sp_nodes_free = nx.shortest_path(G_space, s_node, t_node, weight="weight")
        free_path_coords = [(G_space.nodes[n]["x"], G_space.nodes[n]["y"]) for n in sp_nodes_free]
    except nx.NetworkXNoPath:
        print("[INFO] No free-space path exists between selected pair (after pruning obstacles).")

    # ------------ EXPORTS ------------
    print("Exporting layers…")

    # ... (your existing GeoJSON exports with the 1-based tweaks above) ...

    if comm_path_coords is not None:
        export_path_as_line(
            comm_path_coords, pts.crs,
            os.path.join(OUT_DIR, f"comm_shortest_path_{src_id}_{dst_id}.geojson"),
            {"type": "comm_path", "src_id": src_id, "dst_id": dst_id}
        )

    # Bundle all layers into GPKG (includes the comm_shortest_path geometry)
    export_geopackage(
        buildings, roads, G_space, drones_xy, Gc,
        ox_path_coords, comm_path_coords, free_path_coords,
        pts.crs, os.path.join(OUT_DIR, "geoai_stationary_wide.gpkg")
    )




if __name__ == "__main__":
    main()

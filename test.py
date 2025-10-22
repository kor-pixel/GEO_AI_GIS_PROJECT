# geoai_swarm_connectivity.py
# Limetoad 🐸 — GeoAI midterm scaffold:
# - Load .osm buildings/streets
# - Estimate building heights; build occluders at chosen altitude z
# - Build a roadmap movement graph (sample along street centrelines)
# - LOS + R connectivity model using STRtree
# - Prioritized A* multi-drone planning + simple connectivity guard
# - Step-by-step simulation + exports (GeoJSON/CSV) + quick plots
#
# Requirements:
#   pip install pyrosm osmnx geopandas shapely rtree pyproj networkx pandas matplotlib
#
# Notes:
#  - Choose a local metric CRS for Seoul (EPSG:5186). Adjust if you’re elsewhere.
#  - This is a teaching scaffold: readable > hyper-optimized.

import os
import math
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox


import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree
from pyproj import CRS
from pyrosm import OSM

# ------------------------- CONFIG -------------------------
OSM_PATH = "map.osm"      # <-- put your .osm filename here
OUT_DIR  = "out"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
CRS_METRIC = "EPSG:5186"         # Korea Transverse Mercator (meters). Change if needed.

# Simulation / model
ALTITUDE_M = 60.0                # fixed altitude for all drones (m)
COMM_RANGE_M = 300.0             # max comms link distance (m)
SAMPLE_EVERY_M = 25.0            # roadmap sampling interval along streets (m)
MAX_STEPS = 200                  # sim steps cap
CONNECTIVITY_REQUIRED = True     # enforce connectivity guard (attempt repair)
RANDOM_SEED = 42

# Drones (define start/goal by nearest roadmap nodes after graph build)
DRONE_COUNT = 4

# ----------------------------------------------------------

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def load_osm_layers(osm_path: str, crs_metric: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load buildings and roads from either:
      - .osm.pbf  -> via pyrosm
      - .osm/.xml -> via osmnx (features_from_xml + graph_from_xml)
    Returns (buildings_gdf, roads_gdf) in metric CRS.
    """
    import osmnx as ox
    ext = os.path.splitext(osm_path)[1].lower()

    if ext == ".pbf":
        from pyrosm import OSM
        osm = OSM(osm_path)
        buildings = osm.get_data_by_custom_criteria(
            custom_filter={"building": True},
            osm_keys_to_keep=["building", "height", "levels", "name"]
        )
        if buildings is None or len(buildings) == 0:
            raise ValueError("No buildings found in PBF.")

        roads = osm.get_network(network_type="drive")
        if roads is None or len(roads) == 0:
            raise ValueError("No road network found in PBF.")

        buildings = buildings.to_crs(crs_metric)
        roads = roads.to_crs(crs_metric)
        return buildings.set_geometry("geometry"), roads.set_geometry("geometry")

    elif ext in {".osm", ".xml"}:
        # Buildings from XML
        buildings = ox.features_from_xml(osm_path, tags={"building": True})
        if buildings is None or len(buildings) == 0:
            raise ValueError("No buildings found in OSM XML.")
        buildings = buildings[~buildings.geometry.is_empty].copy()

        # Roads from XML (graph edges). OSMnx 2.x: no network_type here.
        try:
            G = ox.graph_from_xml(osm_path, simplify=True)
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            edges = edges[~edges.geometry.is_empty].copy()
        except Exception:
            # Fallback: pull highway features directly if graph load has trouble
            edges = ox.features_from_xml(osm_path, tags={"highway": True})
            # keep only LineString/MultiLineString
            edges = edges[edges.geometry.type.isin(["LineString", "MultiLineString"])].copy()

        # Project both to metric CRS
        buildings = buildings.to_crs(crs_metric)
        edges = edges.to_crs(crs_metric)

        roads = gpd.GeoDataFrame(edges[["geometry"]].copy(), geometry="geometry", crs=edges.crs)
        return buildings.set_geometry("geometry"), roads

    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .osm, .xml, or .osm.pbf")


def estimate_height_m(row) -> float:
    """Estimate building height from tags; fallback defaults."""
    # height may be like '12', '12.0', or with 'm'—be forgiving
    h = row.get("height")
    lv = row.get("levels")
    try:
        if h is not None and not pd.isna(h):
            s = str(h).lower().replace("m", "").strip()
            return float(s)
    except Exception:
        pass
    try:
        if lv is not None and not pd.isna(lv):
            return 3.3 * float(lv)
    except Exception:
        pass
    return 12.0  # default ~4 floors

def prepare_buildings(buildings: gpd.GeoDataFrame, altitude_m: float) -> Tuple[gpd.GeoDataFrame, gpd.GeoSeries, STRtree]:
    """Add height field; build occluders at given altitude; prep STRtree."""
    b = buildings.copy()
    b["h_m"] = b.apply(estimate_height_m, axis=1)
    # Occluders are buildings taller than or equal to drone altitude
    occluders = b[b["h_m"] >= altitude_m].geometry
    tree = STRtree(list(occluders.values))
    return b, occluders, tree

def roads_to_centerlines(roads: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Extract merged centerlines from road geometries."""
    # roads geometries may be MultiLineString/LineString; merge into cleaner set
    merged = linemerge(unary_union(roads.geometry))
    if isinstance(merged, LineString):
        return gpd.GeoSeries([merged], crs=roads.crs)
    elif hasattr(merged, "geoms"):
        return gpd.GeoSeries(list(merged.geoms), crs=roads.crs)
    else:
        return gpd.GeoSeries([], crs=roads.crs)

def sample_lines(lines: gpd.GeoSeries, every_m: float) -> gpd.GeoDataFrame:
    """Sample points every N meters along line centerlines."""
    pts = []
    for ln in lines:
        if ln.length < 1.0:
            continue
        d = 0.0
        while d <= ln.length:
            pts.append(Point(ln.interpolate(d)))
            d += every_m
    gdf = gpd.GeoDataFrame(geometry=pts, crs=lines.crs)
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf

def build_roadmap(points: gpd.GeoDataFrame, k_nn: int = 6) -> nx.Graph:
    """Build a sparse roadmap by connecting k nearest neighbors within a radius."""
    from sklearn.neighbors import NearestNeighbors  # lightweight dependency; if missing, pip install scikit-learn
    coords = np.vstack([points["x"].values, points["y"].values]).T
    nbrs = NearestNeighbors(n_neighbors=min(k_nn, len(points)), algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    G = nx.Graph()
    for i, (x, y) in enumerate(coords):
        G.add_node(i, x=float(x), y=float(y))
    for i, row in enumerate(indices):
        for j_idx, j in enumerate(row):
            if j == i:
                continue
            d = float(distances[i][j_idx])
            # add undirected edge
            G.add_edge(i, j, weight=d)
    return G

def prune_nodes_over_tall_buildings(G: nx.Graph, points: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, altitude_m: float) -> nx.Graph:
    """Remove roadmap nodes whose vertical projection lies inside tall buildings (>= z)."""
    tall = buildings[buildings["h_m"] >= altitude_m]
    if len(tall) == 0:
        return G
    tall_union = unary_union(tall.geometry)
    to_remove = []
    for n, data in G.nodes(data=True):
        p = Point(data["x"], data["y"])
        if tall_union.contains(p):
            to_remove.append(n)
    G.remove_nodes_from(to_remove)
    # remove orphaned edges automatically
    return G

def export_graph_geojson(G: nx.Graph, crs, out_nodes: str, out_edges: str):
    """Export graph nodes and edges to GeoJSON for QGIS."""
    nodes = []
    for n, d in G.nodes(data=True):
        nodes.append({"id": n, "geometry": Point(d["x"], d["y"])})
    gdf_nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs=crs)
    gdf_nodes.to_file(out_nodes, driver="GeoJSON")

    edges = []
    for u, v, d in G.edges(data=True):
        p1 = Point(G.nodes[u]["x"], G.nodes[u]["y"])
        p2 = Point(G.nodes[v]["x"], G.nodes[v]["y"])
        edges.append({"u": u, "v": v, "weight": d.get("weight", 0.0), "geometry": LineString([p1, p2])})
    gdf_edges = gpd.GeoDataFrame(edges, geometry="geometry", crs=crs)
    gdf_edges.to_file(out_edges, driver="GeoJSON")

def los_ok(pxy: Tuple[float, float], qxy: Tuple[float, float], occluder_tree: STRtree, occluders: gpd.GeoSeries) -> bool:
    """2D LOS test treating tall building polygons as occluders."""
    seg = LineString([pxy, qxy])
    # Quick spatial prefilter
    candidates = occluder_tree.query(seg)
    for poly in candidates:
        # crosses or lies within/along polygon blocks LOS
        if seg.crosses(poly) or seg.within(poly) or seg.touches(poly):
            return False
    return True

def comm_graph_from_positions(positions_xy: List[Tuple[float, float]],
                              occluder_tree: STRtree,
                              occluders: gpd.GeoSeries,
                              R: float) -> nx.Graph:
    """Build communication graph among drones based on LOS + range."""
    Gc = nx.Graph()
    n = len(positions_xy)
    Gc.add_nodes_from(range(n))
    for i in range(n):
        xi, yi = positions_xy[i]
        for j in range(i + 1, n):
            xj, yj = positions_xy[j]
            if math.hypot(xi - xj, yi - yj) <= R and los_ok((xi, yi), (xj, yj), occluder_tree, occluders):
                Gc.add_edge(i, j)
    return Gc

def nearest_node(G: nx.Graph, xy: Tuple[float, float]) -> int:
    """Return nearest graph node to a coordinate."""
    best = None
    bx, by = xy
    bestd = 1e18
    for n, d in G.nodes(data=True):
        dd = (d["x"] - bx) ** 2 + (d["y"] - by) ** 2
        if dd < bestd:
            bestd = dd
            best = n
    return best

def astar(G: nx.Graph, s: int, t: int) -> List[int]:
    """A* shortest path over movement graph using euclidean heuristic."""
    def h(a, b):
        ax, ay = G.nodes[a]["x"], G.nodes[a]["y"]
        bx, by = G.nodes[b]["x"], G.nodes[b]["y"]
        return math.hypot(ax - bx, ay - by)

    try:
        return nx.astar_path(G, s, t, heuristic=h, weight="weight")
    except nx.NetworkXNoPath:
        return [s]  # fallback: stay put

@dataclass
class Drone:
    did: int
    path: List[int]     # node ids along planned path
    idx: int            # current index into path

    def current(self) -> int:
        return self.path[self.idx]

    def at_goal(self) -> bool:
        return self.idx >= len(self.path) - 1

def resolve_collisions(proposed: List[int]) -> List[int]:
    """Prevent multiple drones occupying the same node: lower id wins, others loiter."""
    final = proposed.copy()
    seen = {}
    for i, node in enumerate(proposed):
        if node in seen:
            # conflict: loiter (set back to None for decision later)
            final[i] = None
        else:
            seen[node] = i
    # Loiter conflicted by making them not move (will be set by caller)
    return final

def simulate(drones: List[Drone],
             G: nx.Graph,
             occluder_tree: STRtree,
             occluders: gpd.GeoSeries,
             comm_R: float,
             max_steps: int = 200,
             enforce_connectivity: bool = True) -> Dict:
    """Run the time-stepped simulation with a simple connectivity guard."""
    rng = np.random.default_rng(RANDOM_SEED)
    history = []   # list of dicts with per-step positions + metrics
    step = 0

    while step < max_steps:
        step += 1

        # Propose next nodes (follow path if not at goal)
        proposals = []
        for d in drones:
            if d.at_goal():
                proposals.append(d.current())
            else:
                proposals.append(d.path[d.idx + 1])

        # Resolve node collisions: losers loiter
        resolved = resolve_collisions(proposals)
        for i, nd in enumerate(resolved):
            if nd is None:
                # loiter: stay on current node
                resolved[i] = drones[i].current()

        # Preview positions after move
        preview_nodes = resolved
        preview_xy = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in preview_nodes]
        Gc_prev = comm_graph_from_positions(preview_xy, occluder_tree, occluders, comm_R)
        connected = nx.is_connected(Gc_prev) if len(preview_xy) > 1 else True

        # Enforce connectivity by loitering one drone if that can fix it
        if enforce_connectivity and not connected:
            # try each drone: force loiter and see if that restores connectivity
            fixed = False
            for i in range(len(drones)):
                test_nodes = preview_nodes.copy()
                test_nodes[i] = drones[i].current()  # force loiter
                test_xy = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in test_nodes]
                Gc_test = comm_graph_from_positions(test_xy, occluder_tree, occluders, comm_R)
                if nx.is_connected(Gc_test):
                    preview_nodes = test_nodes
                    connected = True
                    fixed = True
                    # mark i as loiterer by not advancing its idx
                    break
            # If cannot fix, accept disconnected step but mark it
            if not fixed:
                pass

        # Commit moves (advance idx where moved)
        for i, d in enumerate(drones):
            if preview_nodes[i] != d.current():
                # moved along path (advance index)
                if d.idx + 1 < len(d.path) and d.path[d.idx + 1] == preview_nodes[i]:
                    d.idx += 1
                else:
                    # in rare cases (repair), we might loiter — no idx change
                    pass

        # Log metrics
        nodes_now = [dr.current() for dr in drones]
        xy_now = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in nodes_now]
        Gc = comm_graph_from_positions(xy_now, occluder_tree, occluders, comm_R)
        comps = list(nx.connected_components(Gc))
        hop_lengths = []
        if len(drones) > 0:
            # pick drone 0 as sink for demo; compute shortest-path hops in comm graph
            for j in range(len(drones)):
                try:
                    hop_lengths.append(nx.shortest_path_length(Gc, 0, j))
                except nx.NetworkXNoPath:
                    hop_lengths.append(np.inf)

        history.append({
            "step": step,
            "connected": int(nx.is_connected(Gc)) if len(Gc) > 0 else 1,
            "components": len(comps),
            "avg_hops_to_sink": float(np.mean([h for h in hop_lengths if np.isfinite(h)])) if any(np.isfinite(hop_lengths)) else np.inf,
            "positions": xy_now,
            "node_ids": nodes_now
        })

        # Done if all at goal
        if all(d.at_goal() for d in drones):
            break

    return {"history": history}

def export_buildings(buildings: gpd.GeoDataFrame, out_path: str):
    buildings[["h_m", "geometry"]].to_file(out_path, driver="GeoJSON")

def export_sim_positions(history: Dict, crs, out_path: str):
    """Export per-step drone positions as GeoJSON points with properties."""
    feats = []
    for rec in history["history"]:
        step = rec["step"]
        for i, (x, y) in enumerate(rec["positions"]):
            feats.append({
                "type": "Feature",
                "geometry": mapping(Point(x, y)),
                "properties": {
                    "step": int(step),
                    "drone_id": int(i)
                }
            })
    fc = {"type": "FeatureCollection", "features": feats}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)

def export_metrics_csv(history: Dict, out_csv: str):
    rows = []
    for rec in history["history"]:
        rows.append({
            "step": rec["step"],
            "connected": rec["connected"],
            "components": rec["components"],
            "avg_hops_to_sink": rec["avg_hops_to_sink"]
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def quick_plots(history: Dict):
    steps = [r["step"] for r in history["history"]]
    connected = [r["connected"] for r in history["history"]]
    comps = [r["components"] for r in history["history"]]
    hops = [r["avg_hops_to_sink"] for r in history["history"]]

    plt.figure()
    plt.plot(steps, connected)
    plt.xlabel("Step"); plt.ylabel("Connected (1=yes)")
    plt.title("Connectivity over time")
    plt.savefig(os.path.join(PLOTS_DIR, "connectivity.png"), bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(steps, comps)
    plt.xlabel("Step"); plt.ylabel("# Components")
    plt.title("Components over time")
    plt.savefig(os.path.join(PLOTS_DIR, "components.png"), bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(steps, hops)
    plt.xlabel("Step"); plt.ylabel("Avg hops to sink (finite only)")
    plt.title("Hop distance over time")
    plt.savefig(os.path.join(PLOTS_DIR, "hops.png"), bbox_inches="tight"); plt.close()

def demo_start_goals(G: nx.Graph, k: int) -> Tuple[List[int], List[int]]:
    """Pick k start/goal pairs spread over the graph."""
    rng = np.random.default_rng(RANDOM_SEED)
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    # ensure distinct starts/goals
    starts = nodes[:k]
    goals = nodes[-k:]
    return starts, goals


# ---------- EXTRA EXPORTS & VISUALS ----------

def export_street_buffer(roads_gdf: gpd.GeoDataFrame, buffer_m: float, out_path: str):
    """Make a flight corridor buffer around road centerlines for visualization."""
    buf = roads_gdf.copy()
    buf["geometry"] = buf.geometry.buffer(buffer_m)
    # dissolve to a single corridor polygon layer (optional)
    buf = buf.dissolve()  # single multi-poly
    buf.to_file(out_path, driver="GeoJSON")

def export_paths_as_lines(G: nx.Graph, drones: List[Drone], out_path: str):
    """Export each drone's planned path as a LineString."""
    feats = []
    for d in drones:
        coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in d.path]
        line = LineString(coords)
        feats.append({"drone_id": d.did, "geometry": line, "path_len": len(d.path)})
    gdf = gpd.GeoDataFrame(feats, geometry="geometry", crs=CRS_METRIC)
    gdf.to_file(out_path, driver="GeoJSON")

def export_positions_timeaware(history: Dict, crs, out_path: str, step_seconds: int = 1):
    """
    Export points with a 'time' field so QGIS Temporal Controller can animate.
    """
    base_ts = pd.Timestamp("2025-01-01T00:00:00")  # arbitrary
    feats = []
    for rec in history["history"]:
        t = base_ts + pd.Timedelta(seconds=step_seconds * (rec["step"] - 1))
        for i, (x, y) in enumerate(rec["positions"]):
            feats.append({
                "type": "Feature",
                "geometry": mapping(Point(x, y)),
                "properties": {
                    "step": int(rec["step"]),
                    "time": t.isoformat(),
                    "drone_id": int(i)
                }
            })
    fc = {"type": "FeatureCollection", "features": feats}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)

def plot_2d_map(buildings: gpd.GeoDataFrame,
                roads: gpd.GeoDataFrame,
                G: nx.Graph,
                drones: List[Drone],
                out_png: str,
                add_basemap: bool = True):
    """
    Static 2D figure: buildings, road buffer hint, roadmap graph, and planned paths.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))

    # buildings footprint, colored by height
    buildings.plot(ax=ax, column="h_m", alpha=0.5, linewidth=0, legend=True)

    # roadmap graph
    # draw edges
    for u, v in G.edges():
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        ax.plot([x1, x2], [y1, y2], linewidth=0.5)

    # paths
    colors = {}
    for d in drones:
        xs = [G.nodes[n]["x"] for n in d.path]
        ys = [G.nodes[n]["y"] for n in d.path]
        hnd, = ax.plot(xs, ys, linewidth=2)
        colors[d.did] = hnd.get_color()
        # start/goal markers
        ax.scatter(xs[0], ys[0], s=40, marker="o", edgecolor="k")
        ax.scatter(xs[-1], ys[-1], s=40, marker="*", edgecolor="k")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("2D overview: buildings, roadmap, and planned paths")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    # Optional basemap (requires internet). Must reproject to EPSG:3857.
    if add_basemap:
        try:
            import contextily as cx
            # Build a quick GeoDataFrame bbox in 3857
            xmin, ymin, xmax, ymax = buildings.total_bounds
            gbox = gpd.GeoDataFrame(geometry=[Polygon([(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)])], crs=buildings.crs)
            gbox_3857 = gbox.to_crs(3857)
            ax.set_xlim(gbox_3857.total_bounds[0], gbox_3857.total_bounds[2])
            ax.set_ylim(gbox_3857.total_bounds[1], gbox_3857.total_bounds[3])
            cx.add_basemap(ax, crs=buildings.crs, source=cx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"[plot_2d_map] Basemap skipped ({e})")

    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)

def plot_3d_scene(buildings: gpd.GeoDataFrame,
                  drones: List[Drone],
                  G: nx.Graph,
                  altitude_m: float,
                  out_png: str):
    """
    Simple 3D: extruded buildings + drone paths at fixed altitude.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Buildings: extrude polygon footprints to height
    for _, row in buildings.iterrows():
        poly = row.geometry
        if poly.is_empty: continue
        if poly.geom_type == "MultiPolygon":
            polys = list(poly.geoms)
        elif poly.geom_type == "Polygon":
            polys = [poly]
        else:
            continue
        h = float(row["h_m"])
        for p in polys:
            xs, ys = p.exterior.coords.xy
            # create vertical walls
            verts = []
            for i in range(len(xs)-1):
                x1, y1 = xs[i], ys[i]
                x2, y2 = xs[i+1], ys[i+1]
                verts.append([(x1,y1,0), (x2,y2,0), (x2,y2,h), (x1,y1,h)])
            pc = Poly3DCollection(verts, alpha=0.15)
            ax.add_collection3d(pc)

    # Drone paths at fixed z
    for d in drones:
        xs = [G.nodes[n]["x"] for n in d.path]
        ys = [G.nodes[n]["y"] for n in d.path]
        zs = [altitude_m] * len(xs)
        ax.plot(xs, ys, zs, linewidth=2)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("3D scene: extruded buildings + drone paths")
    ax.view_init(elev=35, azim=-60)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)

def animate_topdown(history: Dict, G: nx.Graph, buildings: gpd.GeoDataFrame, out_gif: str):
    """
    Make a simple top-down animation GIF of drone positions.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    steps = [r["step"] for r in history["history"]]
    pos_by_step = [r["positions"] for r in history["history"]]

    fig, ax = plt.subplots(figsize=(7,7))
    buildings.plot(ax=ax, column="h_m", alpha=0.35, linewidth=0)

    # roadmap edges (faint)
    for u, v in G.edges():
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        ax.plot([x1, x2], [y1, y2], linewidth=0.3)

    scat = ax.scatter([], [], s=40)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Drone positions over time")

    xs = [G.nodes[n]["x"] for n in G.nodes()]
    ys = [G.nodes[n]["y"] for n in G.nodes()]
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))

    def init():
        scat.set_offsets(np.empty((0,2)))
        return (scat,)

    def update(frame_idx):
        coords = np.array(pos_by_step[frame_idx])
        scat.set_offsets(coords)
        ax.set_xlabel(f"Step {steps[frame_idx]}")
        return (scat,)

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(steps), interval=200, blit=True)
    anim.save(out_gif, writer="pillow", dpi=150)
    plt.close(fig)


def export_geopackage(buildings, roads, G_space, drones, sim, pts, out_path):
    """Bundle all key layers into a single .gpkg file for easy import into QGIS."""
    import geopandas as gpd
    from shapely.geometry import Point, LineString

    print(f"Creating GeoPackage: {out_path}")

    # Buildings
    buildings_gdf = buildings[["h_m", "geometry"]].copy()
    buildings_gdf.to_file(out_path, layer="buildings", driver="GPKG")

    # Roads
    roads_gdf = roads[["geometry"]].copy()
    roads_gdf.to_file(out_path, layer="roads", driver="GPKG")

    # Roadmap nodes
    node_rows = []
    for n, d in G_space.nodes(data=True):
        node_rows.append({"id": n, "geometry": Point(d["x"], d["y"])})
    gpd.GeoDataFrame(node_rows, geometry="geometry", crs=pts.crs).to_file(out_path, layer="roadmap_nodes", driver="GPKG")

    # Roadmap edges
    edge_rows = []
    for u, v, d in G_space.edges(data=True):
        p1 = Point(G_space.nodes[u]["x"], G_space.nodes[u]["y"])
        p2 = Point(G_space.nodes[v]["x"], G_space.nodes[v]["y"])
        edge_rows.append({"u": u, "v": v, "weight": d.get("weight", 0.0), "geometry": LineString([p1, p2])})
    gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=pts.crs).to_file(out_path, layer="roadmap_edges", driver="GPKG")

    # Drone paths
    path_rows = []
    for d in drones:
        coords = [(G_space.nodes[n]["x"], G_space.nodes[n]["y"]) for n in d.path]
        path_rows.append({"drone_id": d.did, "path_len": len(d.path), "geometry": LineString(coords)})
    gpd.GeoDataFrame(path_rows, geometry="geometry", crs=pts.crs).to_file(out_path, layer="drone_paths", driver="GPKG")

    # Drone positions (time-aware)
    pos_rows = []
    for rec in sim["history"]:
        step = rec["step"]
        for i, (x, y) in enumerate(rec["positions"]):
            pos_rows.append({"drone_id": i, "step": step, "geometry": Point(x, y)})
    gpd.GeoDataFrame(pos_rows, geometry="geometry", crs=pts.crs).to_file(out_path, layer="drone_positions", driver="GPKG")

    print("GeoPackage export complete.")


def main():
    ensure_dirs()

    print("Loading OSM...")
    buildings, roads = load_osm_layers(OSM_PATH, CRS_METRIC)

    print("Preparing buildings & occluders...")
    buildings, occluders, occ_tree = prepare_buildings(buildings, ALTITUDE_M)
    export_buildings(buildings, os.path.join(OUT_DIR, "buildings.geojson"))

    print("Building centerlines & sampling roadmap...")
    lines = roads_to_centerlines(roads)
    pts = sample_lines(lines, SAMPLE_EVERY_M)
    G_space = build_roadmap(pts, k_nn=6)
    G_space = prune_nodes_over_tall_buildings(G_space, pts, buildings, ALTITUDE_M)
    export_graph_geojson(G_space, pts.crs, os.path.join(OUT_DIR, "roadmap_nodes.geojson"), os.path.join(OUT_DIR, "roadmap_edges.geojson"))
    print(f"Roadmap nodes: {G_space.number_of_nodes()}, edges: {G_space.number_of_edges()}")

    # Define start/goal nodes for DRONE_COUNT drones (demo). You can set your own.
    starts, goals = demo_start_goals(G_space, DRONE_COUNT)

    # Plan paths independently via A*
    print("Planning individual paths with A*...")
    drones: List[Drone] = []
    for i in range(DRONE_COUNT):
        s, g = starts[i], goals[i]
        p = astar(G_space, s, g)
        drones.append(Drone(did=i, path=p, idx=0))
        print(f"Drone {i}: start {s} -> goal {g}, path_len={len(p)}")

    # Simulate
    print("Simulating...")
    sim = simulate(drones, G_space, occ_tree, occluders, COMM_RANGE_M, MAX_STEPS, CONNECTIVITY_REQUIRED)

    # Core exports (already in your script)
    print("Exporting results...")
    export_sim_positions(sim, pts.crs, os.path.join(OUT_DIR, "sim_positions.geojson"))
    export_metrics_csv(sim, os.path.join(OUT_DIR, "metrics.csv"))
    quick_plots(sim)  # connectivity/components/hops PNGs in out/plots/

    # === EXTRA OUTPUTS ===
    # 2D street corridor buffer for visualization (10 m each side; tweak as needed)
    roads_edges = roads  # from load_osm_layers()
    export_street_buffer(roads_edges, buffer_m=10.0, out_path=os.path.join(OUT_DIR, "street_buffer.geojson"))

    # Per-drone path lines
    export_paths_as_lines(G_space, drones, os.path.join(OUT_DIR, "paths.geojson"))

    # Time-aware positions for QGIS Temporal Controller
    export_positions_timeaware(sim, pts.crs, os.path.join(OUT_DIR, "sim_positions_time.geojson"), step_seconds=1)

    # Static 2D overview (with optional basemap)
    plot_2d_map(buildings, roads_edges, G_space, drones, os.path.join(PLOTS_DIR, "overview_2d.png"), add_basemap=True)

    # Simple 3D rendering
    plot_3d_scene(buildings, drones, G_space, ALTITUDE_M, os.path.join(PLOTS_DIR, "scene_3d.png"))

    # Animated GIF (top-down)
    animate_topdown(sim, G_space, buildings, os.path.join(PLOTS_DIR, "run.gif"))

    print("Done. Open the GeoJSONs in QGIS and the PNGs/GIF in out/plots/.")

    # Bundle all outputs into one GeoPackage for QGIS
    export_geopackage(buildings, roads_edges, G_space, drones, sim, pts, os.path.join(OUT_DIR, "geoai_project.gpkg"))



if __name__ == "__main__":
    main()

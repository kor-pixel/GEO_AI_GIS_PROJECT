import heapq
import random
import csv
import math
import json
from pathlib import Path
import os
import numpy as np
import cupy as cp  # optional GPU

from datetime import datetime, timedelta

# from uav_rl_routing_torch import (
#     UAVRoutingEnvTorch,
#     train_dqn_routing,
#     extract_greedy_path_dqn,
# )

# from uav_rl_routing_with_gps import (
#     UAVRoutingEnvTorch,
#     train_dqn_routing,
#     extract_greedy_path_dqn,
# )


# from uav_rl_routing_with_gps_prioritzed_replay import (
#     UAVRoutingEnvTorch,
#     train_dqn_routing,
#     extract_greedy_path_dqn,
# )

from uav_gnn_rl import (
    UAVRoutingEnvTorch,
    train_dqn_routing,
    extract_greedy_path_dqn,
    build_graph_from_env, 
    graph_features_to_tables,
    GraphDuelingDQN
)

# =============================
# CONFIGURATION
# =============================

BASE_TIME = datetime.now()              # start of simulation
TIME_DELTA = timedelta(seconds=2)       # delta between time slots

# ---- Base map (GeoJSON) ----
BASE_GEOJSON_PATH = "geo_with_centers.geojson"
OUTPUT_GEOJSON_PATH = "uav_sim.geojson"

# ---- Simulation parameters ----
NUM_DRONES = 40
NUM_TIMESLOTS = 10
STEP_DEG = 100
COMM_RANGE_DEG = 1000
MAX_HOPS = 10


REQUIRED_BW_1080P_MBPS = 6.0           # minimal bandwidth per hop. good: 3.0Mbps
MAX_DELAY_1080P_MS = 200.0             # not used directly here (for later QoS checks). Good: 200ms
MAX_LOSS_1080P = 0.1                  # not used directly here
MAX_JITTER_1080P_MS = 25.0             # not used directly here (for later QoS checks)


HEX_CELLS = {}   # id -> {"lat": ..., "lon": ..., "row": ..., "col": ..., "hexscore": ...}
HEX_ADJ = {}     # id -> [neighbor_ids]
ALLOWED_IDS = [] # ids where HexScore is 1 or 2

DRONE_BOX_IDS = [45, 320, 313, 99]
BS1_ID = 23
BS2_ID = 334

# Base station coordinates (filled AFTER loading)
BS1_COORD = None
BS2_COORD = None

# Bounding box values (computed from all hex centers)
MIN_LAT = None
MAX_LAT = None
MIN_LON = None
MAX_LON = None

# ---- Output CSV files ----
POSITIONS_CSV = "positions.csv"
PATHS_CSV = "paths.csv"

# ---- GPU / Hybrid settings ----
USE_GPU = True   # True => use CuPy if available


# =============================
# HELPER: LOAD HEX GRID
# =============================

def load_hex_grid(geojson_path):
    """
    Loads all hex features from the GeoJSON.
    Extracts Center_X / Center_Y (as lon, lat), HexScore, row_index, col_index.
    Computes bounding box from all hex centers.
    Also sets BS1_COORD and BS2_COORD based on BS1_ID and BS2_ID.
    """

    global HEX_CELLS, HEX_ADJ, ALLOWED_IDS
    global MIN_LAT, MAX_LAT, MIN_LON, MAX_LON
    global BS1_COORD, BS2_COORD

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])

    HEX_CELLS = {}
    ALLOWED_IDS = []
    all_lats, all_lons = [], []

    for feat in features:
        props = feat.get("properties", {})
        cid = props.get("id")
        if cid is None:
            continue

        cx = props.get("Center_X")
        cy = props.get("Center_Y")
        if cx is None or cy is None:
            continue

        # Convert to (lat, lon) format
        lat = cy
        lon = cx

        row = props.get("row_index")
        col = props.get("col_index")
        hexscore = props.get("HexScore", 0)

        HEX_CELLS[cid] = {
            "lat": lat,
            "lon": lon,
            "row": row,
            "col": col,
            "hexscore": hexscore
        }

        all_lats.append(lat)
        all_lons.append(lon)

        if hexscore in (1, 2):
            ALLOWED_IDS.append(cid)

    # Compute bounding box (for the whole hex grid)
    MIN_LAT = min(all_lats)
    MAX_LAT = max(all_lats)
    MIN_LON = min(all_lons)
    MAX_LON = max(all_lons)

    # Set Base station coordinates
    if BS1_ID in HEX_CELLS:
        BS1_COORD = (HEX_CELLS[BS1_ID]["lat"], HEX_CELLS[BS1_ID]["lon"])
    else:
        raise ValueError(f"BS1_ID {BS1_ID} not found in GeoJSON")

    if BS2_ID in HEX_CELLS:
        BS2_COORD = (HEX_CELLS[BS2_ID]["lat"], HEX_CELLS[BS2_ID]["lon"])
    else:
        raise ValueError(f"BS2_ID {BS2_ID} not found in GeoJSON")

    print(f"Loaded {len(HEX_CELLS)} hex cells, {len(ALLOWED_IDS)} allowed for drones.")
    print(f"BS1 = ID {BS1_ID} at {BS1_COORD}")
    print(f"BS2 = ID {BS2_ID} at {BS2_COORD}")

    # Build adjacency graph (row/col grid neighbors)
    build_hex_adjacency()


def build_hex_adjacency():
    global HEX_ADJ
    HEX_ADJ = {cid: [] for cid in HEX_CELLS.keys()}

    cell_list = list(HEX_CELLS.items())

    for i, (id1, c1) in enumerate(cell_list):
        r1, c1_ = c1["row"], c1["col"]
        for j in range(i + 1, len(cell_list)):
            id2, c2 = cell_list[j]
            r2, c2_ = c2["row"], c2["col"]

            # 4-neighbor adjacency: up/down/left/right
            if abs(r1 - r2) + abs(c1_ - c2_) == 1:
                HEX_ADJ[id1].append(id2)
                HEX_ADJ[id2].append(id1)


# =============================
# DRONE BOUNDING BOX (DRONE_BOX_IDS)
# =============================

def _compute_drone_bbox_from_geojson(geojson_path, box_ids=None):
    """
    Read GeoJSON, get Center_X / Center_Y of cells with ids in box_ids
    (default = DRONE_BOX_IDS), then return bounding box:

        (min_lat, max_lat, min_lon, max_lon)

    Note:
        - Center_X = tọa độ X (lon)
        - Center_Y = tọa độ Y (lat)
    """
    if box_ids is None:
        box_ids = DRONE_BOX_IDS

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])

    xs = []
    ys = []

    for feat in features:
        props = feat.get("properties", {})
        cid = props.get("id")
        if cid in box_ids:
            cx = props.get("Center_X")
            cy = props.get("Center_Y")
            if cx is None or cy is None:
                continue
            xs.append(cx)
            ys.append(cy)

    if not xs or not ys:
        raise ValueError(
            f"Center_X/Center_Y not found for ids in {box_ids} "
            f"in file {geojson_path}"
        )

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    # trong code của bạn: lat = Center_Y, lon = Center_X
    min_lon, max_lon = min_x, max_x
    min_lat, max_lat = min_y, max_y

    return min_lat, max_lat, min_lon, max_lon


def init_drone_cells(num_drones):
    """
    Initialize each drone on a distinct hex inside the bounding box
    defined by DRONE_BOX_IDS in the GeoJSON.

    Conditions:
      - Cell has HexScore = 1 or 2.
      - Cell center (lat, lon) lies within the bounding box of DRONE_BOX_IDS.

    Returns: dict drone_id -> current_cell_id
    """
    # Compute bounding box from DRONE_BOX_IDS in GeoJSON
    min_lat, max_lat, min_lon, max_lon = _compute_drone_bbox_from_geojson(
        BASE_GEOJSON_PATH,
        box_ids=DRONE_BOX_IDS
    )

    # Filter cells that meet the conditions (inside bbox + HexScore 1,2)
    candidate_cells = [
        cid for cid, cell in HEX_CELLS.items()
        if cell["hexscore"] in (1, 2)
        and min_lat <= cell["lat"] <= max_lat
        and min_lon <= cell["lon"] <= max_lon
    ]

    if not candidate_cells:
        raise ValueError(
            "No hex with HexScore 1 or 2 found within the bounding box "
            "defined by DRONE_BOX_IDS."
        )

    if num_drones > len(candidate_cells):
        raise ValueError(
            f"Not enough hexes in the bounding box for {num_drones} drones "
            f"(only {len(candidate_cells)} valid cells available)."
        )

    # Randomly select without replacement
    chosen_cells = random.sample(candidate_cells, num_drones)

    drone_cells = {}
    for d in range(num_drones):
        cell_id = chosen_cells[d]
        drone_cells[f"D{d}"] = cell_id

    return drone_cells


def step_drones_no_overlap(drone_cells):
    """
    Update the positions of all drones for one time slot, avoiding overlapping cells, and
    ONLY move within the bounding box defined by DRONE_BOX_IDS.
    Returns: dict drone_id -> new_cell_id
    """

    # Compute bounding box from DRONE_BOX_IDS
    min_lat, max_lat, min_lon, max_lon = _compute_drone_bbox_from_geojson(
        BASE_GEOJSON_PATH,
        box_ids=DRONE_BOX_IDS
    )

    new_drone_cells = {}
    occupied = set()  # cells already occupied in the new time slot

    drone_ids = list(drone_cells.keys())
    random.shuffle(drone_ids)  # random order for fairness

    for drone_id in drone_ids:
        current_cell_id = drone_cells[drone_id]

        neighbors = HEX_ADJ.get(current_cell_id, [])

        # Only select neighbors:
        #   - HexScore 1 or 2
        valid_neighbors = []
        for nid in neighbors:
            cell = HEX_CELLS[nid]
            lat = cell["lat"]
            lon = cell["lon"]
            if (
                cell["hexscore"] in (1, 2)
                and min_lat <= lat <= max_lat
                and min_lon <= lon <= max_lon
            ):
                valid_neighbors.append(nid)

        # Allow drone to stay in the current cell (assuming it's within the bbox)
        candidates = valid_neighbors + [current_cell_id]
        random.shuffle(candidates)

        chosen = None
        for cid in candidates:
            if cid not in occupied:
                chosen = cid
                break

        # If all are occupied, stay in the current cell (may overlap, but rare)
        if chosen is None:
            chosen = current_cell_id

        new_drone_cells[drone_id] = chosen
        occupied.add(chosen)

    return new_drone_cells


# =============================
# DISTANCE & GRAPH
# =============================

def get_xp():
    """
    Return the array module to use (NumPy or CuPy)
    depending on USE_GPU and CuPy availability.
    """
    if USE_GPU and cp is not None:
        return cp
    return np


def euclidean_distance_deg(lat1, lon1, lat2, lon2):
    """
    Simple Euclidean distance (OK for small areas).
    """
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def build_graph(node_positions, comm_range_deg=COMM_RANGE_DEG):
    """
    Build an undirected graph based on COMM_RANGE_DEG using a
    vectorized (and optionally GPU-accelerated) distance matrix.

    node_positions: dict[node_id] = (lat, lon)
    Returns: adjacency dict: graph[node] = set(neighbors)
    """
    xp = get_xp()

    node_ids = list(node_positions.keys())
    lats = xp.array([node_positions[n][0] for n in node_ids])
    lons = xp.array([node_positions[n][1] for n in node_ids])

    # pairwise distance matrix
    dlat = lats[:, None] - lats[None, :]
    dlon = lons[:, None] - lons[None, :]
    dist_matrix = xp.sqrt(dlat ** 2 + dlon ** 2)

    # adjacency dict (CPU side)
    graph = {n: set() for n in node_ids}

    # mask for neighbors (ignore self-distances)
    mask = (dist_matrix <= comm_range_deg) & (dist_matrix > 0)

    # get edge indices
    idx_i, idx_j = xp.where(mask)

    # If using GPU, bring indices back to CPU
    if xp is cp:
        idx_i = idx_i.get()
        idx_j = idx_j.get()

    for i, j in zip(idx_i, idx_j):
        n1 = node_ids[int(i)]
        n2 = node_ids[int(j)]
        graph[n1].add(n2)
        graph[n2].add(n1)

    return graph


# =============================
# HEXSCORE=3 (FORBIDDEN AREAS)
# =============================

def load_hexscore3_polygons(geojson_path):
    """
    Read GeoJSON and extract all polygons of cells with HexScore = 3.
    Returns a list of polygons, each polygon is a list of (x, y) tuples with:
        x = coordinate along the X axis (Center_X, left/right)
        y = coordinate along the Y axis (Center_Y, top/bottom)
    Note that geojson geometry.coordinates are in the form [x, y].
    """
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polys = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        if props.get("HexScore") != 3:
            continue

        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue

        # coordinates: [ [ [x, y], [x, y], ... ] ]
        rings = geom.get("coordinates", [])
        if not rings:
            continue

        outer_ring = rings[0]
        poly = [(pt[0], pt[1]) for pt in outer_ring]  # (x, y)
        polys.append(poly)

    return polys


def _orientation(ax, ay, bx, by, cx, cy):
    """Return 0, 1, 2 corresponding to collinear / left / right."""
    val = (by - ay) * (cx - bx) - (bx - ax) * (cy - by)
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def _on_segment(ax, ay, bx, by, cx, cy):
    """Check if point B lies on segment AC (assuming collinear)."""
    return (min(ax, cx) - 1e-9 <= bx <= max(ax, cx) + 1e-9 and
            min(ay, cy) - 1e-9 <= by <= max(ay, cy) + 1e-9)


def segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    """
    o1 = _orientation(x1, y1, x2, y2, x3, y3)
    o2 = _orientation(x1, y1, x2, y2, x4, y4)
    o3 = _orientation(x3, y3, x4, y4, x1, y1)
    o4 = _orientation(x3, y3, x4, y4, x2, y2)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special collinear cases
    if o1 == 0 and _on_segment(x1, y1, x3, y3, x2, y2):
        return True
    if o2 == 0 and _on_segment(x1, y1, x4, y4, x2, y2):
        return True
    if o3 == 0 and _on_segment(x3, y3, x1, y1, x4, y4):
        return True
    if o4 == 0 and _on_segment(x3, y3, x2, y2, x4, y4):
        return True

    return False


def point_in_polygon(x, y, polygon):
    """
    Check if point (x,y) is inside the polygon (ray casting).
    polygon: list[(x, y)]
    """
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i

    return inside


def segment_crosses_hexscore3(p1_lat, p1_lon, p2_lat, p2_lon, hex3_polygons):
    """
    Check if the line segment connecting two nodes (p1_lat, p1_lon) -> (p2_lat, p2_lon)
    intersects any polygon in hex3_polygons.
    hex3_polygons: list of polygons, each polygon is a list of (x, y) tuples in GeoJSON (X,Y) format.
    """
    # Node positions are stored as (lat, lon) => convert to (x, y) = (lon, lat)
    x1, y1 = p1_lon, p1_lat
    x2, y2 = p2_lon, p2_lat

    for poly in hex3_polygons:
        # (opt) bounding box để loại nhanh các trường hợp xa
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)


        # 1) Check intersection between the segment and each edge of the polygon
        for i in range(len(poly) - 1):
            x3, y3 = poly[i]
            x4, y4 = poly[i + 1]
            if segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
                return True

        # 2) Case where the segment lies entirely inside the polygon without crossing edges:
        #    check if the midpoint of the segment is inside the polygon.
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        if point_in_polygon(mid_x, mid_y, poly):
            return True

    return False


# =============================
# SIMULATION (POSITIONS OVER TIME)
# =============================

def run_simulation():
    """
    Run the full time-slot simulation:
      - Drones are placed only on hexes with HexScore 1 or 2
        within bounding box DRONE_BOX_IDS.
      - Their positions are the Center_X / Center_Y (swapped to lat/lon).
      - Each time slot, each drone moves between neighbor hex cells
        (row/col adjacency), restricted to HexScore 1 or 2 and inside box.
    """
    # Make sure hex grid is loaded
    load_hex_grid(BASE_GEOJSON_PATH)

    positions = {}

    # 1) Initialize time slot 0
    positions[0] = {}
    positions[0]["BS1"] = BS1_COORD
    positions[0]["BS2"] = BS2_COORD

    # Initialize drone cell locations (only HexScore 1 or 2 inside DRONE_BOX)
    drone_cells = init_drone_cells(NUM_DRONES)

    # Set drone positions from cell centers
    for d in range(NUM_DRONES):
        drone_id = f"D{d}"
        cell_id = drone_cells[drone_id]
        cell = HEX_CELLS[cell_id]
        lat, lon = cell["lat"], cell["lon"]
        positions[0][drone_id] = (lat, lon)

    # 2) Simulate remaining time slots
    for t in range(1, NUM_TIMESLOTS):
        positions[t] = {}
        positions[t]["BS1"] = BS1_COORD
        positions[t]["BS2"] = BS2_COORD

        # Update all drones at once, ensuring no overlapping cells
        drone_cells = step_drones_no_overlap(drone_cells)

        # Record the corresponding coordinates of each drone
        for d in range(NUM_DRONES):
            drone_id = f"D{d}"
            cell_id = drone_cells[drone_id]
            cell = HEX_CELLS[cell_id]
            lat, lon = cell["lat"], cell["lon"]
            positions[t][drone_id] = (lat, lon)

    return positions


# =============================
# CSV EXPORT: POSITIONS
# =============================

def export_positions_csv(positions, csv_path):
    """
    Save positions of all nodes for all time slots:
    time_slot,node_id,node_type,lat,lon
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_slot", "node_id", "node_type", "lat", "lon"])

        for t in range(len(positions)):
            for node_id, (lat, lon) in positions[t].items():
                if node_id in ("BS1", "BS2"):
                    node_type = "BS"
                else:
                    node_type = "drone"
                writer.writerow([t, node_id, node_type, lat, lon])


# =============================
# LINK MODELS & WEIGHTED GRAPH
# =============================


def link_delay_from_distance(d):
    """
    Calc link delay (ms) from distance d (same units as COMM_RANGE_DEG).
    Assumes:
      delay = base_delay + k * d
    """
    base_delay_ms = 5.0     # 5 ms fixed (MAC, processing...)
    k_ms_per_unit = 0.01    # 0.01 ms per unit distance
    return base_delay_ms + k_ms_per_unit * d


def link_capacity_from_distance_physical(
    d,
    base_bw=15.0,               # Mbps, "maximum" bandwidth at reference distance
    comm_range=COMM_RANGE_DEG,  # maximum communication range (same units as d)
    path_loss_exp=2.0,          # n: path loss exponent (2~4 depending on environment)
    ref_fraction=0.1,           # d0 = ref_fraction * comm_range
    ref_snr_db=30.0,            # SNR at d0 (dB), e.g., 30 dB
    bandwidth_hz=10e6           # Physical bandwidth (Hz), e.g., 10 MHz
):
    """
    Physical fading model (log-distance + Shannon):

      - SNR(d) = SNR(d0) - 10 * n * log10(d / d0)
      - C(d) = B * log2(1 + SNR_linear)
      - standard C(d0) = base_bw (Mbps)

    Tham số:
      d             : distance between 2 nodes (same units as comm_range)
      base_bw       : target capacity at d0, in Mbps (default 10)
      comm_range    : maximum communication range
      path_loss_exp : path loss exponent n (2~4)
      ref_fraction  : fraction of d0 relative to comm_range (0.1 => d0 = 10% comm_range)
      ref_snr_db    : SNR at d0, in dB
      bandwidth_hz  : physical bandwidth Hz for Shannon calculation

    Returns:
      capacity_mbps >= 0
    """

    if comm_range <= 0:
        return base_bw

    # Ensure d > 0 to avoid log10(0)
    # If d is too small, clamp to d0 (considered "very close")
    d0 = max(ref_fraction * comm_range, 1e-6)
    d_eff = max(d, d0)

    # SNR at d0 (linear)
    snr0_linear = 10 ** (ref_snr_db / 10.0)

    # SNR(d) theo log-distance model:
    # SNR_linear(d) = SNR0 * (d0 / d)^n
    ratio = d0 / d_eff
    snr_linear = snr0_linear * (ratio ** path_loss_exp)

    # Shannon capacity (bps)
    capacity_bps = bandwidth_hz * math.log2(1.0 + snr_linear)
    capacity_mbps_raw = capacity_bps / 1e6  # convert to Mbps

    # Normalize so that capacity at d0 equals base_bw
    # Calculate capacity at d0 using the same formula
    snr0 = snr0_linear
    cap0_bps = bandwidth_hz * math.log2(1.0 + snr0)
    cap0_mbps = cap0_bps / 1e6

    if cap0_mbps <= 0:
        # fallback: if configuration is invalid, return base_bw
        return base_bw

    # scale so that at d0: capacity_scaled(d0) = base_bw
    capacity_mbps = base_bw * (capacity_mbps_raw / cap0_mbps)

    # Ensure non-negative, and optionally cap at base_bw
    capacity_mbps = max(0.0, min(capacity_mbps, base_bw))

    return capacity_mbps

def link_jitter_from_distance(d, comm_range=COMM_RANGE_DEG,
                              base_jitter_ms=5.0,
                              max_extra_jitter_ms=20.0):
    """
    Jitter increases with distance.
      - d: distance between 2 nodes
      - base_jitter_ms: minimum jitter (close nodes)
      - max_extra_jitter_ms: maximum additional jitter at the edge of comm_range

    Returns jitter_ms >= 0
    """
    if comm_range <= 0:
        return base_jitter_ms

    # Distance fraction (0..1)
    fraction = min(max(d / comm_range, 0.0), 1.0)

    # Mean jitter increases linearly with fraction
    mean_jitter = base_jitter_ms + fraction * max_extra_jitter_ms

    # Add random noise ±20%
    low = 0.8 * mean_jitter
    high = 1.2 * mean_jitter

    jitter = random.uniform(low, high)
    return max(jitter, 0.0)


def link_loss_from_distance(d,
                            comm_range=COMM_RANGE_DEG,
                            base_loss=0.001,          # 0.1%
                            max_extra_loss=0.001):     # +5% at the edge of comm_range
    """
    Packet loss (%) increases with distance.
    - base_loss: loss minimum when d = 0
    - max_extra_loss: maximum increase when d = comm_range

    Returns loss (ratio 0.0 .. 1.0)
    """
    if comm_range <= 0:
        return base_loss

    fraction = min(max(d / comm_range, 0.0), 1.0)

    # Tăng loss tuyến tính
    mean_loss = base_loss + fraction * max_extra_loss

    # Thêm nhiễu ngẫu nhiên ±20%
    low = 0.8 * mean_loss
    high = 1.2 * mean_loss

    loss = random.uniform(low, high)

    # Giới hạn 0..1
    return min(max(loss, 0.0), 1.0)


def build_weighted_graph(graph, positions_t, hex3_polygons,
                         required_bw=REQUIRED_BW_1080P_MBPS):
    """
    Từ:
      - graph: dict node -> set(neighbors) ( build_graph)
      - positions_t: dict node -> (lat, lon) at time slot t
      - hex3_polygons: list polygon (HexScore=3)
    Tạo ra:
      - weighted_graph: dict node -> list of
            (neighbor, distance, delay_ms, bw, jitter_ms, loss)

    
    """
    weighted_graph = {n: [] for n in graph.keys()}

    for u in graph:
        lat_u, lon_u = positions_t[u]
        for v in graph[u]:
            lat_v, lon_v = positions_t[v]

            # Skip edges crossing HexScore=3 regions
            if hex3_polygons and segment_crosses_hexscore3(
                lat_u, lon_u, lat_v, lon_v, hex3_polygons
            ):
                continue

            d = euclidean_distance_deg(lat_u, lon_u, lat_v, lon_v)

            bw = link_capacity_from_distance_physical(d)
            jitter_ms = link_jitter_from_distance(d)
            delay_ms = link_delay_from_distance(d)
            loss = link_loss_from_distance(d)

            # Filter by minimum bandwidth for streaming
            if bw < required_bw:
                continue

            weighted_graph[u].append(
                (v, d, delay_ms, bw, jitter_ms, loss)
            )

    return weighted_graph

# =============================
# CSV EXPORT: DQN PATHS
# =============================

def export_paths_dqn_csv(
    positions,
    csv_path,
    visualize: bool = False,
    vis_dir: str = "graphs_dqn",
    device: str = "cuda",
):
    """
    Use DQN (PyTorch + GraphDuelingDQN) to find the best path BS1->BS2 
    for EACH time_slot, then write to CSV.

    Nếu visualize=True:
      - Sau khi build env (trước khi train), vẽ topology của time_slot đó.

    CSV format:
      time_slot, path_index, hop_count, path_length, total_delay_ms, node_sequence
    """
    device = "cuda"

    # HexScore=3 regions shared across all time_slots
    hex3_polygons = load_hexscore3_polygons(BASE_GEOJSON_PATH)

    # Nếu muốn lưu hình graph
    if visualize and not os.path.exists(vis_dir):
        os.makedirs(vis_dir, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_slot",
            "path_index",
            "hop_count",
            "path_length",
            "total_delay_ms",
            "node_sequence"
        ])

        for t in range(len(positions)):
            print(f"[DQN] Processing time_slot {t} ...")

            positions_t = positions[t]

            # 1) Build unweighted graph theo comm range
            base_graph = build_graph(positions_t, COMM_RANGE_DEG)

            # 2) Build weighted graph (delay/bw/jitter/loss)
            weighted_graph = build_weighted_graph(
                base_graph,
                positions_t,
                hex3_polygons,
                required_bw=REQUIRED_BW_1080P_MBPS,
            )

            if not weighted_graph:
                print(f"[DQN]  time_slot {t}: weighted_graph empty (no edges satisfy requirements).")
                continue

            # 3) Tạo env RL
            env = UAVRoutingEnvTorch(
                weighted_graph=weighted_graph,
                max_hops=MAX_HOPS,
                start_id="BS1",
                goal_id="BS2",
                node_positions=positions_t,
                target_bw_mbps=REQUIRED_BW_1080P_MBPS,
                max_delay_ms=MAX_DELAY_1080P_MS,
                max_jitter_ms=MAX_JITTER_1080P_MS,
                max_loss=MAX_LOSS_1080P,   
                w_bw=1.0,
                w_delay=1.0,
                w_jitter=0.5,
                w_loss=2.0,                
                w_hop=0.2,
                w_progress=10.0,
                goal_bonus=50.0,
                fail_penalty=-30.0,
            )

            # 4) Build graph tensors cho GNN từ env
            node_feats, edge_index, edge_attr = build_graph_from_env(env, device=device)

            # 5) Train DQN với GraphDuelingDQN (GNN)
            agent = train_dqn_routing(
                env,
                num_episodes=5000,
                gamma=0.95,
                lr=1e-3,
                epsilon_start=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.998,
                batch_size=64,
                buffer_capacity=100000,
                target_update_freq=500,
                device=device,
                log_interval=500,

                use_gnn=True,
                graph_tensors=(node_feats, edge_index, edge_attr),
            )

            # 6) Extract best path
            path, total_dist, total_delay = extract_greedy_path_dqn(
                env, agent, device=device
            )

            if path is None:
                print(f"[DQN]  time_slot {t}: No path BS1->BS2.")
                continue

            hop_count = len(path) - 1
            path_length = total_dist
            node_seq_str = "->".join(path)
            path_index = 0  

            writer.writerow([
                t,
                path_index,
                hop_count,
                path_length,
                total_delay,
                node_seq_str
            ])

            print(f"[DQN]  time_slot {t}: path =", node_seq_str)
            print(f"       hop_count={hop_count}, length={path_length:.2f}, delay={total_delay:.2f} ms")



# =============================
# GEOJSON EXPORT FOR QGIS
# =============================

def load_or_init_geojson(path):
    """
    Load a GeoJSON file if it exists; otherwise return an empty FeatureCollection.
    """
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # If it's not a FeatureCollection, wrap it
        if data.get("type") != "FeatureCollection":
            data = {"type": "FeatureCollection", "features": [data]}
    else:
        data = {"type": "FeatureCollection", "features": []}
    return data


def export_geojson_with_uav(
    base_geojson_path,
    output_geojson_path,
    positions,
    paths_csv_path,
):

    # 1) Load base GeoJSON
    base = load_or_init_geojson(base_geojson_path)
    features = base.get("features", [])

    # 2) Add BS features (static, không temporal)
    for bs_name, coord, bs_idx in [("BS1", BS1_COORD, 1), ("BS2", BS2_COORD, 2)]:
        lat, lon = coord
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "feature_type": bs_name,
                "bs_name": bs_name,
                "bs_id": bs_idx
            }
        }
        features.append(feat)

    # 3) Add drone point features for each time_slot
    for t in range(len(positions)):
        current_time = BASE_TIME + t * TIME_DELTA
        iso_time = current_time.isoformat()

        for node_id, (lat, lon) in positions[t].items():
            if node_id in ("BS1", "BS2"):
                continue

            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "feature_type": node_id,
                    "node_id": node_id,
                    "time_slot": t,
                    "time": iso_time  # used as Temporal field in QGIS
                }
            }
            features.append(feat)

    # 4) Read paths from CSV (A* or DQN) and draw LineString
    with open(paths_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Check if there is a total_delay_ms column (DQN) – if not, ignore
        has_delay = reader.fieldnames is not None and "total_delay_ms" in reader.fieldnames

        for row in reader:
            try:
                t = int(row["time_slot"])
                path_index = int(row["path_index"])
                hop_count = int(row["hop_count"])
                path_length = float(row.get("path_length", 0.0))
            except (KeyError, ValueError):
                continue

            node_seq_str = row.get("node_sequence", "")
            if not node_seq_str:
                continue

            node_list = node_seq_str.split("->")

            current_time = BASE_TIME + t * TIME_DELTA
            iso_time = current_time.isoformat()

            # Build list of coordinates [lon, lat] according to positions[t]
            if t not in positions:
                # no positions for this time_slot
                continue

            positions_t = positions[t]
            coords = []
            valid_path = True
            for node_name in node_list:
                if node_name not in positions_t:
                    valid_path = False
                    break
                lat, lon = positions_t[node_name]
                coords.append([lon, lat])

            if not valid_path or len(coords) < 2:
                # path không hợp lệ
                continue

            props = {
                "feature_type": "uav_path",
                "time_slot": t,
                "time": iso_time,
                "path_index": path_index,
                "hop_count": hop_count,
                "path_length": path_length,
                "node_sequence": node_seq_str,
            }

            if has_delay:
                try:
                    props["total_delay_ms"] = float(row["total_delay_ms"])
                except Exception:
                    pass

            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
                "properties": props
            }
            features.append(feat)

    base["features"] = features

    with open(output_geojson_path, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

    print(f"GeoJSON written to {output_geojson_path}")
    print(f"  - Paths loaded from: {paths_csv_path}")

# =============================
# MAIN
# =============================

def main():
    print("Running simulation...")
    positions = run_simulation()

    print("Exporting positions CSV...")
    export_positions_csv(positions, POSITIONS_CSV)
    print(f"Positions CSV: {POSITIONS_CSV}")

    load_hex_grid(BASE_GEOJSON_PATH)

    print("Exporting paths CSV (DQN)...")
    export_paths_dqn_csv(positions, "paths_dqn.csv", visualize=True, vis_dir="graphs_dqn")
    print("Paths CSV (DQN): paths_dqn.csv")



    # If there is paths_dqn.csv:
    print("Exporting GeoJSON with UAV data (DQN)...")
    export_geojson_with_uav(
        BASE_GEOJSON_PATH,
        "uav_dqn.geojson",
        positions,
        "paths_dqn.csv"
    )


if __name__ == "__main__":
    main()

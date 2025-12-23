import heapq
import random
import csv
import math
import json
from pathlib import Path

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

from uav_rl_gps_per_noisylinera import (
    UAVRoutingEnvTorch,
    train_dqn_routing,
    extract_greedy_path_dqn,
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
        #   - HexScore 1 hoặc 2
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
      - Chuẩn hoá sao cho C(d0) = base_bw (Mbps)

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
# A* / DIJKSTRA ROUTING
# =============================

def find_best_path_astar(
    graph,
    positions_t,
    hex3_polygons,
    start="BS1",
    goal="BS2",
    required_bw=REQUIRED_BW_1080P_MBPS,
    lambda_dist=0.001,
    lambda_loss=1000.0,          # trọng số phạt loss
    max_e2e_loss=None, # ngưỡng loss tối đa, có thể None nếu không muốn hard constraint
    use_heuristic=True
):
    """
    Find the best path from start to goal with constraints:
      - Each hop has bandwidth >= required_bw (Mbps).
      - Does not cross HexScore=3 regions.
      - End-to-end loss <= max_e2e_loss (if max_e2e_loss is not None).
      - Optimize cost:
          cost = total_delay_ms + lambda_dist * total_distance
                                    + lambda_loss * e2e_loss

    Trả về:
      path (list node_id),
      total_distance,
      total_delay_ms
    hoặc (None, None, None) nếu không tìm được path.
    """

    # 1) Graph có trọng số + QoS filter
    w_graph = build_weighted_graph(graph, positions_t, hex3_polygons, required_bw)

    if start not in w_graph or goal not in w_graph:
        return None, None, None

    lat_goal, lon_goal = positions_t[goal]

    def h(node):
        """Heuristic for A*: estimate remaining delay based on distance."""
        if not use_heuristic:
            return 0.0
        lat, lon = positions_t[node]
        dist = euclidean_distance_deg(lat, lon, lat_goal, lon_goal)
        # giống link_delay_from_distance: ~ 0.01 ms / unit
        return 0.01 * dist

    # priority queue: (f_score, g_delay, g_dist, g_hops, success_prob, node, path)
    pq = []
    start_h = h(start)
    # ban đầu: delay=0, dist=0, hops=0, success_prob=1.0 (loss=0)
    heapq.heappush(pq, (start_h, 0.0, 0.0, 0, 1.0, start, [start]))

    # best_cost[node] = best cost seen so far to reach node
    best_cost = {start: 0.0}

    while pq:
        f_score, g_delay, g_dist, g_hops, g_success_prob, u, path = heapq.heappop(pq)

        if u == goal:
            # The first time the goal is popped is the best path
            return path, g_dist, g_delay

        if u not in w_graph:
            continue

        for (v, d_uv, delay_uv, bw_uv, jitter_uv, loss_uv) in w_graph[u]:
            # Avoid cycles
            if v in path:
                continue

            new_hops = g_hops + 1
            if new_hops > MAX_HOPS:
                continue

            new_delay = g_delay + delay_uv
            new_dist = g_dist + d_uv

            # Update success probability multiplicatively:
            # P_succ_new = P_succ_old * (1 - loss_uv)
            new_success_prob = g_success_prob * (1.0 - loss_uv)
            e2e_loss = 1.0 - new_success_prob

            # Hard constraint: if loss exceeds threshold, skip
            if (max_e2e_loss is not None) and (e2e_loss > max_e2e_loss):
                continue

            # Combined cost:
            new_cost = (
                new_delay
                + lambda_dist * new_dist
                + lambda_loss * e2e_loss
            )

            # If a better cost to v has been seen before, skip
            if v in best_cost and new_cost >= best_cost[v]:
                continue

            best_cost[v] = new_cost
            f_v = new_cost + h(v)

            heapq.heappush(
                pq,
                (f_v, new_delay, new_dist, new_hops, new_success_prob, v, path + [v])
            )

    # No path satisfies QoS
    return None, None, None

# =============================
# CSV EXPORT: DQN PATHS
# =============================

def export_paths_dqn_csv(positions, csv_path):
    """
    Use DQN (PyTorch) to find the best path BS1->BS2 for EACH time_slot,
    then write to CSV.

    
    CSV format:
      time_slot, path_index, hop_count, path_length, total_delay_ms, node_sequence
    """
    # HexScore=3 regions shared across all time_slots
    hex3_polygons = load_hexscore3_polygons(BASE_GEOJSON_PATH)

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

            base_graph = build_graph(positions_t, COMM_RANGE_DEG)

            weighted_graph = build_weighted_graph(
                base_graph,
                positions_t,
                hex3_polygons,
                required_bw=REQUIRED_BW_1080P_MBPS,
            )

            
            if not weighted_graph:
                print(f"[DQN]  time_slot {t}: weighted_graph empty (no edges satisfy requirements).")
                continue

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


            agent = train_dqn_routing(
                env,
                num_episodes=5000,
                gamma=0.95,
                lr=1e-3,
                epsilon_start=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.998,
                device="cuda", 
            )

            path, total_dist, total_delay = extract_greedy_path_dqn(env, agent, device="cuda")

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


def export_paths_csv(positions, csv_path):
    """
    For each time slot, build graph, find best path BS1->BS2 (A*),
    and only write paths that meet QoS (do not cross HexScore=3, bw >= required_bw).
    Add metrics:
      - path_length: total length (sum distance)
      - total_delay_ms
    """
    # Prepare polygons for HexScore=3 from the original file
    hex3_polygons = load_hexscore3_polygons(BASE_GEOJSON_PATH)

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
            node_positions = positions[t]
            graph = build_graph(node_positions, COMM_RANGE_DEG)

            weighted_graph = build_weighted_graph(
                graph, node_positions, hex3_polygons,
                required_bw=REQUIRED_BW_1080P_MBPS
            )
            print(f"[t={t}] nodes={len(graph)}, edges_raw={sum(len(v) for v in graph.values())}")
            print(f"[t={t}] edges_after_qos={sum(len(v) for v in weighted_graph.values())}")
            path, total_dist, total_delay = find_best_path_astar(
                graph,
                node_positions,
                hex3_polygons,
                start="BS1",
                goal="BS2",
                required_bw=REQUIRED_BW_1080P_MBPS,
                lambda_dist=0.001,
                max_e2e_loss=None, 
                use_heuristic=True
            )

            if path is None:
                # No path meets QoS at this time_slot
                print(f"[t={t}] No QoS path found (bw >= {REQUIRED_BW_1080P_MBPS} Mbps)")
                continue

            hop_count = len(path) - 1
            path_length = total_dist
            node_seq_str = "->".join(path)
            path_index = 0  # each time_slot has only 1 best path

            writer.writerow([
                t,
                path_index,
                hop_count,
                path_length,
                total_delay,
                node_seq_str
            ])

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

    print("Exporting paths CSV (A*)...")
    export_paths_csv(positions, PATHS_CSV)
    print(f"Paths CSV (A*): {PATHS_CSV}")

    print("Exporting GeoJSON with UAV data (A*)...")
    export_geojson_with_uav(
        BASE_GEOJSON_PATH,
        "uav_astar.geojson",
        positions,
        "paths.csv"
    )


    print("Exporting paths CSV (DQN)...")
    export_paths_dqn_csv(positions, "paths_dqn.csv")
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

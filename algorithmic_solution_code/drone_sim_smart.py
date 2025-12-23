#!/usr/bin/env python3
"""
Dynamic Mesh Network Drone Simulation (Smart Routing V10: DUAL MODE & BASE INCLUSION).

Fixes & Features:
1.  **Base Station Inclusion**: Bases A and B are now fixed endpoints of the chain visualization.
2.  **Dual Rendering**: Renders two GIFs: one for Sum-Min and one for Min-Max assignment.
3.  **Constant Power Draw**: Movement model ensures equal power drain rate.
"""

import json
import random
import sys
import io
import math
from typing import List, Tuple
from datetime import datetime, timedelta

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D

import networkx as nx
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ---------- Configuration ----------
COST_CLEAR = 1      
COST_FOLIAGE = 8    
INF_COST = 1e9      
INITIAL_BATTERY = 50.0 

def load_geojson(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("features", [])
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def get_map_data(features):
    """Extracts centers and scores."""
    centers = []
    hex_scores = []
    
    for feat in features:
        props = feat.get("properties", {})
        if "Center_X" in props and "Center_Y" in props:
            cx = float(props["Center_X"])
            cy = float(props["Center_Y"])
        else:
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [])[0]
            poly_arr = np.array(coords)
            cx, cy = poly_arr.mean(axis=0)
            
        centers.append((cx, cy))
        hex_scores.append(int(props.get("HexScore", 1)))

    return np.array(centers), hex_scores

def build_network_graph(centers, hex_scores):
    """Builds the navigation graph."""
    G = nx.Graph()
    valid_indices = []
    
    for i, (x, y) in enumerate(centers):
        if hex_scores[i] == 3: continue
        G.add_node(i, pos=(x, y), score=hex_scores[i])
        valid_indices.append(i)

    if not valid_indices:
        print("Error: No valid (non-building) hexes found.")
        sys.exit(1)

    valid_positions = centers[valid_indices]
    dists = cdist(valid_positions, valid_positions)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    threshold = np.median(min_dists) * 1.5
    
    rows, cols = np.where(dists < threshold)
    for r, c in zip(rows, cols):
        if r >= c: continue
        
        u, v = valid_indices[r], valid_indices[c]
        dist = dists[r, c]
        
        terrain_cost = COST_FOLIAGE if hex_scores[v] == 2 else COST_CLEAR
        weight = (dist / 100.0) * terrain_cost
        
        G.add_edge(u, v, weight=weight)
        
    return G, valid_indices

def find_chain_path(G, centers, valid_indices):
    """Finds the optimal communication path between Base A and Base B (nodes)."""
    valid_centers = centers[valid_indices]
    scores = valid_centers[:, 0] - valid_centers[:, 1] 
    
    start_node = valid_indices[np.argmin(scores)]
    end_node = valid_indices[np.argmax(scores)]

    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
        return start_node, end_node, path
    except nx.NetworkXNoPath:
        return start_node, end_node, []

def interpolate_targets(path_nodes, centers, num_drones):
    """Generates 'num_drones' evenly spaced coordinates along the hex path, EXCLUDING the bases."""
    path_coords = centers[path_nodes]
    
    if len(path_coords) < 2:
        return np.array([]), centers[path_nodes[0]], centers[path_nodes[0]]
        
    dists = np.linalg.norm(path_coords[1:] - path_coords[:-1], axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_dist = cum_dist[-1]
    
    # We need N targets, placed BETWEEN Base A (0.0) and Base B (total_dist).
    # The first target is at total_dist / (N+1), the last is at total_dist * N / (N+1).
    target_dists = np.linspace(
        total_dist / (num_drones + 1), 
        total_dist * num_drones / (num_drones + 1), 
        num_drones
    )
    
    tx = np.interp(target_dists, cum_dist, path_coords[:, 0])
    ty = np.interp(target_dists, cum_dist, path_coords[:, 1])
    
    return np.column_stack((tx, ty)), centers[path_nodes[0]], centers[path_nodes[-1]]

def get_nearest_node(G, point, centers):
    """Finds the graph node closest to a geometric point."""
    nodes = list(G.nodes())
    node_pos = centers[nodes]
    dists = np.linalg.norm(node_pos - point, axis=1)
    return nodes[np.argmin(dists)]

def min_max_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Solves the Min-Max Assignment Problem."""
    num_active = cost_matrix.shape[0]
    if np.all(cost_matrix >= INF_COST): return None, None 

    sorted_costs = cost_matrix[cost_matrix < INF_COST]
    if sorted_costs.size == 0: return None, None 
    
    low, high = np.min(sorted_costs), np.max(sorted_costs)
    final_row_ind, final_col_ind = None, None
    
    for _ in range(100): 
        M = (low + high) / 2
        constrained_cost_matrix = np.copy(cost_matrix)
        constrained_cost_matrix[constrained_cost_matrix > M] = INF_COST 

        try:
            row_ind, col_ind = linear_sum_assignment(constrained_cost_matrix)
            if len(row_ind) == num_active and np.all(constrained_cost_matrix[row_ind, col_ind] < INF_COST):
                final_row_ind, final_col_ind = row_ind, col_ind
                high = M 
            else:
                low = M 
        except ValueError:
            low = M 
    
    if final_row_ind is None: return linear_sum_assignment(cost_matrix)
        
    optimal_max_cost = cost_matrix[final_row_ind, final_col_ind].max() 
    constrained_cost_matrix = np.copy(cost_matrix)
    constrained_cost_matrix[constrained_cost_matrix > optimal_max_cost + 1e-6] = INF_COST 
    row_ind, col_ind = linear_sum_assignment(constrained_cost_matrix)
    
    return row_ind, col_ind

def sum_min_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Solves the standard Linear Sum Assignment Problem (Hungarian)."""
    if np.all(cost_matrix >= INF_COST): return None, None 
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def sample_path_at_t(path, t):
    """Samples a polyline path at progress t (0.0 to 1.0)."""
    if t <= 0: return path[0]
    if t >= 1: return path[-1]
    
    diffs = path[1:] - path[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    total_len = np.sum(dists)
    
    if total_len == 0: return path[0]
    
    target_dist = t * total_len
    
    current_dist = 0
    for i, d in enumerate(dists):
        if current_dist + d >= target_dist:
            local_t = (target_dist - current_dist) / d
            return path[i] + diffs[i] * local_t
        current_dist += d
        
    return path[-1]


def calculate_assignment(G, start_positions, target_positions, centers, base_a, base_b, assignment_mode: str):
    """
    Core function to calculate paths and costs based on the specified assignment mode.
    """
    num_drones_req = len(target_positions)
    num_total_drones = len(start_positions)
    
    # --- 1. Isolation Check ---
    component_of_interest = set()
    try:
        for component in nx.connected_components(G):
            if base_a in component and base_b in component:
                component_of_interest = component
                break
    except Exception:
        pass 

    all_start_nodes = [get_nearest_node(G, p, centers) for p in start_positions]
    active_drones_indices = [i for i, node in enumerate(all_start_nodes) if node in component_of_interest]
    active_start_nodes = [all_start_nodes[i] for i in active_drones_indices]
    
    num_active = len(active_drones_indices)
    static_count = num_total_drones - num_active 
    
    if num_active == 0:
        zero_costs = [0.0] * num_total_drones
        return [np.array([start_positions[i], start_positions[i]]) for i in range(num_total_drones)], np.array(target_positions), np.zeros(num_drones_req, dtype=int), static_count, zero_costs, 0.0, 0.0

    # --- 2. Build Cost Matrix ---
    cost_matrix = np.full((num_active, num_drones_req), INF_COST, dtype=float)
    target_nodes = [get_nearest_node(G, p, centers) for p in target_positions]
    
    for r in range(num_active):
        for c in range(num_drones_req):
            try:
                d = nx.shortest_path_length(G, source=active_start_nodes[r], target=target_nodes[c], weight="weight")
                cost_matrix[r, c] = d
            except nx.NetworkXNoPath:
                cost_matrix[r, c] = INF_COST

    # --- 3. Assignment ---
    if assignment_mode == 'MIN_MAX':
        row_ind, col_ind = min_max_assignment(cost_matrix)
    else: # SUM_MIN
        row_ind, col_ind = sum_min_assignment(cost_matrix)
    
    # --- 4. Handle Failed Assignment ---
    if row_ind is None or col_ind is None or len(row_ind) != num_active:
        print(f"CRITICAL: {assignment_mode} assignment failed. All drones static.")
        zero_costs = [0.0] * num_total_drones
        return [np.array([start_positions[i], start_positions[i]]) for i in range(num_total_drones)], np.array(target_positions), np.zeros(num_drones_req, dtype=int), num_total_drones, zero_costs, 0.0, 0.0

    # --- 5. Generate detailed paths and metrics ---
    active_drone_paths = []
    
    # Store path costs
    all_drone_path_costs = [0.0] * num_total_drones
    chain_order = np.zeros(num_drones_req, dtype=int)
    
    # Calculate costs and assign chain order
    costs = cost_matrix[row_ind, col_ind]
    max_cost = costs.max()
    total_cost = costs.sum()

    for r, c in zip(row_ind, col_ind):
        original_drone_idx = active_drones_indices[r]
        u, v = active_start_nodes[r], target_nodes[c]
        
        try:
            path_nodes = nx.shortest_path(G, source=u, target=v, weight="weight")
            coords = centers[path_nodes]
            
            # Path includes start pos, intermediate nodes, and final target pos
            full_path = np.vstack([start_positions[original_drone_idx], coords[1:-1], target_positions[c]])
            active_drone_paths.append((original_drone_idx, full_path))
        except:
            full_path = np.vstack([start_positions[original_drone_idx], target_positions[c]])
            active_drone_paths.append((original_drone_idx, full_path))

        all_drone_path_costs[original_drone_idx] = cost_matrix[r, c]
        chain_order[c] = original_drone_idx
        
    # Create the full list of paths, including static drones
    all_drone_paths = [np.array([start_positions[i], start_positions[i]]) for i in range(num_total_drones)]
    for original_idx, path in active_drone_paths:
        all_drone_paths[original_idx] = path
        
    return all_drone_paths, target_positions, chain_order, static_count, all_drone_path_costs, max_cost, total_cost


def render_simulation(
    mode,
    drone_paths,
    final_targets,
    chain_order,
    static_count,
    path_costs,
    base_a_pos,
    base_b_pos,
    max_cost,
    total_cost,
    centers,
    features,
    xlim,
    ylim,
    duration_sec=10.0,
    fps=15
):
    """
    Renders a single simulation run and saves:
      - GIF animation (as before)
      - Temporal GeoJSON for QGIS (points per drone per frame)
    """
    frames = []
    total_frames = int(duration_sec * fps)

    # For temporal data: 1 frame = 1 second in 'simulation time'
    # (this is arbitrary but convenient for QGIS)
    start_datetime = datetime(2025, 1, 1, 0, 0, 0)

    geojson_features = []

    print(f"\nRendering {mode} version (Max Cost: {max_cost:.2f})...")

    for f in range(total_frames):
        fig, ax = plt.subplots(figsize=(12, 10), dpi=80)

        # 1. Draw Terrain
        for i, feat in enumerate(features):
            coords = feat["geometry"]["coordinates"][0]
            hs = centers[i]["score"]  # Re-use the score/color logic
            color = 'lightgreen' if hs == 1 else ('gold' if hs == 2 else 'lightcoral')
            alpha = 0.3 if hs != 3 else 0.5
            ax.add_patch(MplPolygon(coords, closed=True, fc=color, ec='gray', alpha=alpha, lw=0.5))

        # 2. Calculate Current Drone Positions
        t_global = f / (total_frames - 1) if total_frames > 1 else 1.0

        current_pos = []
        is_static_flags = []

        for i in range(len(drone_paths)):
            path_cost = path_costs[i]

            # Check if drone is static (start == end)
            is_static = np.linalg.norm(drone_paths[i][0] - drone_paths[i][-1]) < 1e-6
            is_static_flags.append(is_static)

            # Constant Power Draw Logic: Time is proportional to required energy (cost)
            if path_cost == 0 or max_cost == 0:
                t_individual = 0.0
            else:
                time_ratio = path_cost / max_cost
                t_individual = min(1.0, t_global / time_ratio)

            # Apply easing
            ease = t_individual * t_individual * (3 - 2 * t_individual)
            current_pos.append(sample_path_at_t(drone_paths[i], ease))

        current_pos = np.array(current_pos)

        # 3. Draw Connectivity (The Cyan Chain)
        chain_points = [base_a_pos] + [current_pos[idx] for idx in chain_order] + [base_b_pos]

        for k in range(len(chain_points) - 1):
            p1 = chain_points[k]
            p2 = chain_points[k + 1]

            if np.linalg.norm(p1 - p2) > 1e-6:
                dist = np.linalg.norm(p1 - p2)
                if dist < 250:
                    color = 'cyan' if dist < 180 else 'blue'
                    lw = 2.5 if dist < 180 else 1.0
                    ax.add_line(Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, zorder=20))

        # 4. Draw Drones and Battery Levels + build GeoJSON features
        current_time = start_datetime + timedelta(seconds=f)
        timestamp_str = current_time.isoformat()

        for i, pos in enumerate(current_pos):
            is_static = is_static_flags[i]
            drone_color = 'gray' if is_static else 'black'

            ax.scatter(pos[0], pos[1], c=drone_color, s=40, zorder=30, ec='white')

            path_cost = path_costs[i]

            if path_cost > 0 and max_cost > 0:
                time_progress_factor = np.clip(t_global * (max_cost / path_cost), 0, 1)
                energy_used = path_cost * time_progress_factor
            else:
                energy_used = 0.0

            remaining_battery = max(0.0, INITIAL_BATTERY - energy_used)
            text_color = 'red' if remaining_battery < (INITIAL_BATTERY * 0.2) else 'black'

            ax.text(
                pos[0] + 30,
                pos[1],
                f"D{i+1}: {remaining_battery:.1f}",
                fontsize=8,
                color=text_color,
                ha='left',
                va='center',
                zorder=40
            )

            # ---- QGIS GeoJSON feature for this drone at this frame ----
            geojson_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(pos[0]), float(pos[1])]
                },
                "properties": {
                    "kind": "drone",
                    "drone_id": int(i + 1),
                    "frame": int(f),
                    "time_s": float(f),         # 1 frame = 1 second
                    "timestamp": timestamp_str, # ISO 8601 datetime
                    "mode": mode,
                    "is_static": bool(is_static),
                    "battery": float(remaining_battery)
                }
            })

        # 5. Bases (draw + temporal features)
        ax.scatter(base_a_pos[0], base_a_pos[1], c='blue', marker='^', s=150, zorder=35, label="Base A")
        ax.scatter(base_b_pos[0], base_b_pos[1], c='purple', marker='s', s=150, zorder=35, label="Base B")

        for base_name, base_pos in [("A", base_a_pos), ("B", base_b_pos)]:
            geojson_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(base_pos[0]), float(base_pos[1])]
                },
                "properties": {
                    "kind": f"base_{base_name}",
                    "drone_id": 0,
                    "frame": int(f),
                    "time_s": float(f),
                    "timestamp": timestamp_str,
                    "mode": mode,
                    "is_static": True,
                    "battery": float(INITIAL_BATTERY)
                }
            })

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis('off')

        title = (
            f"Drone Swarm {mode} Assignment (V10)\n"
            f"Max Cost: {max_cost:.2f} | Total Cost: {total_cost:.2f} | Time: {f/total_frames:.0%}"
        )
        ax.set_title(title, fontsize=14)

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    # ---- Write GIF ----
    gif_name = f'drone_{mode.lower()}.gif'
    imageio.mimsave(gif_name, frames, fps=fps)
    print(f"Saved {gif_name}")

    # ---- Write Temporal GeoJSON for QGIS ----
    geojson_name = f'drone_{mode.lower()}_temporal.geojson'
    geojson_dict = {
        "type": "FeatureCollection",
        "features": geojson_features
    }
    with open(geojson_name, "w", encoding="utf-8") as f_out:
        json.dump(geojson_dict, f_out)

    print(f"Saved {geojson_name}")

    return gif_name, geojson_name


def run_simulation(geojson_path, num_drones=20, duration_sec=10.0, fps=15):
    # --- Setup Phase ---
    features = load_geojson(geojson_path)
    centers_pos, hex_scores = get_map_data(features)
    
    # Store centers and scores together for easy lookup in rendering
    centers = [{"pos": centers_pos[i], "score": hex_scores[i]} for i in range(len(centers_pos))]
    
    G, valid_indices = build_network_graph(centers_pos, hex_scores)
    
    base_a_node, base_b_node, chain_nodes = find_chain_path(G, centers_pos, valid_indices)
    if len(chain_nodes) < 2:
        print("Simulation Failed: No path between bases.")
        return

    num_drones = min(num_drones, len(valid_indices) - 2) # Leave out base nodes
    if num_drones <= 0:
        print("Not enough non-building hexes for bases and drones.")
        return
        
    # Targets EXCLUDE the bases, but interpolate between them.
    target_positions, base_a_pos, base_b_pos = interpolate_targets(chain_nodes, centers_pos, num_drones)
    
    # Randomly select initial drone positions
    base_nodes_set = {base_a_node, base_b_node}
    drone_start_indices = [i for i in valid_indices if i not in base_nodes_set]
    
    start_indices = random.sample(drone_start_indices, num_drones)
    start_positions = centers_pos[start_indices]
    
    # --- Execute Dual Assignments ---
    
    # 1. Sum-Min (Global Efficiency)
    paths_sum, targets_sum, chain_sum, static_sum, costs_sum, max_sum, total_sum = calculate_assignment(
        G, start_positions, target_positions, centers_pos, base_a_node, base_b_node, 'SUM_MIN'
    )
    
    # 2. Min-Max (Energy Fairness)
    paths_minmax, targets_minmax, chain_minmax, static_minmax, costs_minmax, max_minmax, total_minmax = calculate_assignment(
        G, start_positions, target_positions, centers_pos, base_a_node, base_b_node, 'MIN_MAX'
    )
    
    print("\n" + "="*70)
    print("FINAL OPTIMALITY PROOF")
    print("="*70)
    print("--- Sum-Min (Global Efficiency) ---")
    print(f"  Total Travel Cost (Sum E_i): {total_sum:.2f}")
    print(f"  Max Drone Cost (Max E_i):    {max_sum:.2f}")
    print("\n--- Min-Max (Energy Fairness) ---")
    print(f"  Total Travel Cost (Sum E_i): {total_minmax:.2f}")
    print(f"  Max Drone Cost (Max E_i):    {max_minmax:.2f} (Lowest possible worst-case drain)")
    print("======================================================================")

    # --- Render Dual Simulations ---
    
    all_x = centers_pos[:, 0]
    all_y = centers_pos[:, 1]
    pad = 50
    xlim = (min(all_x)-pad, max(all_x)+pad + 150) 
    ylim = (min(all_y)-pad, max(all_y)+pad)
    
    # Render Sum-Min (GIF + GeoJSON)
    gif_sum, gj_sum = render_simulation(
        'SUM_MIN', paths_sum, targets_sum, chain_sum, static_sum, costs_sum,
        base_a_pos, base_b_pos, max_sum, total_sum,
        centers, features, xlim, ylim, duration_sec, fps
    )

    # Render Min-Max (GIF + GeoJSON)
    gif_minmax, gj_minmax = render_simulation(
        'MIN_MAX', paths_minmax, targets_minmax, chain_minmax, static_minmax, costs_minmax,
        base_a_pos, base_b_pos, max_minmax, total_minmax,
        centers, features, xlim, ylim, duration_sec, fps
    )

    print("\nGenerated files:")
    print(f"  SUM_MIN  -> {gif_sum}, {gj_sum}")
    print(f"  MIN_MAX  -> {gif_minmax}, {gj_minmax}")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="map_with_centers.txt")
    parser.add_argument("--drones", type=int, default=20)
    args = parser.parse_args()
    
    run_simulation(args.file, num_drones=args.drones)
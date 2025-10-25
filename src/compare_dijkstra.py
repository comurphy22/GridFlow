import argparse
import csv
import json
import math
import os
import random
import heapq
from typing import Dict, List, Tuple, Optional, Set

import networkx as nx


# ----------------------------
# Utilities and graph builder
# ----------------------------

def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def closest_node(x: float, y: float, nodes: List[Dict]) -> int:
    return min(range(len(nodes)), key=lambda i: (nodes[i]["x"] - x) ** 2 + (nodes[i]["y"] - y) ** 2)


def determine_edge_keys(edge_obj: Dict) -> Tuple[str, str]:
    keys = set(edge_obj.keys())
    if {"u", "v"}.issubset(keys):
        return "u", "v"
    if {"from", "to"}.issubset(keys):
        return "from", "to"
    raise KeyError(f"Unexpected edge keys: {sorted(keys)}")


def build_graph_from_json(data: Dict) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes or not edges:
        raise ValueError("JSON missing 'nodes' or 'edges'.")

    G = nx.Graph()

    # Nodes
    for i, n in enumerate(nodes):
        if "x" in n and "y" in n:
            x, y = float(n["x"]), float(n["y"]) 
        elif "id" in n and isinstance(n["id"], (list, tuple)) and len(n["id"]) == 2:
            x, y = float(n["id"][0]), float(n["id"][1])
        else:
            raise KeyError("Node must have x/y or id [x,y].")
        G.add_node(i, x=x, y=y, blocked=bool(n.get("blocked", False)), stop_delay=float(n.get("stop_delay", 0.0)))

    from_key, to_key = determine_edge_keys(edges[0])

    def to_index(pt):
        if isinstance(pt, int):
            return pt
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            return closest_node(float(pt[0]), float(pt[1]), nodes)
        if isinstance(pt, dict) and "x" in pt and "y" in pt:
            return closest_node(float(pt["x"]), float(pt["y"]), nodes)
        raise ValueError(f"Unsupported endpoint format: {pt}")

    # Edges
    for e in edges:
        u_idx = to_index(e[from_key])
        v_idx = to_index(e[to_key])
        u_xy = (G.nodes[u_idx]["x"], G.nodes[u_idx]["y"])
        v_xy = (G.nodes[v_idx]["x"], G.nodes[v_idx]["y"])
        base_dist = distance(u_xy, v_xy)
        true_w = float(e.get("weight", base_dist))
        blocked = bool(e.get("blocked", False))
        sm = float(e.get("speed_multiplier", 1.0))
        G.add_edge(
            u_idx,
            v_idx,
            true_weight=true_w,
            belief_weight=base_dist,
            blocked=blocked,
            speed_multiplier=sm,
        )

    pos = {i: (G.nodes[i]["x"], G.nodes[i]["y"]) for i in G.nodes}
    return G, pos


# ----------------------------
# Dijkstra variants
# ----------------------------

def dijkstra_full(G: nx.Graph, s: int, t: int, skip_blocked: bool = True) -> Tuple[float, List[int]]:
    dist: Dict[int, float] = {n: math.inf for n in G.nodes}
    prev: Dict[int, Optional[int]] = {n: None for n in G.nodes}
    visited: Set[int] = set()

    dist[s] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, s)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        if u == t:
            break

        for v in G.neighbors(u):
            if v in visited:
                continue
            e = G[u][v]
            if skip_blocked and e.get("blocked", False):
                continue
            w = float(e.get("true_weight", e.get("weight", 1.0)))
            node_delay = float(G.nodes[v].get("stop_delay", 0.0))
            nd = d + w + node_delay
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if not math.isfinite(dist[t]):
        return math.inf, []
    # reconstruct
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return dist[t], path


def dijkstra_reveal(G: nx.Graph, s: int, t: int, skip_blocked: bool = True) -> Tuple[float, List[int]]:
    dist_belief: Dict[int, float] = {n: math.inf for n in G.nodes}
    dist_true: Dict[int, float] = {n: math.inf for n in G.nodes}
    prev: Dict[int, Optional[int]] = {n: None for n in G.nodes}
    visited: Set[int] = set()

    dist_belief[s] = 0.0
    dist_true[s] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, s)]

    while heap:
        d_belief, u = heapq.heappop(heap)
        if u in visited:
            continue

        # reveal the edge to u (except source)
        if prev[u] is not None:
            p = prev[u]
            e = G[p][u]
            if skip_blocked and e.get("blocked", False):
                dist_belief[u] = math.inf
                continue
            true_w = float(e.get("true_weight", e.get("weight", 1.0)))
            node_delay = float(G.nodes[u].get("stop_delay", 0.0))
            corrected = dist_true[p] + true_w + node_delay
            if corrected - d_belief > 1e-9:
                # optimistic estimate too low, push back with corrected cost
                dist_belief[u] = corrected
                heapq.heappush(heap, (corrected, u))
                continue
            dist_true[u] = corrected
            dist_belief[u] = corrected
        else:
            dist_true[u] = 0.0
            dist_belief[u] = 0.0

        visited.add(u)
        if u == t:
            break

        for v in G.neighbors(u):
            if v in visited:
                continue
            e = G[u][v]
            nd_belief = dist_belief[u] + float(e.get("belief_weight", e.get("weight", 1.0)))
            if nd_belief + 1e-12 < dist_belief[v]:
                dist_belief[v] = nd_belief
                prev[v] = u
                heapq.heappush(heap, (nd_belief, v))

    if not math.isfinite(dist_true[t]):
        return math.inf, []
    # reconstruct
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return dist_true[t], path


# ----------------------------
# Batch compare to CSV
# ----------------------------

def run_trials_to_csv(
    json_path: str,
    trials: int,
    seed: int,
    output_csv: str,
    skip_blocked: bool = True,
    ensure_connected: bool = False,
) -> None:
    with open(json_path, "r") as f:
        data = json.load(f)

    G, _ = build_graph_from_json(data)

    rng = random.Random(seed)
    nodes = list(G.nodes())

    rows = []
    attempts = 0
    i = 0
    while i < trials and attempts < trials * 10:
        attempts += 1
        a, b = rng.sample(nodes, 2)

        t_full, path_full = dijkstra_full(G, a, b, skip_blocked=skip_blocked)
        t_reveal, path_reveal = dijkstra_reveal(G, a, b, skip_blocked=skip_blocked)

        if ensure_connected and (not math.isfinite(t_full) or not math.isfinite(t_reveal)):
            # retry until both are finite up to a cap
            continue

        ax, ay = G.nodes[a]["x"], G.nodes[a]["y"]
        bx, by = G.nodes[b]["x"], G.nodes[b]["y"]
        delta = (t_reveal - t_full) if (math.isfinite(t_full) and math.isfinite(t_reveal)) else float("nan")
        rows.append({
            "trial": i + 1,
            "a_idx": a,
            "a_x": ax,
            "a_y": ay,
            "b_idx": b,
            "b_x": bx,
            "b_y": by,
            "full_time": t_full,
            "reveal_time": t_reveal,
            "delta": delta,
            "path_len_full": len(path_full),
            "path_len_reveal": len(path_reveal),
            "feasible_full": int(math.isfinite(t_full)),
            "feasible_reveal": int(math.isfinite(t_reveal)),
        })
        i += 1

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "trial","a_idx","a_x","a_y","b_idx","b_x","b_y","full_time","reveal_time","delta","path_len_full","path_len_reveal","feasible_full","feasible_reveal"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {output_csv}")


def main():
    p = argparse.ArgumentParser(description="Run multiple Dijkstra trials and write comparison CSV.")
    p.add_argument("input", help="Path to road JSON (original or obstructed)")
    p.add_argument("--trials", type=int, default=50, help="Number of random A/B trials")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output", "-o", default="compare_results.csv", help="Output CSV path")
    p.add_argument("--include-blocked", action="store_true", help="Consider blocked edges by weight instead of skipping them")
    p.add_argument("--ensure-connected", action="store_true", help="Retry picking A/B until both modes find a finite path (with a cap)")
    args = p.parse_args()

    skip_blocked = not args.include_blocked

    run_trials_to_csv(
        json_path=args.input,
        trials=args.trials,
        seed=args.seed,
        output_csv=args.output,
        skip_blocked=skip_blocked,
        ensure_connected=args.ensure_connected,
    )


if __name__ == "__main__":
    main()

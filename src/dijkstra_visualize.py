import argparse
import json
import math
import os
import random
import heapq
from typing import Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
import networkx as nx


# ----------------------------
# Utilities
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


# ----------------------------
# Loader and graph builder
# ----------------------------

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
        # Store true and optimistic (belief) weights; keep weight=true_w for compatibility
        G.add_edge(
            u_idx,
            v_idx,
            weight=true_w,
            true_weight=true_w,
            belief_weight=base_dist,
            blocked=blocked,
            speed_multiplier=sm,
        )

    pos = {i: (G.nodes[i]["x"], G.nodes[i]["y"]) for i in G.nodes}
    return G, pos


# ----------------------------
# Dijkstra implementation with step-by-step snapshots
# ----------------------------

def dijkstra_steps(G: nx.Graph, source: int, target: int, skip_blocked: bool = True, reveal_on_settle: bool = False):
    """Generator yielding snapshots of Dijkstra's algorithm state per settled node.

    Snapshot fields:
    - current: the node being settled
    - dist: dict mapping node -> best distance so far
    - prev: dict mapping node -> predecessor
    - visited: set of settled nodes
    - heap_nodes: set of nodes currently in frontier
    """
    if not reveal_on_settle:
        dist: Dict[int, float] = {n: math.inf for n in G.nodes}
        prev: Dict[int, Optional[int]] = {n: None for n in G.nodes}
        visited: Set[int] = set()

        dist[source] = 0.0
        heap: List[Tuple[float, int]] = [(0.0, source)]

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)

            # Yield snapshot after settling u
            heap_nodes = {v for _, v in heap}
            yield {
                "current": u,
                "dist": dist.copy(),
                "prev": prev.copy(),
                "visited": visited.copy(),
                "heap_nodes": heap_nodes,
            }

            if u == target:
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
        return

    # Reveal-on-settle mode: explore with optimistic edges, reveal truth when settling next node
    dist_belief: Dict[int, float] = {n: math.inf for n in G.nodes}
    dist_true: Dict[int, float] = {n: math.inf for n in G.nodes}
    prev: Dict[int, Optional[int]] = {n: None for n in G.nodes}
    visited: Set[int] = set()

    dist_belief[source] = 0.0
    dist_true[source] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, source)]

    while heap:
        d_belief, u = heapq.heappop(heap)
        if u in visited:
            continue

        # Reveal the edge used to reach u (except for source)
        if prev[u] is not None:
            p = prev[u]
            e = G[p][u]
            if skip_blocked and e.get("blocked", False):
                # Edge is actually blocked; discard u
                dist_belief[u] = math.inf
                continue
            true_w = float(e.get("true_weight", e.get("weight", 1.0)))
            node_delay = float(G.nodes[u].get("stop_delay", 0.0))
            corrected = dist_true[p] + true_w + node_delay
            if corrected - d_belief > 1e-9:
                # Our optimistic guess was too low; push back with corrected cost
                dist_belief[u] = corrected
                heapq.heappush(heap, (corrected, u))
                continue
            dist_true[u] = corrected
            dist_belief[u] = corrected
        else:
            dist_true[u] = 0.0
            dist_belief[u] = 0.0

        visited.add(u)

        # Yield snapshot after accepting u
        heap_nodes = {v for _, v in heap}
        yield {
            "current": u,
            "dist": dist_true.copy(),
            "prev": prev.copy(),
            "visited": visited.copy(),
            "heap_nodes": heap_nodes,
        }

        if u == target:
            break

        # Relax neighbors optimistically
        for v in G.neighbors(u):
            if v in visited:
                continue
            e = G[u][v]
            nd_belief = dist_belief[u] + float(e.get("belief_weight", e.get("weight", 1.0)))
            if nd_belief + 1e-12 < dist_belief[v]:
                dist_belief[v] = nd_belief
                prev[v] = u
                heapq.heappush(heap, (nd_belief, v))


def reconstruct_path(prev: Dict[int, Optional[int]], target: int) -> List[int]:
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


# ----------------------------
# Visualization
# ----------------------------

def compute_result(
    G: nx.Graph,
    s: int,
    t: int,
    skip_blocked: bool,
    reveal_on_settle: bool,
):
    last = None
    for snapshot in dijkstra_steps(G, s, t, skip_blocked=skip_blocked, reveal_on_settle=reveal_on_settle):
        last = snapshot
    if last is None:
        return math.inf, []
    dist_t = float(last["dist"].get(t, math.inf))
    path_nodes = reconstruct_path(last["prev"], t)
    return dist_t, path_nodes


def visualize_dijkstra(
    data: Dict,
    point_a: Optional[Tuple[float, float]] = None,
    point_b: Optional[Tuple[float, float]] = None,
    step_pause: float = 0.2,
    skip_blocked: bool = True,
    save_frames: Optional[str] = None,
    save_final: Optional[str] = None,
    show: bool = True,
    random_nodes: bool = False,
    seed: int = 42,
    reveal_on_settle: bool = False,
):
    G, pos = build_graph_from_json(data)

    # Snap A/B to nearest nodes
    nodes = data.get("nodes", [])
    if random_nodes:
        rng = random.Random(seed)
        if len(nodes) < 2:
            raise ValueError("Need at least two nodes to pick random A/B")
        s, t = rng.sample(range(len(nodes)), 2)
    else:
        if point_a is None or point_b is None:
            raise ValueError("Provide point_a/point_b or enable random_nodes")
        s = closest_node(point_a[0], point_a[1], nodes)
        t = closest_node(point_b[0], point_b[1], nodes)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw base graph with obstruction styling
    blocked_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("blocked", False)]
    speed_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("blocked", False) and d.get("speed_multiplier", 1.0) > 1.0]
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("blocked", False) and d.get("speed_multiplier", 1.0) == 1.0]

    if normal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color="#c0c0c0", width=0.7, alpha=0.6, ax=ax)
    if speed_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for u, v in speed_edges],
            edge_color="#ff9800",
            width=1.2,
            style="solid",
            ax=ax,
            label="Speed-reduced",
        )
    if blocked_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=blocked_edges,
            edge_color="#e53935",
            width=1.8,
            style="dashed",
            ax=ax,
            label="Blocked",
        )

    nx.draw_networkx_nodes(G, pos, node_size=6, node_color="#444444", alpha=0.85, ax=ax)

    # Mark A and B
    ax.scatter([pos[s][0]], [pos[s][1]], s=80, c="#2e7d32", marker="o", zorder=5, label="A (start)")
    ax.text(pos[s][0] + 3, pos[s][1] + 3, "A", color="#2e7d32", fontsize=10, weight="bold")
    ax.scatter([pos[t][0]], [pos[t][1]], s=80, c="#1565c0", marker="o", zorder=5, label="B (end)")
    ax.text(pos[t][0] + 3, pos[t][1] + 3, "B", color="#1565c0", fontsize=10, weight="bold")

    ax.set_title("Dijkstra exploration")
    ax.axis("equal")
    ax.axis("off")

    step_idx = 0
    if save_frames:
        os.makedirs(save_frames, exist_ok=True)

    # Colors for layers
    visited_color = "#64b5f6"  # light blue
    frontier_color = "#ff9800"  # orange
    current_color = "#ab47bc"   # purple
    path_color = "#e53935"      # red

    for snapshot in dijkstra_steps(G, s, t, skip_blocked=skip_blocked, reveal_on_settle=reveal_on_settle):
        u = snapshot["current"]
        visited = snapshot["visited"]
        heap_nodes = snapshot["heap_nodes"]

        # Overlay layers per step: clear previous overlays by re-plotting layers only
        # Visited nodes
        ax.scatter([pos[n][0] for n in visited], [pos[n][1] for n in visited], s=14, c=visited_color, zorder=6)
        # Frontier
        fn = [n for n in heap_nodes if n not in visited]
        if fn:
            ax.scatter([pos[n][0] for n in fn], [pos[n][1] for n in fn], s=12, facecolors="none", edgecolors=frontier_color, linewidths=1.5, zorder=7)
        # Current
        ax.scatter([pos[u][0]], [pos[u][1]], s=30, c=current_color, zorder=8)

        ax.set_title(f"Dijkstra exploration — settled: {len(visited)} | current: {u}")

        if save_frames:
            outpath = os.path.join(save_frames, f"step_{step_idx:04d}.png")
            fig.savefig(outpath, dpi=160)
        if show:
            plt.pause(max(0.001, step_pause))
        step_idx += 1

    # Final path
    final_prev = snapshot["prev"]  # last snapshot in loop
    path_nodes = reconstruct_path(final_prev, t)

    # Draw final path as thick red edges
    path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
    if path_edges:
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=path_color, width=2.5, ax=ax, label="Shortest path")
        ax.set_title(f"Shortest path length: {snapshot['dist'][t]:.2f}")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    if save_final:
        os.makedirs(os.path.dirname(save_final), exist_ok=True)
        fig.savefig(save_final, dpi=180)
        print(f"Saved final path figure -> {save_final}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def compare_modes(
    data: Dict,
    point_a: Optional[Tuple[float, float]] = None,
    point_b: Optional[Tuple[float, float]] = None,
    skip_blocked: bool = True,
    random_nodes: bool = False,
    seed: int = 42,
    save_final: Optional[str] = None,
    show: bool = True,
):
    G, pos = build_graph_from_json(data)

    nodes = data.get("nodes", [])
    if random_nodes:
        rng = random.Random(seed)
        if len(nodes) < 2:
            raise ValueError("Need at least two nodes to pick random A/B")
        s, t = rng.sample(range(len(nodes)), 2)
    else:
        if point_a is None or point_b is None:
            raise ValueError("Provide point_a/point_b or enable random_nodes")
        s = closest_node(point_a[0], point_a[1], nodes)
        t = closest_node(point_b[0], point_b[1], nodes)

    # Compute both modes
    t_full, path_full = compute_result(G, s, t, skip_blocked=skip_blocked, reveal_on_settle=False)
    t_reveal, path_reveal = compute_result(G, s, t, skip_blocked=skip_blocked, reveal_on_settle=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Base edges with styling
    blocked_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("blocked", False)]
    speed_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("blocked", False) and d.get("speed_multiplier", 1.0) > 1.0]
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("blocked", False) and d.get("speed_multiplier", 1.0) == 1.0]

    if normal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color="#c0c0c0", width=0.7, alpha=0.6, ax=ax)
    if speed_edges:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in speed_edges], edge_color="#ff9800", width=1.2, style="solid", ax=ax, label="Speed-reduced")
    if blocked_edges:
        nx.draw_networkx_edges(G, pos, edgelist=blocked_edges, edge_color="#e53935", width=1.8, style="dashed", ax=ax, label="Blocked")

    nx.draw_networkx_nodes(G, pos, node_size=6, node_color="#444444", alpha=0.85, ax=ax)

    # Draw A/B
    ax.scatter([pos[s][0]], [pos[s][1]], s=80, c="#2e7d32", marker="o", zorder=5, label="A (start)")
    ax.text(pos[s][0] + 3, pos[s][1] + 3, "A", color="#2e7d32", fontsize=10, weight="bold")
    ax.scatter([pos[t][0]], [pos[t][1]], s=80, c="#1565c0", marker="o", zorder=5, label="B (end)")
    ax.text(pos[t][0] + 3, pos[t][1] + 3, "B", color="#1565c0", fontsize=10, weight="bold")

    # Paths
    path_edges_full = list(zip(path_full[:-1], path_full[1:]))
    path_edges_reveal = list(zip(path_reveal[:-1], path_reveal[1:]))
    if path_edges_full:
        nx.draw_networkx_edges(G, pos, edgelist=path_edges_full, edge_color="#2e7d32", width=2.8, ax=ax, label=f"Full knowledge ({t_full:.2f})")
    if path_edges_reveal:
        nx.draw_networkx_edges(G, pos, edgelist=path_edges_reveal, edge_color="#1e88e5", width=2.2, style="dashdot", ax=ax, label=f"Reveal-on-settle ({t_reveal:.2f})")

    delta = t_reveal - t_full if math.isfinite(t_full) and math.isfinite(t_reveal) else float("nan")
    ax.set_title(f"Compare modes — full: {t_full:.2f} | reveal: {t_reveal:.2f} | Δ: {delta:.2f}")
    ax.axis("equal")
    ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    if save_final:
        os.makedirs(os.path.dirname(save_final), exist_ok=True)
        fig.savefig(save_final, dpi=180)
        print(f"Saved comparison figure -> {save_final}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def parse_point(arg_pair: List[str]) -> Tuple[float, float]:
    if len(arg_pair) != 2:
        raise argparse.ArgumentTypeError("Point must be two numbers: --a x y")
    try:
        return float(arg_pair[0]), float(arg_pair[1])
    except ValueError:
        raise argparse.ArgumentTypeError("Point coordinates must be numeric")


def main():
    parser = argparse.ArgumentParser(description="Visualize Dijkstra's algorithm from A to B on a road network JSON.")
    parser.add_argument("file", help="Path to the road JSON file")
    parser.add_argument("--a", nargs=2, metavar=("AX", "AY"), help="Point A coordinates: x y (ignored with --random-nodes)")
    parser.add_argument("--b", nargs=2, metavar=("BX", "BY"), help="Point B coordinates: x y (ignored with --random-nodes)")
    parser.add_argument("--random-nodes", action="store_true", help="Pick A and B as random nodes from the graph")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --random-nodes")
    parser.add_argument("--pause", type=float, default=0.2, help="Pause per step in seconds when showing animation")
    parser.add_argument("--no-show", action="store_true", help="Do not display the interactive animation")
    parser.add_argument("--save-frames", help="Directory to save step frames (PNG). If provided, frames are written per step.")
    parser.add_argument("--save-final", help="Path to save the final path figure (PNG)")
    parser.add_argument("--include-blocked", action="store_true", help="Consider blocked edges by weight (instead of skipping them)")
    parser.add_argument("--reveal-on-settle", action="store_true", help="Assume optimistic edges; discover obstructions only when settling the next node")
    parser.add_argument("--compare", action="store_true", help="Compare full-knowledge vs reveal-on-settle times and overlay both paths")

    args = parser.parse_args()

    with open(args.file, "r") as f:
        data = json.load(f)

    # Determine points or random mode
    p_a = parse_point(args.a) if (args.a and not args.random_nodes) else None
    p_b = parse_point(args.b) if (args.b and not args.random_nodes) else None

    if args.compare:
        compare_modes(
            data=data,
            point_a=p_a,
            point_b=p_b,
            skip_blocked=(not args.include_blocked),
            random_nodes=args.random_nodes,
            seed=args.seed,
            save_final=args.save_final,
            show=(not args.no_show),
        )
    else:
        visualize_dijkstra(
            data=data,
            point_a=p_a,
            point_b=p_b,
            step_pause=args.pause,
            skip_blocked=(not args.include_blocked),
            save_frames=args.save_frames,
            save_final=args.save_final,
            show=(not args.no_show),
            random_nodes=args.random_nodes,
            seed=args.seed,
            reveal_on_settle=args.reveal_on_settle,
        )


if __name__ == "__main__":
    main()

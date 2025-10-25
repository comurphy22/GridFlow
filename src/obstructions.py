import argparse
import json
import math
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def closest_node(x: float, y: float, nodes: List[Dict]) -> int:
    """Return index of the closest node by Euclidean distance."""
    return min(range(len(nodes)), key=lambda i: (nodes[i]["x"] - x) ** 2 + (nodes[i]["y"] - y) ** 2)


def canonical_edge_key(u_xy: Tuple[int, int], v_xy: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Canonicalize an undirected edge as a sorted tuple of endpoints (x,y)."""
    a, b = tuple(u_xy), tuple(v_xy)
    return (a, b) if a <= b else (b, a)


def determine_edge_keys(edge_obj: Dict) -> Tuple[str, str]:
    keys = edge_obj.keys()
    if "from" in keys and "to" in keys:
        return "from", "to"
    if "u" in keys and "v" in keys:
        return "u", "v"
    raise KeyError(f"Unexpected edge keys. Expected ('u','v') or ('from','to'), got: {list(keys)}")


def select_k(items: List[int], k: int, rng: random.Random) -> List[int]:
    k = max(0, min(k, len(items)))
    return rng.sample(items, k)


def generate_obstructions(
    data: Dict,
    edge_block_count: int = 0,
    edge_block_percent: float = 0.05,
    node_block_count: int = 0,
    node_block_percent: float = 0.0,
    speed_reduction_count: int = 0,
    speed_multiplier: float = 1.5,
    apply_to_weights: bool = True,
    seed: int = 42,
    # New scenarios
    tree_drops_nodes: int = 0,
    damaged_minor_count: int = 0,
    damaged_minor_multiplier: float = 1.5,
    damaged_major_count: int = 0,
    flooding_count: int = 0,
    flooding_block_threshold: float = 0.7,
    flooding_max_multiplier: float = 2.5,
    traffic_signals_count: int = 0,
    traffic_signal_delay: float = 10.0,
) -> Dict:
    """Create obstructions on top of an existing road-network-generator JSON.

    - Adds flags to edges/nodes: 'blocked': true, 'speed_multiplier': float
    - Optionally inflates weights for blocked edges and multiplies weights for speed-reduced edges
    - Returns modified data with an added top-level 'obstructions' overlay for explicit listings
    """

    rng = random.Random(seed)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes or not edges:
        return data

    from_key, to_key = determine_edge_keys(edges[0])

    # Build quick lookup structures
    node_xy = [(int(n["x"]), int(n["y"])) for n in nodes]
    node_index_by_xy = {xy: idx for idx, xy in enumerate(node_xy)}

    edge_endpoints_xy = [
        (int(e[from_key]["x"]) if isinstance(e[from_key], dict) else int(nodes[e[from_key]]["x"]),
         int(e[from_key]["y"]) if isinstance(e[from_key], dict) else int(nodes[e[from_key]]["y"]))
        for e in edges
    ]
    edge_endpoints_xy = [
        (edge_endpoints_xy[i],
         (int(edges[i][to_key]["x"]) if isinstance(edges[i][to_key], dict) else int(nodes[edges[i][to_key]]["x"]),
          int(edges[i][to_key]["y"]) if isinstance(edges[i][to_key], dict) else int(nodes[edges[i][to_key]]["y"]))
        )
        for i in range(len(edges))
    ]

    # Decide counts
    total_edges = len(edges)
    total_nodes = len(nodes)
    eb_count = edge_block_count if edge_block_count > 0 else int(round(edge_block_percent * total_edges))
    nb_count = node_block_count if node_block_count > 0 else int(round(node_block_percent * total_nodes))
    sr_count = max(0, min(speed_reduction_count, total_edges))

    all_edge_indices = list(range(total_edges))
    blocked_edge_indices = set(select_k(all_edge_indices, eb_count, rng))

    non_blocked_edge_indices = [i for i in all_edge_indices if i not in blocked_edge_indices]
    speed_reduced_indices = set(select_k(non_blocked_edge_indices, sr_count, rng))

    all_node_indices = list(range(total_nodes))
    blocked_node_indices = set(select_k(all_node_indices, nb_count, rng))

    # Apply to edges
    blocked_edges_overlay = []
    speed_reductions_overlay = []

    for i, e in enumerate(edges):
        u_xy, v_xy = edge_endpoints_xy[i]
        # Initialize fields idempotently
        if "blocked" not in e:
            e["blocked"] = False
        if "speed_multiplier" not in e:
            e["speed_multiplier"] = 1.0

        # Blocked by edge selection
        if i in blocked_edge_indices:
            e["blocked"] = True
            if apply_to_weights and "weight" in e:
                # Inflate weight to emulate closure for naive consumers
                e["weight"] = float(e["weight"]) * 1e6
            blocked_edges_overlay.append({"u": {"x": u_xy[0], "y": u_xy[1]}, "v": {"x": v_xy[0], "y": v_xy[1]}, "reason": "edge_block"})
            continue  # Don't also speed-reduce blocked edges

        # Speed reduction
        if i in speed_reduced_indices:
            e["speed_multiplier"] = max(1.0, float(speed_multiplier))
            if apply_to_weights and "weight" in e:
                e["weight"] = float(e["weight"]) * e["speed_multiplier"]
            speed_reductions_overlay.append({
                "u": {"x": u_xy[0], "y": u_xy[1]},
                "v": {"x": v_xy[0], "y": v_xy[1]},
                "multiplier": e["speed_multiplier"],
            })

    # Apply node blocks: mark node + mark incident edges as blocked
    blocked_nodes_overlay = []
    if blocked_node_indices:
        blocked_xy = {node_xy[idx] for idx in blocked_node_indices}
        for idx in blocked_node_indices:
            n = nodes[idx]
            n["blocked"] = True
            blocked_nodes_overlay.append({"x": int(n["x"]), "y": int(n["y"]), "reason": "node_block"})
        # Mark incident edges
        for i, e in enumerate(edges):
            u_xy, v_xy = edge_endpoints_xy[i]
            if u_xy in blocked_xy or v_xy in blocked_xy:
                e["blocked"] = True
                if apply_to_weights and "weight" in e:
                    e["weight"] = float(e["weight"]) * 1e6
                # Keep overlay (avoid duplicates)
                k = canonical_edge_key(u_xy, v_xy)
                if not any(canonical_edge_key((be["u"]["x"], be["u"]["y"]), (be["v"]["x"], be["v"]["y"])) == k for be in blocked_edges_overlay):
                    blocked_edges_overlay.append({"u": {"x": u_xy[0], "y": u_xy[1]}, "v": {"x": v_xy[0], "y": v_xy[1]}, "reason": "node_incident"})

    # ----------------------------
    # New scenarios
    # ----------------------------
    # Precompute edge incidence per node
    incidence: Dict[int, List[int]] = {i: [] for i in range(total_nodes)}
    edge_idx_to_node_idx = []  # (u_idx, v_idx)
    for i, (u_xy, v_xy) in enumerate(edge_endpoints_xy):
        u_idx = node_index_by_xy.get(tuple(u_xy))
        v_idx = node_index_by_xy.get(tuple(v_xy))
        if u_idx is None or v_idx is None:
            continue
        edge_idx_to_node_idx.append((u_idx, v_idx))
        incidence[u_idx].append(i)
        incidence[v_idx].append(i)

    # Tree drops: pick nodes and block all but one incident edge
    tree_drops_overlay = []
    if tree_drops_nodes > 0:
        candidate_nodes = [n for n, eidxs in incidence.items() if len(eidxs) >= 2]
        rng_nodes = rng
        pick = select_k(candidate_nodes, tree_drops_nodes, rng_nodes)
        for n_idx in pick:
            eidxs = incidence[n_idx]
            if not eidxs:
                continue
            keep = rng_nodes.choice(eidxs)
            blocked_this_node = []
            for ei in eidxs:
                if ei == keep:
                    continue
                e = edges[ei]
                u_xy, v_xy = edge_endpoints_xy[ei]
                e["blocked"] = True
                if apply_to_weights and "weight" in e:
                    e["weight"] = float(e["weight"]) * 1e6
                k = canonical_edge_key(u_xy, v_xy)
                if not any(canonical_edge_key((be["u"]["x"], be["u"]["y"]), (be["v"]["x"], be["v"]["y"])) == k for be in blocked_edges_overlay):
                    blocked_edges_overlay.append({"u": {"x": u_xy[0], "y": u_xy[1]}, "v": {"x": v_xy[0], "y": v_xy[1]}, "reason": "tree_drop"})
                blocked_this_node.append({"u": {"x": u_xy[0], "y": u_xy[1]}, "v": {"x": v_xy[0], "y": v_xy[1]}})
            tree_drops_overlay.append({"node": {"x": int(nodes[n_idx]["x"]), "y": int(nodes[n_idx]["y"])}, "kept_edge_index": keep, "blocked_edges": blocked_this_node})

    # Damaged infrastructure: minor slows, major blockages
    damaged_overlay = {"minor": [], "major": []}
    # Major
    if damaged_major_count > 0:
        available = [i for i, e in enumerate(edges) if not e.get("blocked", False)]
        for ei in select_k(available, damaged_major_count, rng):
            e = edges[ei]
            u_xy, v_xy = edge_endpoints_xy[ei]
            e["blocked"] = True
            if apply_to_weights and "weight" in e:
                e["weight"] = float(e["weight"]) * 1e6
            damaged_overlay["major"].append({"u": {"x": u_xy[0], "y": u_xy[1]}, "v": {"x": v_xy[0], "y": v_xy[1]}})
            k = canonical_edge_key(u_xy, v_xy)
            if not any(canonical_edge_key((be["u"]["x"], be["u"]["y"]), (be["v"]["x"], be["v"]["y"])) == k for be in blocked_edges_overlay):
                blocked_edges_overlay.append({"u": {"x": u_xy[0], "y": u_xy[1]}, "v": {"x": v_xy[0], "y": v_xy[1]}, "reason": "damaged_major"})
    # Minor
    if damaged_minor_count > 0:
        available = [i for i, e in enumerate(edges) if not e.get("blocked", False)]
        for ei in select_k(available, damaged_minor_count, rng):
            e = edges[ei]
            u_xy, v_xy = edge_endpoints_xy[ei]
            e["speed_multiplier"] = max(1.0, float(damaged_minor_multiplier))
            if apply_to_weights and "weight" in e:
                e["weight"] = float(e["weight"]) * e["speed_multiplier"]
            damaged_overlay["minor"].append({
                "u": {"x": u_xy[0], "y": u_xy[1]},
                "v": {"x": v_xy[0], "y": v_xy[1]},
                "multiplier": e["speed_multiplier"],
            })

    # Flooding: apply severity and possibly block
    flooding_overlay = []
    if flooding_count > 0:
        available = [i for i, e in enumerate(edges) if not e.get("blocked", False)]
        for ei in select_k(available, flooding_count, rng):
            e = edges[ei]
            u_xy, v_xy = edge_endpoints_xy[ei]
            severity = rng.random()  # 0..1
            e["flooding_level"] = severity
            if severity >= flooding_block_threshold:
                e["blocked"] = True
                if apply_to_weights and "weight" in e:
                    e["weight"] = float(e["weight"]) * 1e6
                flooding_overlay.append({
                    "u": {"x": u_xy[0], "y": u_xy[1]},
                    "v": {"x": v_xy[0], "y": v_xy[1]},
                    "severity": round(severity, 3),
                    "blocked": True,
                })
                k = canonical_edge_key(u_xy, v_xy)
                if not any(canonical_edge_key((be["u"]["x"], be["u"]["y"]), (be["v"]["x"], be["v"]["y"])) == k for be in blocked_edges_overlay):
                    blocked_edges_overlay.append({"u": {"x": u_xy[0], "y": u_xy[1]}, "v": {"x": v_xy[0], "y": v_xy[1]}, "reason": "flooding"})
            else:
                # Slowdown proportional to severity up to max multiplier
                multiplier = 1.0 + severity * (float(flooding_max_multiplier) - 1.0)
                e["speed_multiplier"] = max(float(e.get("speed_multiplier", 1.0)), multiplier)
                if apply_to_weights and "weight" in e:
                    e["weight"] = float(e["weight"]) * multiplier
                flooding_overlay.append({
                    "u": {"x": u_xy[0], "y": u_xy[1]},
                    "v": {"x": v_xy[0], "y": v_xy[1]},
                    "severity": round(severity, 3),
                    "blocked": False,
                    "multiplier": round(multiplier, 3),
                })

    # Traffic signals: choose nodes and add stop delay
    traffic_signals_overlay = []
    if traffic_signals_count > 0:
        picks = select_k(list(range(total_nodes)), traffic_signals_count, rng)
        for idx in picks:
            nodes[idx]["stop_delay"] = float(traffic_signal_delay)
            traffic_signals_overlay.append({
                "x": int(nodes[idx]["x"]),
                "y": int(nodes[idx]["y"]),
                "stop_delay": float(traffic_signal_delay),
            })

    # Compose overlay
    data.setdefault("obstructions", {})
    data["obstructions"]["blocked_edges"] = blocked_edges_overlay
    data["obstructions"]["blocked_nodes"] = blocked_nodes_overlay
    data["obstructions"]["speed_reductions"] = speed_reductions_overlay
    data["obstructions"]["tree_drops"] = tree_drops_overlay
    data["obstructions"]["damaged_infrastructure"] = damaged_overlay
    data["obstructions"]["flooding"] = flooding_overlay
    data["obstructions"]["traffic_signals"] = traffic_signals_overlay

    return data


def plot_network(data: Dict, title: str = "Road Network with Obstructions") -> None:
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes or not edges:
        print("Nothing to plot: missing nodes/edges")
        return

    # Build index-based graph for plotting
    G = nx.Graph()
    for i, n in enumerate(nodes):
        G.add_node(i, x=n["x"], y=n["y"], blocked=n.get("blocked", False))

    from_key, to_key = determine_edge_keys(edges[0])

    def find_idx(pt):
        if isinstance(pt, dict):
            return closest_node(pt["x"], pt["y"], nodes)
        return int(pt)

    for e in edges:
        u_idx = find_idx(e[from_key])
        v_idx = find_idx(e[to_key])
        G.add_edge(
            u_idx,
            v_idx,
            blocked=e.get("blocked", False),
            speed_multiplier=e.get("speed_multiplier", 1.0),
        )

    pos = {i: (G.nodes[i]["x"], G.nodes[i]["y"]) for i in G.nodes}

    plt.figure(figsize=(10, 10))
    # Draw base edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for u, v, d in G.edges(data=True) if not d.get("blocked", False)],
        edge_color="gray",
        alpha=0.6,
        width=[0.8 if d.get("speed_multiplier", 1.0) == 1.0 else 1.5 for _, _, d in G.edges(data=True) if not d.get("blocked", False)],
    )
    # Draw blocked edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for u, v, d in G.edges(data=True) if d.get("blocked", False)],
        edge_color="red",
        width=2.5,
        style="dashed",
        label="Blocked",
    )
    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[n for n in G.nodes if not G.nodes[n].get("blocked", False)],
        node_size=6,
        node_color="black",
        alpha=0.8,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[n for n in G.nodes if G.nodes[n].get("blocked", False)],
        node_size=20,
        node_color="#ff7f7f",
        label="Blocked node",
    )
    # Traffic signals (stop_delay)
    signal_nodes = [n for n in G.nodes if G.nodes[n].get("blocked", False) is False and nodes[n].get("stop_delay")]
    if signal_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=signal_nodes,
            node_size=30,
            node_color="#34b4eb",
            label="Traffic signal",
        )

    plt.legend()
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Create obstructions on a road network JSON.")
    parser.add_argument("--input", "-i", required=True, help="Path to input road JSON (e.g., road-network-generator/road3.json)")
    parser.add_argument("--output", "-o", help="Path to write modified JSON (default: <input> with _obstructed suffix)")
    parser.add_argument("--edge-blocks", type=int, default=0, help="Number of edges to block. Overrides --edge-percent if > 0")
    parser.add_argument("--edge-percent", type=float, default=0.05, help="Fraction of edges to block (0-1)")
    parser.add_argument("--node-blocks", type=int, default=0, help="Number of nodes to block")
    parser.add_argument("--node-percent", type=float, default=0.0, help="Fraction of nodes to block (0-1)")
    parser.add_argument("--speed-reductions", type=int, default=0, help="Number of edges to apply speed reduction to")
    parser.add_argument("--speed-multiplier", type=float, default=1.5, help="Multiplier > 1.0 to slow down edges")
    # New scenario args
    parser.add_argument("--tree-drops-nodes", type=int, default=0, help="Number of nodes to convert into dead ends (block all but one incident edge)")
    parser.add_argument("--damaged-minor", type=int, default=0, help="Number of edges with minor infrastructure damage (slowdown)")
    parser.add_argument("--damaged-minor-multiplier", type=float, default=1.5, help="Multiplier for minor damage slowdown")
    parser.add_argument("--damaged-major", type=int, default=0, help="Number of edges with major damage (blocked)")
    parser.add_argument("--flooding", type=int, default=0, help="Number of edges affected by flooding")
    parser.add_argument("--flood-threshold", type=float, default=0.7, help="Severity threshold [0-1] above which flooding blocks the edge")
    parser.add_argument("--flood-max-multiplier", type=float, default=2.5, help="Max slowdown multiplier for flooding when not blocked")
    parser.add_argument("--traffic-signals", type=int, default=0, help="Number of nodes to assign traffic signals to (adds stop delay)")
    parser.add_argument("--signal-delay", type=float, default=10.0, help="Stop delay cost added at signal nodes")
    parser.add_argument("--no-weight-apply", action="store_true", help="Do not modify edge weights; only annotate fields")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--plot", action="store_true", help="Visualize the network with obstructions")

    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    modified = generate_obstructions(
        data,
        edge_block_count=args.edge_blocks,
        edge_block_percent=args.edge_percent,
        node_block_count=args.node_blocks,
        node_block_percent=args.node_percent,
        speed_reduction_count=args.speed_reductions,
        speed_multiplier=args.speed_multiplier,
        apply_to_weights=(not args.no_weight_apply),
        seed=args.seed,
        tree_drops_nodes=args.tree_drops_nodes,
        damaged_minor_count=args.damaged_minor,
        damaged_minor_multiplier=args.damaged_minor_multiplier,
        damaged_major_count=args.damaged_major,
        flooding_count=args.flooding,
        flooding_block_threshold=args.flood_threshold,
        flooding_max_multiplier=args.flood_max_multiplier,
        traffic_signals_count=args.traffic_signals,
        traffic_signal_delay=args.signal_delay,
    )

    if args.plot:
        title = "Road Network with Obstructions"
        if args.edge_blocks or args.edge_percent:
            title += f" | blocked~{args.edge_blocks or args.edge_percent}"
        if args.speed_reductions:
            title += f" | speed x{args.speed_multiplier}"
        if args.node_blocks or args.node_percent:
            title += f" | nodes~{args.node_blocks or args.node_percent}"
        plot_network(modified, title=title)

    out_path = args.output
    if not out_path:
        if args.input.lower().endswith(".json"):
            out_path = args.input[:-5] + "_obstructed.json"
        else:
            out_path = args.input + "_obstructed.json"

    with open(out_path, "w") as f:
        json.dump(modified, f, indent=4)

    print(f"Saved modified network with obstructions -> {out_path}")


if __name__ == "__main__":
    main()
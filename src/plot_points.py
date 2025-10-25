import argparse
import json
import math
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


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

    for i, n in enumerate(nodes):
        if "x" in n and "y" in n:
            x, y = float(n["x"]), float(n["y"]) 
        elif "id" in n and isinstance(n["id"], (list, tuple)) and len(n["id"]) == 2:
            x, y = float(n["id"][0]), float(n["id"][1])
        else:
            raise KeyError("Node must have x/y or id [x,y].")
        G.add_node(i, x=x, y=y, blocked=bool(n.get("blocked", False)))

    from_key, to_key = determine_edge_keys(edges[0])

    def to_index(pt):
        if isinstance(pt, int):
            return pt
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            return closest_node(float(pt[0]), float(pt[1]), nodes)
        if isinstance(pt, dict) and "x" in pt and "y" in pt:
            return closest_node(float(pt["x"]), float(pt["y"]), nodes)
        raise ValueError(f"Unsupported endpoint format: {pt}")

    for e in edges:
        u_idx = to_index(e[from_key])
        v_idx = to_index(e[to_key])
        G.add_edge(u_idx, v_idx)

    pos = {i: (G.nodes[i]["x"], G.nodes[i]["y"]) for i in G.nodes}
    return G, pos


def plot_points_on_graph(
    json_path: str,
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    save: str = None,
    show: bool = True,
):
    with open(json_path, "r") as f:
        data = json.load(f)

    G, pos = build_graph_from_json(data)

    plt.figure(figsize=(8, 8))

    # Draw base network
    nx.draw_networkx_edges(G, pos, edge_color="#b0b0b0", width=0.8, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, node_size=6, node_color="#222", alpha=0.85)

    # Plot A and B
    ax, ay = point_a
    bx, by = point_b

    # Start A
    plt.scatter([ax], [ay], s=80, c="#2e7d32", marker="o", zorder=5, label="A (start)")
    plt.text(ax + 3, ay + 3, "A", color="#2e7d32", fontsize=10, weight="bold")

    # End B
    plt.scatter([bx], [by], s=80, c="#1565c0", marker="o", zorder=5, label="B (end)")
    plt.text(bx + 3, by + 3, "B", color="#1565c0", fontsize=10, weight="bold")

    plt.title(os.path.basename(json_path))
    plt.axis("equal")
    plt.axis("off")

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(loc="best")

    if save:
        # If save ends with an image extension, treat as file; else treat as directory
        if save.lower().endswith((".png", ".pdf")):
            outpath = save
        else:
            os.makedirs(save, exist_ok=True)
            base = os.path.splitext(os.path.basename(json_path))[0]
            outpath = os.path.join(save, f"{base}_points.png")
        plt.savefig(outpath, dpi=180)
        print(f"Saved plot -> {outpath}")

    if show:
        plt.show()
    else:
        plt.close()


def parse_point(arg_pair: List[str]) -> Tuple[float, float]:
    if len(arg_pair) != 2:
        raise argparse.ArgumentTypeError("Point must be two numbers: --a x y")
    try:
        return float(arg_pair[0]), float(arg_pair[1])
    except ValueError:
        raise argparse.ArgumentTypeError("Point coordinates must be numeric")


def main():
    parser = argparse.ArgumentParser(description="Plot two simple points A (start) and B (end) on a road graph JSON.")
    parser.add_argument("file", help="Path to the road JSON file")
    parser.add_argument("--a", nargs=2, metavar=("AX", "AY"), help="Point A coordinates: x y (ignored if --random-nodes)")
    parser.add_argument("--b", nargs=2, metavar=("BX", "BY"), help="Point B coordinates: x y (ignored if --random-nodes)")
    parser.add_argument("--random-nodes", action="store_true", help="Pick A and B as two random nodes from the graph")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --random-nodes")
    parser.add_argument("--save", help="Path to save PNG (file path or output directory)")
    parser.add_argument("--no-show", action="store_true", help="Do not display the window; useful with --save")

    args = parser.parse_args()

    # Load once to access nodes if random picking is requested
    with open(args.file, "r") as f:
        data = json.load(f)
    nodes = data.get("nodes", [])

    if args.random_nodes:
        rng = random.Random(args.seed)
        if len(nodes) < 2:
            raise SystemExit("Need at least two nodes to pick random A/B")
        a_idx, b_idx = rng.sample(range(len(nodes)), 2)
        ax, ay = float(nodes[a_idx]["x"]), float(nodes[a_idx]["y"])
        bx, by = float(nodes[b_idx]["x"]), float(nodes[b_idx]["y"])
    else:
        if not args.a or not args.b:
            raise SystemExit("Provide --a and --b, or use --random-nodes")
        ax, ay = parse_point(args.a)
        bx, by = parse_point(args.b)

    plot_points_on_graph(
        json_path=args.file,
        point_a=(ax, ay),
        point_b=(bx, by),
        save=args.save,
        show=(not args.no_show),
    )


if __name__ == "__main__":
    main()

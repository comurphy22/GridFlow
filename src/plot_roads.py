import argparse
import json
import math
import os
from typing import Dict, List, Tuple

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

    # Add nodes
    for i, n in enumerate(nodes):
        # Flexible node schema: either has x/y, or {id:[x,y]} style
        if "x" in n and "y" in n:
            x, y = float(n["x"]), float(n["y"]) 
        elif "id" in n and isinstance(n["id"], (list, tuple)) and len(n["id"]) == 2:
            x, y = float(n["id"][0]), float(n["id"][1])
        else:
            raise KeyError("Node must have either x/y or id [x,y].")
        G.add_node(i, x=x, y=y, blocked=bool(n.get("blocked", False)), type=n.get("type"))

    from_key, to_key = determine_edge_keys(edges[0])

    def to_index(pt):
        # pt may be an index, list/tuple [x,y], or dict {x,y}
        if isinstance(pt, (int,)):
            return int(pt)
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            return closest_node(float(pt[0]), float(pt[1]), nodes)
        if isinstance(pt, dict) and "x" in pt and "y" in pt:
            return closest_node(float(pt["x"]), float(pt["y"]), nodes)
        raise ValueError(f"Unsupported endpoint format: {pt}")

    # Add edges
    for e in edges:
        u_idx = to_index(e[from_key])
        v_idx = to_index(e[to_key])
        weight = float(e.get("weight", distance((G.nodes[u_idx]["x"], G.nodes[u_idx]["y"]), (G.nodes[v_idx]["x"], G.nodes[v_idx]["y"])) ))
        blocked = bool(e.get("blocked", False))
        sm = float(e.get("speed_multiplier", 1.0))
        G.add_edge(u_idx, v_idx, weight=weight, blocked=blocked, speed_multiplier=sm)

    pos = {i: (G.nodes[i]["x"], G.nodes[i]["y"]) for i in G.nodes}
    return G, pos


# ----------------------------
# Plotting
# ----------------------------

def plot_graph(G: nx.Graph, pos: Dict[int, Tuple[float, float]], title: str = "") -> None:
    # Determine edge lists
    blocked_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("blocked", False)]
    speed_edges = [(u, v, d) for u, v, d in G.edges(data=True) if not d.get("blocked", False) and d.get("speed_multiplier", 1.0) > 1.0]
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("blocked", False) and d.get("speed_multiplier", 1.0) == 1.0]

    # Draw edges
    if normal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color="#808080", width=0.8, alpha=0.7)
    if speed_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for u, v, _ in speed_edges],
            edge_color="#ff9800",
            width=1.5,
            style="solid",
            label="Speed-reduced",
        )
    if blocked_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=blocked_edges,
            edge_color="#e53935",
            width=2.2,
            style="dashed",
            label="Blocked",
        )

    # Draw nodes
    blocked_nodes = [n for n in G.nodes if G.nodes[n].get("blocked", False)]
    ok_nodes = [n for n in G.nodes if not G.nodes[n].get("blocked", False)]
    if ok_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=ok_nodes, node_size=8, node_color="#222222", alpha=0.9)
    if blocked_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=blocked_nodes, node_size=30, node_color="#ff7f7f", alpha=0.95, label="Blocked node")

    # Legend & title
    plt.title(title or f"Road network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(loc="best")
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot road-network JSON files (original or obstructed).")
    parser.add_argument("files", nargs="+", help="One or more JSON files to plot")
    parser.add_argument("--save", help="If provided, save plot(s) to this PNG path. If multiple files, treat as output directory.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the figure window; useful with --save.")
    args = parser.parse_args()

    multiple = len(args.files) > 1

    if multiple and args.save and (args.save.lower().endswith(".png") or args.save.lower().endswith(".pdf")):
        print("--save provided as a file path but multiple inputs given; interpreting --save as directory.")

    outdir = None
    outfile = None
    if args.save:
        if multiple:
            outdir = args.save
            os.makedirs(outdir, exist_ok=True)
        else:
            # single file
            if args.save.lower().endswith((".png", ".pdf")):
                outfile = args.save
            else:
                outdir = args.save
                os.makedirs(outdir, exist_ok=True)

    for fpath in args.files:
        with open(fpath, "r") as f:
            data = json.load(f)
        G, pos = build_graph_from_json(data)

        plt.figure(figsize=(8, 8))
        title = os.path.basename(fpath)
        plot_graph(G, pos, title=title)

        # Saving logic
        if outdir is not None:
            base = os.path.splitext(os.path.basename(fpath))[0]
            outpath = os.path.join(outdir, f"{base}.png")
            plt.savefig(outpath, dpi=180)
            print(f"Saved plot -> {outpath}")
        elif outfile is not None:
            plt.savefig(outfile, dpi=180)
            print(f"Saved plot -> {outfile}")

        if not args.no_show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import math
import os
import statistics
from typing import Dict, List, Optional, Tuple

import networkx as nx


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def build_graph_from_json(json_path: str) -> nx.Graph:
    with open(json_path, "r") as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    G = nx.Graph()
    for i, n in enumerate(nodes):
        G.add_node(i, x=float(n.get("x", 0)), y=float(n.get("y", 0)), stop_delay=float(n.get("stop_delay", 0.0)), blocked=bool(n.get("blocked", False)))
    # detect keys
    if not edges:
        return G
    k = edges[0].keys()
    if "u" in k and "v" in k:
        fk, tk = "u", "v"
    elif "from" in k and "to" in k:
        fk, tk = "from", "to"
    else:
        fk, tk = None, None
    def to_idx(pt) -> Optional[int]:
        if isinstance(pt, int):
            return pt
        if isinstance(pt, dict) and "x" in pt and "y" in pt:
            # nearest node index
            x, y = float(pt["x"]), float(pt["y"])
            return min(G.nodes, key=lambda i: (G.nodes[i]["x"]-x)**2 + (G.nodes[i]["y"]-y)**2)
        if isinstance(pt, list) and len(pt) == 2:
            x, y = float(pt[0]), float(pt[1])
            return min(G.nodes, key=lambda i: (G.nodes[i]["x"]-x)**2 + (G.nodes[i]["y"]-y)**2)
        try:
            return int(pt)
        except Exception:
            return None
    for e in edges:
        if fk is None:
            continue
        ui = to_idx(e[fk])
        vi = to_idx(e[tk])
        if ui is None or vi is None:
            continue
        G.add_edge(ui, vi,
                   blocked=bool(e.get("blocked", False)),
                   speed_multiplier=float(e.get("speed_multiplier", 1.0)),
                   weight=float(e.get("weight", 1.0)))
    return G


def summarize(rows: List[Dict[str, str]]) -> Dict:
    n = len(rows)
    ft = [to_float(r.get("full_time", "nan")) for r in rows]
    rt = [to_float(r.get("reveal_time", "nan")) for r in rows]
    dt = [to_float(r.get("delta", "nan")) for r in rows]
    pf = [int(r.get("feasible_full", 0)) for r in rows]
    pr = [int(r.get("feasible_reveal", 0)) for r in rows]
    plf = [int(r.get("path_len_full", 0)) for r in rows]
    plr = [int(r.get("path_len_reveal", 0)) for r in rows]
    euclid = [math.hypot(to_float(r.get("b_x","0")) - to_float(r.get("a_x","0")), to_float(r.get("b_y","0")) - to_float(r.get("a_y","0"))) for r in rows]

    def safe_stats(v: List[float]) -> Dict:
        w = [x for x in v if math.isfinite(x)]
        if not w:
            return {"count": 0}
        return {
            "count": len(w),
            "mean": statistics.fmean(w),
            "median": statistics.median(w),
            "stdev": statistics.pstdev(w) if len(w) > 1 else 0.0,
            "min": min(w),
            "max": max(w),
            "p10": statistics.quantiles(w, n=10)[0] if len(w) > 9 else None,
            "p90": statistics.quantiles(w, n=10)[-1] if len(w) > 9 else None,
        }

    # proportions
    num_better_full = sum(1 for i in range(n) if math.isfinite(ft[i]) and math.isfinite(rt[i]) and ft[i] <= rt[i])
    num_better_reveal = sum(1 for i in range(n) if math.isfinite(ft[i]) and math.isfinite(rt[i]) and rt[i] < ft[i])

    # collect worst deltas
    paired = [(dt[i], i) for i in range(n) if math.isfinite(dt[i])]
    paired.sort(reverse=True)
    top_worst = [rows[i] | {"rank": k+1} for k, (_, i) in enumerate(paired[:10])]

    return {
        "trials": n,
        "full_time": safe_stats(ft),
        "reveal_time": safe_stats(rt),
        "delta": safe_stats(dt),
        "feasible_full_rate": sum(pf) / n if n else 0.0,
        "feasible_reveal_rate": sum(pr) / n if n else 0.0,
        "path_len_full": safe_stats([float(x) for x in plf]),
        "path_len_reveal": safe_stats([float(x) for x in plr]),
        "euclid_distance": safe_stats(euclid),
        "full_better_or_equal": num_better_full,
        "reveal_better": num_better_reveal,
        "top_worst_delta_rows": top_worst,
    }


def write_summary_artifacts(rows: List[Dict[str, str]], summary: Dict, out_dir: str, csv_path: str, json_path: Optional[str]) -> Tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    # JSON summary
    json_out = os.path.join(out_dir, "optimization_summary.json")
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    md_out = os.path.join(out_dir, "optimization_summary.md")
    with open(md_out, "w") as f:
        f.write("# Simulation Comparison Summary\n\n")
        f.write(f"- trials: {summary['trials']}\n")
        f.write(f"- feasible_full_rate: {summary['feasible_full_rate']:.2f}\n")
        f.write(f"- feasible_reveal_rate: {summary['feasible_reveal_rate']:.2f}\n")
        if summary.get("delta", {}).get("count", 0):
            f.write(f"- delta mean: {summary['delta']['mean']:.2f} (reveal - full)\n")
            f.write(f"- delta median: {summary['delta']['median']:.2f}\n")
            f.write(f"- delta p90: {summary['delta'].get('p90','NA')}\n")
        f.write("\n## Top worst delta cases\n")
        for row in summary.get("top_worst_delta_rows", []):
            f.write(f"- rank {row['rank']}: trial {row.get('trial')} | delta={row.get('delta')} | A=({row.get('a_x')},{row.get('a_y')}) -> B=({row.get('b_x')},{row.get('b_y')})\n")

        f.write("\n## Input sources\n")
        f.write(f"- csv: {csv_path}\n")
        if json_path:
            f.write(f"- graph json: {json_path}\n")

    # LLM prompt
    prompt_out = os.path.join(out_dir, "llm_prompt.txt")
    with open(prompt_out, "w") as f:
        f.write(
            "You are 'GridFlow AI,' an expert power restoration logistics coordinator for NextEra Energy. "
            "Your task is to analyze simulation data comparing optimal routing (full knowledge of outages) vs. "
            "real-world routing (discovering outages upon arrival) and provide a DIRECT, ACTIONABLE PLAN for managers "
            "to execute post-natural disaster to achieve the LOWEST RESTORATION TIME.\n\n"
            "Prioritize actions that:\n"
            "- Restore power to critical infrastructure first (hospitals, emergency services, water treatment)\n"
            "- Address immediate safety hazards (downed lines, gas leaks, flooding)\n"
            "- Reduce outage duration for socially vulnerable areas (elderly, medical needs, low-income)\n"
            "- Minimize crew travel time and maximize productive repair time\n\n"
            "Your response must be concise, clear, and formatted in markdown with 'recommendation', 'justification', and 'predicted_impact' fields.\n\n"
        )

        f.write("## Simulation Context\n")
        f.write(json.dumps({k: summary[k] for k in ["trials","feasible_full_rate","feasible_reveal_rate","delta","euclid_distance"] if k in summary}, indent=2))

        f.write("\n\n## Critical Cases (worst restoration delays)\n")
        for row in summary.get("top_worst_delta_rows", []):
            f.write(json.dumps({
                "trial": row.get("trial"),
                "delay": row.get("delta"),
                "location_a": [row.get("a_x"), row.get("a_y")],
                "location_b": [row.get("b_x"), row.get("b_y")],
                "optimal_time": row.get("full_time"),
                "actual_time": row.get("reveal_time"),
                "optimal_stops": row.get("path_len_full"),
                "actual_stops": row.get("path_len_reveal"),
            }) + "\n")

        f.write(
            "\n## Required Output Format\n\n"
            "Provide a **POST-DISASTER ACTION PLAN** optimized for minimal restoration time. Structure your response as:\n\n"
            "### IMMEDIATE ACTIONS (Hour 0-6)\n"
            "For each action:\n"
            "- **Recommendation**: [specific action]\n"
            "- **Justification**: [why this reduces restoration time, based on simulation data]\n"
            "- **Predicted Impact**: [quantitative estimate: e.g., '15-25% reduction in median delay']\n"
            "- **Owner**: [role/team responsible]\n"
            "- **Success Metric**: [how to measure]\n\n"
            "### SHORT-TERM ACTIONS (Day 1-7)\n"
            "[Same format as above]\n\n"
            "### OPERATIONAL IMPROVEMENTS (Week 2-4)\n"
            "[Same format as above]\n\n"
            "### Key Focus Areas (address ALL):\n"
            "1. **Crew Dispatch Optimization**: How to sequence repairs to minimize travel time and prioritize critical loads\n"
            "2. **Real-Time Intelligence**: How crews should report/share outage discoveries to update routing for other teams\n"
            "3. **Contingency Routing**: Pre-planned alternate paths when primary routes are blocked\n"
            "4. **Resource Staging**: Where to position crews, equipment, materials to minimize response time\n"
            "5. **Decision Triggers**: When to replan routes vs. continue with current plan (avoid thrashing)\n"
            "6. **Communication Protocol**: How often to update dispatch, what information to share, escalation criteria\n"
            "7. **Priority Triage**: Decision matrix for which outages to address first based on criticality and accessibility\n\n"
        )

        f.write(
            "## Constraints\n"
            "- Output must be 600-800 words\n"
            "- Use the markdown format specified above with recommendation/justification/predicted_impact for each action\n"
            "- Focus on actions managers can implement IMMEDIATELY post-disaster\n"
            "- Base ALL recommendations on the simulation data provided\n"
            "- Quantify predicted impact wherever possible (% reduction in delay, time saved, etc.)\n"
            "- Prioritize non-hardware interventions (process, coordination, decision-making)\n"
            "- Include at least 8 specific, actionable recommendations across the three time horizons\n"
        )

    return json_out, md_out, prompt_out


def main():
    p = argparse.ArgumentParser(description="Summarize comparison CSV and prepare an LLM prompt for optimization insights.")
    p.add_argument("--csv", required=True, help="Path to comparison CSV (from compare_dijkstra.py)")
    p.add_argument("--json", help="Optional JSON graph path for context (not strictly required)")
    p.add_argument("--outdir", default="plots", help="Directory to write summary and prompt")
    args = p.parse_args()

    rows = load_csv(args.csv)
    summary = summarize(rows)
    json_out, md_out, prompt_out = write_summary_artifacts(rows, summary, args.outdir, args.csv, args.json)

    print("Artifacts written:")
    print("-", json_out)
    print("-", md_out)
    print("-", prompt_out)


if __name__ == "__main__":
    main()

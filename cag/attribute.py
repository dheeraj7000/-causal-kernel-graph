"""Join SASS PC-attribution to CAG SourceOp / IRNode nodes.

Inputs:
  * <run>/cag_graph.jsonl       Causal Attribution Graph from cag.link.
  * <run>/pc_map.json           PC -> (file, line) map from cag.sass_map.

Output: <run>/pc_attribution.json — for each SASS PC:
  {
    "pc": "0x130",
    "sass": "CS2R R8, SRZ",
    "file": "/.../baseline.py", "line": 26,
    "sourceop_ids": [...],            # all SourceOp nodes at that line
    "irnode_uids": [...],             # IRNode chain UIDs visiting that line
    "irnode_op_names": [...]          # tt.load, arith.addi, ...
  }

This is the substrate for Phase 2's manifested_as edges:
  IRNode -[manifested_as]-> Symptom
where the Symptom (NCU stall counter at a PC) is joined into a SassRange
node and then linked back to IRNodes via the SourceOp it shares a line with.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_graph(graph_path: Path) -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    edges: list[dict] = []
    for line in graph_path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "node" in obj:
            nodes.append(obj["node"])
        else:
            edges.append(obj["edge"])
    return nodes, edges


def attribute(run_dir: Path) -> Path:
    pc_map = json.loads((run_dir / "pc_map.json").read_text())
    graph_nodes, _ = load_graph(run_dir / "cag_graph.jsonl")

    # Index 1: (file, line) -> [SourceOp ids]
    source_by_loc: dict[tuple[str, int], list[str]] = defaultdict(list)
    sourceops_by_file: set[str] = set()
    for n in graph_nodes:
        if n["kind"] == "SourceOp":
            a = n["attrs"]
            source_by_loc[(a["file"], a["line"])].append(n["id"])
            sourceops_by_file.add(a["file"])

    # Index 2: (file, line) -> [IRNode {uid, op_name, pass_idx}]
    irnodes_by_loc: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for n in graph_nodes:
        if n["kind"] == "IRNode":
            a = n["attrs"]
            if a.get("loc_file") and a.get("loc_line"):
                irnodes_by_loc[(a["loc_file"], a["loc_line"])].append({
                    "id": n["id"],
                    "uid": a["uid"],
                    "op_name": a["op_name"],
                    "pass_idx": a["pass_idx"],
                })

    # The DWARF file column is just the basename; pc_map["full_path"] is the
    # real path. Use it as the canonical file for joins, and verify the
    # SourceOps also live in this file.
    canonical_file = pc_map["full_path"]
    if canonical_file not in sourceops_by_file:
        # Fall back: pick the unique SourceOp file (vector_add's case).
        if len(sourceops_by_file) == 1:
            canonical_file = next(iter(sourceops_by_file))

    out_rows: list[dict] = []
    matched_pcs = 0
    for a in pc_map["attributions"]:
        line = a["line"]
        if line is None:
            out_rows.append({**a, "sourceop_ids": [], "irnodes": []})
            continue
        key = (canonical_file, line)
        sops = source_by_loc.get(key, [])
        irns = irnodes_by_loc.get(key, [])
        if sops or irns:
            matched_pcs += 1
        # Aggregate distinct UIDs (a UID may appear in many IRNode versions)
        uniq_uids = sorted({i["uid"] for i in irns})
        op_names = sorted({i["op_name"] for i in irns})
        out_rows.append({
            "pc": a["pc"],
            "sass": a["sass"],
            "file": canonical_file,
            "line": line,
            "sourceop_ids": sops,
            "irnode_uids": uniq_uids,
            "irnode_op_names": op_names,
        })

    report = {
        "run": str(run_dir),
        "graph_file": str(run_dir / "cag_graph.jsonl"),
        "pc_map_file": str(run_dir / "pc_map.json"),
        "canonical_file": canonical_file,
        "total_sass_inst": pc_map["sass_inst_count"],
        "line_resolved": pc_map["coverage"]["resolved"],
        "irnode_matched_pcs": matched_pcs,
        "irnode_match_pct": (100.0 * matched_pcs / pc_map["sass_inst_count"])
                             if pc_map["sass_inst_count"] else 0.0,
        "attributions": out_rows,
    }
    out_path = run_dir / "pc_attribution.json"
    out_path.write_text(json.dumps(report, indent=2))
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    p = attribute(Path(args.run_dir))
    rep = json.loads(p.read_text())
    print(f"[+] wrote {p}")
    print(f"    SASS inst: {rep['total_sass_inst']}")
    print(f"    DWARF line-resolved: {rep['line_resolved']}")
    print(f"    Matched to IRNode/SourceOp: {rep['irnode_matched_pcs']} "
          f"({rep['irnode_match_pct']:.1f}%)")

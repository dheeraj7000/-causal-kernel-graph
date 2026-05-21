"""CAG linker: consume capture events.ndjson, emit a validated CAG.

Nodes emitted:
  SourceOp        one per (file, line, col) in any captured IR's loc table
  Pass            one per MLIR pass invocation
  Parameter       one per (knob_name, value) observed in module attrs
  LayoutEncoding  one per (#name, canonical-attrs) decl in TTGIR/LLIR
  Stage           one per Triton stage (ttir/ttgir/llir/ptx/sass)

Edges emitted:
  Pass -[transformed_to]-> Pass   (consecutive pass invocations)
  Stage -[transformed_to]-> Stage (ttir->ttgir->llir->ptx->sass)
  Parameter -[governed_by]-> Pass (first pass whose post-IR contains the attr)
  Pass -[derived_from]-> SourceOp (every source loc referenced in the pass's IR)
  Pass -[produced]-> LayoutEncoding (first pass whose IR declares the encoding)
  Pass -[lowered_to]-> Stage      (heuristic: pass's IR matches the stage IR text)

We do NOT (yet) emit:
  IRNode-level chains (transformed_to between individual ops); needs an MLIR pass
  manifested_as edges; needs NCU + DWARF PC->loc back-mapping (Phase 2).
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

from .schema import Edge, Node, cid, validate, write_jsonl
from .identity import assign_identities


# ----- IR text parsers -------------------------------------------------------

_MODULE_ATTR_RE = re.compile(
    r'"?(?P<key>ttg\.[a-zA-Z_\-]+|ttg\.target)"?\s*=\s*'
    r'(?:"(?P<sval>[^"]*)"|(?P<ival>-?\d+)\s*:\s*i32)'
)

# #name = #ttg.<kind><{ ... }>
_LAYOUT_DECL_RE = re.compile(
    r"^#(?P<name>\w+)\s*=\s*#(?P<dialect>ttg|ttng)\.(?P<kind>\w+)<\{(?P<body>.+?)\}>",
    re.MULTILINE,
)

# loc("/path/to/file.py":LINE:COL)
_LOC_FILE_RE = re.compile(r'loc\("([^"]+)":(\d+):(\d+)\)')

# tt.make_range {end = N : i32, start = 0 : i32}
_MAKE_RANGE_RE = re.compile(r"tt\.make_range\s*\{\s*end\s*=\s*(\d+)\s*:\s*i32\s*,\s*start\s*=\s*0\s*:\s*i32\s*\}")


def _canon_layout_body(body: str) -> str:
    """Normalize whitespace inside layout attrs so equivalent decls hash identically."""
    return re.sub(r"\s+", " ", body.strip())


def parse_module_attrs(ir: str) -> dict[str, str | int]:
    out: dict[str, str | int] = {}
    # The module attributes line begins with 'module attributes {...}'. Limit
    # scan to that line/block to avoid picking up identically-named keys
    # appearing elsewhere as string literals.
    mod_idx = ir.find("module attributes")
    if mod_idx < 0:
        return out
    end_idx = ir.find("{", mod_idx + len("module attributes"))
    block_end = ir.find("}", end_idx)
    if end_idx < 0 or block_end < 0:
        return out
    block = ir[end_idx + 1:block_end]
    for m in _MODULE_ATTR_RE.finditer(block):
        key = m.group("key")
        if m.group("sval") is not None:
            out[key] = m.group("sval")
        else:
            out[key] = int(m.group("ival"))
    return out


def parse_layout_decls(ir: str) -> list[dict[str, str]]:
    decls: list[dict[str, str]] = []
    for m in _LAYOUT_DECL_RE.finditer(ir):
        decls.append({
            "name": m.group("name"),
            "dialect": m.group("dialect"),
            "kind": m.group("kind"),
            "body": _canon_layout_body(m.group("body")),
        })
    return decls


def parse_source_locs(ir: str) -> list[tuple[str, int, int]]:
    seen: OrderedDict[tuple[str, int, int], None] = OrderedDict()
    for m in _LOC_FILE_RE.finditer(ir):
        key = (m.group(1), int(m.group(2)), int(m.group(3)))
        seen.setdefault(key, None)
    return list(seen.keys())


def parse_block_size(ir: str) -> int | None:
    m = _MAKE_RANGE_RE.search(ir)
    return int(m.group(1)) if m else None


# ----- Event loading ---------------------------------------------------------

def load_events(events_path: Path) -> list[dict]:
    return [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]


def _read_ir_for(event: dict, root: Path) -> str:
    p = root / event["ir_path"] if event.get("ir_path") else None
    if p is None or not p.exists():
        return ""
    return p.read_text()


# ----- Graph builder ---------------------------------------------------------

def build_graph(run_dir: Path) -> tuple[list[Node], list[Edge]]:
    events = load_events(run_dir / "events.ndjson")
    pass_events = [e for e in events if e["type"] == "pass_before"]
    stage_events = [e for e in events if e["type"] == "stage_end"]
    sass_events = [e for e in events if e["type"] == "sass_dump"]

    nodes: list[Node] = []
    edges: list[Edge] = []
    seen_ids: set[str] = set()

    def add_node(n: Node) -> None:
        if n.id in seen_ids:
            return
        nodes.append(n)
        seen_ids.add(n.id)

    # --- SourceOp + Pass + Parameter + LayoutEncoding from per-pass IRs ---
    first_seen_param: dict[tuple[str, str | int], str] = {}  # (name, val) -> pass_id
    first_seen_layout: dict[str, str] = {}                   # canonical body -> pass_id
    prev_pass_id: str | None = None

    for ev in pass_events:
        ir = _read_ir_for(ev, run_dir)
        pass_node_id = cid("Pass", {
            "idx": ev["idx"], "pass_id": ev["pass_id"], "op_type": ev["op_type"],
            "op_sym": ev.get("op_sym"), "ir_sha": ev["ir_sha256"],
        })
        add_node(Node(id=pass_node_id, kind="Pass", attrs={
            "idx": ev["idx"],
            "pass_name": ev["pass_name"],
            "pass_id": ev["pass_id"],
            "op_type": ev["op_type"],
            "op_sym": ev.get("op_sym"),
            "ir_sha256": ev["ir_sha256"],
            "ir_path": ev["ir_path"],
        }))

        # transformed_to chain across passes (chronological)
        if prev_pass_id is not None:
            edges.append(Edge(source=prev_pass_id, target=pass_node_id,
                              kind="transformed_to", attrs={"step": ev["idx"]}))
        prev_pass_id = pass_node_id

        # Parameters (module attributes) — only emit governed_by on first sighting.
        for k, v in parse_module_attrs(ir).items():
            param_id = cid("Parameter", {"name": k, "value": v})
            add_node(Node(id=param_id, kind="Parameter",
                          attrs={"name": k, "value": v, "source": "module_attr"}))
            key = (k, v)
            if key not in first_seen_param:
                first_seen_param[key] = pass_node_id
                edges.append(Edge(source=param_id, target=pass_node_id,
                                  kind="governed_by",
                                  attrs={"introduced_at_idx": ev["idx"]}))

        # BLOCK_SIZE as a Parameter — surfaced from tt.make_range
        block_size = parse_block_size(ir)
        if block_size is not None:
            param_id = cid("Parameter", {"name": "BLOCK_SIZE", "value": block_size})
            add_node(Node(id=param_id, kind="Parameter",
                          attrs={"name": "BLOCK_SIZE", "value": block_size,
                                 "source": "tt.make_range"}))
            key = ("BLOCK_SIZE", block_size)
            if key not in first_seen_param:
                first_seen_param[key] = pass_node_id
                edges.append(Edge(source=param_id, target=pass_node_id,
                                  kind="governed_by",
                                  attrs={"introduced_at_idx": ev["idx"]}))

        # LayoutEncoding decls — produced by the pass that first emits them.
        for ld in parse_layout_decls(ir):
            layout_id = cid("LayoutEncoding", {
                "dialect": ld["dialect"], "kind": ld["kind"], "body": ld["body"],
            })
            add_node(Node(id=layout_id, kind="LayoutEncoding", attrs={
                "name": ld["name"], "dialect": ld["dialect"],
                "kind": ld["kind"], "body": ld["body"],
            }))
            if ld["body"] not in first_seen_layout:
                first_seen_layout[ld["body"]] = pass_node_id
                edges.append(Edge(source=pass_node_id, target=layout_id,
                                  kind="produced",
                                  attrs={"introduced_at_idx": ev["idx"]}))

        # SourceOp derived_from — every distinct source loc referenced in this pass IR
        for f, line, col in parse_source_locs(ir):
            so_id = cid("SourceOp", {"file": f, "line": line, "col": col})
            add_node(Node(id=so_id, kind="SourceOp",
                          attrs={"file": f, "line": line, "col": col}))
            edges.append(Edge(source=pass_node_id, target=so_id, kind="derived_from"))

    # --- Stage nodes (chronological in Triton's lowering order) ---
    stage_order = ["ttir", "ttgir", "llir", "ptx"]  # cubin/sass last
    stages_by_ext: dict[str, dict] = {ev["ext"]: ev for ev in stage_events}
    if sass_events:
        stages_by_ext["sass"] = sass_events[0]
        stage_order.append("sass")

    prev_stage_id: str | None = None
    stage_id_by_ext: dict[str, str] = {}
    for ext in stage_order:
        if ext not in stages_by_ext:
            continue
        ev = stages_by_ext[ext]
        sid = cid("Stage", {"ext": ev["ext"], "name": ev["name"],
                            "sha256": ev.get("sha256")})
        stage_id_by_ext[ext] = sid
        add_node(Node(id=sid, kind="Stage", attrs={
            "ext": ev["ext"], "name": ev["name"], "path": ev["path"],
            "size_bytes": ev["size_bytes"], "sha256": ev.get("sha256"),
        }))
        if prev_stage_id is not None:
            edges.append(Edge(source=prev_stage_id, target=sid,
                              kind="transformed_to", attrs={"step": ext}))
        prev_stage_id = sid

    # --- lowered_to: heuristic, link each pass to the next stage whose end-IR
    # matches the pass's post-IR (we approximate via "first stage emitted after
    # this pass index"). Concrete mapping: passes before convert-triton-to-tritongpu
    # belong to ttir; up to add_to_llvmir belong to ttgir; the rest to llir.
    stage_boundary = {
        "convert-triton-to-tritongpu": "ttir",
        "convert-warp-specialize-to-llvm": "ttgir",  # near end of TTGIR pipeline
    }
    current_stage_ext = "ttir"
    for ev in pass_events:
        pid = ev["pass_id"]
        # When we cross a boundary pass, subsequent passes belong to the *next* stage.
        if pid in stage_boundary:
            # the boundary pass itself ends the named stage
            stage_ext_here = stage_boundary[pid]
            # find next stage
            next_idx = stage_order.index(stage_ext_here) + 1
            current_stage_ext = stage_order[next_idx] if next_idx < len(stage_order) else stage_ext_here
        # heuristic: passes producing ttg.* IR belong to ttgir; llvm.* to llir
        if "tritongpu" in pid or "ttg" in pid or "nvidia-gpu" in pid:
            stage_ext_here = "ttgir"
        elif "llvm" in pid or "convert-arith" in pid or "to-llvmir" in pid:
            stage_ext_here = "llir"
        else:
            stage_ext_here = current_stage_ext
        if stage_ext_here not in stage_id_by_ext:
            continue
        pass_node_id = cid("Pass", {
            "idx": ev["idx"], "pass_id": ev["pass_id"], "op_type": ev["op_type"],
            "op_sym": ev.get("op_sym"), "ir_sha": ev["ir_sha256"],
        })
        edges.append(Edge(source=pass_node_id, target=stage_id_by_ext[stage_ext_here],
                          kind="lowered_to"))

    # --- IRNode chains: cross-pass op identity (Python fallback) ---------
    # Each persistent op identity (uid) becomes a sequence of IRNode nodes,
    # one per pass it appears in, connected by transformed_to edges.
    # IRNode -> SourceOp derived_from is also emitted (when loc is resolved).
    ir_files = sorted(
        (int(p.name.split("_", 1)[0]), p)
        for p in (run_dir / "ir").glob("*.mlir")
    )
    chains = assign_identities(ir_files)

    # Build a quick lookup: SourceOp ids by (file, line, col), so IRNode
    # derived_from edges target the right SourceOp node we already emitted.
    sourceop_lookup: dict[tuple[str, int, int], str] = {}
    for n in nodes:
        if n.kind == "SourceOp":
            a = n.attrs
            sourceop_lookup[(a["file"], a["line"], a["col"])] = n.id

    for chain in chains:
        prev_irnode_id: str | None = None
        for v in chain.versions:
            irnode_id = cid("IRNode", {
                "uid": chain.uid,
                "pass_idx": v.pass_idx,
                "op_name": v.op_name,
            })
            add_node(Node(id=irnode_id, kind="IRNode", attrs={
                "uid": chain.uid,
                "pass_idx": v.pass_idx,
                "op_name": v.op_name,
                "ssa": v.ssa,
                "loc_file": v.resolved_loc.file,
                "loc_line": v.resolved_loc.line,
                "loc_col": v.resolved_loc.col,
                "ir_line": v.ir_line,
            }))
            if prev_irnode_id is not None:
                edges.append(Edge(
                    source=prev_irnode_id, target=irnode_id,
                    kind="transformed_to",
                    attrs={"uid": chain.uid, "to_pass": v.pass_idx},
                ))
            # derived_from edge to the matching SourceOp, if we have one.
            so_key = (v.resolved_loc.file, v.resolved_loc.line, v.resolved_loc.col)
            so_id = sourceop_lookup.get(so_key)
            if so_id is not None:
                edges.append(Edge(
                    source=irnode_id, target=so_id, kind="derived_from",
                ))
            prev_irnode_id = irnode_id

    # --- manifested_as: link SASS PCs (Phase-2 sass_map + ncu_pcsample) to
    # IRNodes/SourceOps. Optional: only fires if both files are present.
    _attach_symptoms(run_dir, nodes, edges, add_node, sourceop_lookup)

    return nodes, edges


def _attach_symptoms(
    run_dir: Path,
    nodes: list[Node],
    edges: list[Edge],
    add_node,
    sourceop_lookup: dict[tuple[str, int, int], str],
) -> None:
    """Emit SassRange + Symptom nodes and manifested_as edges.

    Inputs (any missing -> no-op):
      run_dir/pc_attribution.json   from cag.attribute (PC -> SourceOp/IRNode)
      run_dir/pcsamples.json        from cag.ncu_pcsample (per-PC stall counts)
    """
    attr_path = run_dir / "pc_attribution.json"
    pcs_path = run_dir / "pcsamples.json"
    if not (attr_path.exists() and pcs_path.exists()):
        return
    attr = json.loads(attr_path.read_text())
    pcs = json.loads(pcs_path.read_text())

    # Index PC attribution by relative offset (lowercase hex string).
    by_rel: dict[str, dict] = {}
    for a in attr["attributions"]:
        if a.get("pc"):
            by_rel[a["pc"].lower()] = a

    canonical_file = attr.get("canonical_file")

    # Find live IRNode IDs we already emitted, indexed by (file, line).
    irnode_ids_by_loc: dict[tuple[str, int], list[str]] = {}
    for n in nodes:
        if n.kind == "IRNode":
            a = n.attrs
            key = (a.get("loc_file") or "", a.get("loc_line") or 0)
            if key[0] and key[1]:
                irnode_ids_by_loc.setdefault(key, []).append(n.id)

    for sample in pcs.get("samples", []):
        rel = sample.get("pc_rel")
        if not rel:
            continue
        # SassRange: one node per PC offset (deterministic ID).
        sass_id = cid("SassRange", {
            "kernel": pcs.get("kernel"),
            "pc_rel": rel,
            "sass": sample["sass"],
        })
        add_node(Node(id=sass_id, kind="SassRange", attrs={
            "kernel": pcs.get("kernel"),
            "pc_rel": rel,
            "pc_abs": sample.get("pc_abs"),
            "sass": sample["sass"],
        }))

        # One Symptom node per (sass, stall_reason). manifested_as edges
        # link both the SassRange and any matching IRNodes to it.
        attrib = by_rel.get(rel.lower())
        line = attrib.get("line") if attrib else None
        for reason, count in sample.get("stalls", {}).items():
            if count <= 0:
                continue
            sym_id = cid("Symptom", {
                "kernel": pcs.get("kernel"),
                "pc_rel": rel,
                "reason": reason,
            })
            add_node(Node(id=sym_id, kind="Symptom", attrs={
                "kernel": pcs.get("kernel"),
                "pc_rel": rel,
                "reason": reason,
                "count": count,
                "metric": f"smsp__pcsamp_warps_issue_stalled_{reason}",
            }))
            edges.append(Edge(source=sass_id, target=sym_id,
                              kind="manifested_as"))
            # Fan out manifested_as to every IRNode at the same source line.
            if canonical_file and line is not None:
                for irn_id in irnode_ids_by_loc.get((canonical_file, line), []):
                    edges.append(Edge(source=irn_id, target=sym_id,
                                      kind="manifested_as"))


def link(run_dir: Path, output: Path | None = None) -> Path:
    nodes, edges = build_graph(run_dir)
    validate(nodes, edges)
    out = output or (run_dir / "cag_graph.jsonl")
    write_jsonl(str(out), nodes, edges)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    p = link(Path(args.run_dir), Path(args.out) if args.out else None)
    print(f"[+] graph -> {p}")

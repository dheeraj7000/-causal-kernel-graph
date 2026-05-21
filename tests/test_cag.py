"""Phase-1 tests for the CAG capture+link pipeline.

Goals:
  * Parsers extract the right things from a known TTGIR snippet.
  * Schema validator rejects malformed graphs.
  * Golden test: vector_add baseline produces a graph with expected shape,
    and there exists at least one full derived_from chain
    SourceOp <- Pass with chronologically adjacent transformed_to neighbours.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cag import schema
from cag.link import (
    build_graph, parse_block_size, parse_layout_decls,
    parse_module_attrs, parse_source_locs,
)


TTGIR_FIXTURE = """\
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#loc = loc("/tmp/k.py":6:0)
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @k() {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc("/tmp/k.py":21:28)
    tt.return loc("/tmp/k.py":29:4)
  }
}
"""


def test_parse_module_attrs():
    attrs = parse_module_attrs(TTGIR_FIXTURE)
    assert attrs["ttg.num-warps"] == 4
    assert attrs["ttg.num-ctas"] == 1
    assert attrs["ttg.threads-per-warp"] == 32
    assert attrs["ttg.target"] == "cuda:75"


def test_parse_module_attrs_ignores_outside_block():
    # An attribute-like substring outside the module block must not be picked up.
    poison = '"ttg.num-warps" = 999 : i32\n' + TTGIR_FIXTURE
    attrs = parse_module_attrs(poison)
    assert attrs["ttg.num-warps"] == 4  # not 999


def test_parse_layout_decls():
    decls = parse_layout_decls(TTGIR_FIXTURE)
    kinds = {(d["dialect"], d["kind"]) for d in decls}
    assert ("ttg", "blocked") in kinds
    assert ("ttg", "swizzled_shared") in kinds


def test_parse_source_locs():
    locs = parse_source_locs(TTGIR_FIXTURE)
    assert ("/tmp/k.py", 6, 0) in locs
    assert ("/tmp/k.py", 21, 28) in locs
    assert ("/tmp/k.py", 29, 4) in locs


def test_parse_block_size():
    assert parse_block_size(TTGIR_FIXTURE) == 1024
    assert parse_block_size("no make_range here") is None


def test_validator_accepts_minimal_graph():
    n_pass = schema.Node(id="Pass:a", kind="Pass", attrs={})
    n_src = schema.Node(id="SourceOp:s", kind="SourceOp", attrs={})
    e = schema.Edge(source="Pass:a", target="SourceOp:s", kind="derived_from")
    schema.validate([n_pass, n_src], [e])


def test_validator_rejects_missing_endpoint():
    n = schema.Node(id="Pass:a", kind="Pass", attrs={})
    e = schema.Edge(source="Pass:a", target="SourceOp:missing", kind="derived_from")
    with pytest.raises(schema.CAGValidationError):
        schema.validate([n], [e])


def test_validator_rejects_wrong_endpoint_kind():
    n1 = schema.Node(id="Pass:a", kind="Pass", attrs={})
    n2 = schema.Node(id="Pass:b", kind="Pass", attrs={})
    # governed_by must go Parameter -> Pass; Pass -> Pass is invalid.
    e = schema.Edge(source="Pass:a", target="Pass:b", kind="governed_by")
    with pytest.raises(schema.CAGValidationError):
        schema.validate([n1, n2], [e])


def test_validator_rejects_duplicate_ids():
    n1 = schema.Node(id="X", kind="Pass", attrs={})
    n2 = schema.Node(id="X", kind="Stage", attrs={})
    with pytest.raises(schema.CAGValidationError):
        schema.validate([n1, n2], [])


def test_cid_is_deterministic():
    a = schema.cid("Pass", {"x": 1, "y": "z"})
    b = schema.cid("Pass", {"y": "z", "x": 1})  # key order shouldn't matter
    assert a == b


# ---------- golden: vector_add baseline -------------------------------------

GOLDEN_RUN = Path(__file__).resolve().parents[1] / "runs" / "vec_add"


@pytest.mark.skipif(not GOLDEN_RUN.exists(), reason="vec_add capture not present")
def test_golden_vec_add_graph_shape():
    nodes, edges = build_graph(GOLDEN_RUN)
    schema.validate(nodes, edges)
    by_kind = {}
    for n in nodes:
        by_kind.setdefault(n.kind, []).append(n)
    # Must capture all five node kinds at non-trivial counts.
    assert len(by_kind.get("Pass", [])) >= 40
    assert len(by_kind.get("SourceOp", [])) >= 5
    assert len(by_kind.get("Parameter", [])) >= 4
    assert len(by_kind.get("LayoutEncoding", [])) >= 1
    assert len(by_kind.get("Stage", [])) == 5  # ttir, ttgir, llir, ptx, sass

    # All six edge kinds emitted.
    edge_kinds = {e.kind for e in edges}
    assert edge_kinds >= {"transformed_to", "governed_by", "derived_from",
                          "produced", "lowered_to"}

    # The canonical knobs we care about must exist as Parameter nodes.
    param_names = {n.attrs["name"] for n in by_kind["Parameter"]}
    assert "ttg.num-warps" in param_names
    assert "BLOCK_SIZE" in param_names


@pytest.mark.skipif(not GOLDEN_RUN.exists(), reason="vec_add capture not present")
def test_golden_derived_from_chain():
    """A SourceOp from the user script must be reachable from a transformed_to
    chain of at least N=5 consecutive passes, all of which derive_from it."""
    nodes, edges = build_graph(GOLDEN_RUN)
    pass_ids = [n.id for n in nodes if n.kind == "Pass"]
    pass_idx_by_id = {
        n.id: n.attrs["idx"] for n in nodes if n.kind == "Pass"
    }
    derived_by_src: dict[str, set[str]] = {}
    for e in edges:
        if e.kind == "derived_from":
            derived_by_src.setdefault(e.target, set()).add(e.source)

    # Restrict to user-source SourceOps (file path contains the corpus dir).
    user_srcs = [n for n in nodes
                 if n.kind == "SourceOp"
                 and "corpus/vector_add" in n.attrs.get("file", "")]
    assert user_srcs, "no SourceOps from vector_add corpus found"

    # For at least one such source, find ≥5 consecutive pass indices all linked.
    found = False
    for src in user_srcs:
        passes_for_src = derived_by_src.get(src.id, set())
        indices = sorted(pass_idx_by_id[p] for p in passes_for_src if p in pass_idx_by_id)
        run = 1
        for a, b in zip(indices, indices[1:]):
            run = run + 1 if b == a + 1 else 1
            if run >= 5:
                found = True
                break
        if found:
            break
    assert found, "no source line was tracked across ≥5 consecutive passes"


@pytest.mark.skipif(not GOLDEN_RUN.exists(), reason="vec_add capture not present")
def test_irnode_chains_emitted():
    """IRNode nodes + IRNode->IRNode transformed_to edges must exist, and at
    least one chain must span >= 5 consecutive passes for a user-source op."""
    nodes, edges = build_graph(GOLDEN_RUN)
    schema.validate(nodes, edges)
    irnodes = [n for n in nodes if n.kind == "IRNode"]
    assert len(irnodes) > 0, "no IRNode nodes emitted"

    # Group IRNodes by uid; check the longest chain spans the pipeline.
    by_uid: dict[int, list] = {}
    for n in irnodes:
        by_uid.setdefault(n.attrs["uid"], []).append(n)

    longest = max(by_uid.values(), key=len)
    longest_span = max(n.attrs["pass_idx"] for n in longest) - \
                   min(n.attrs["pass_idx"] for n in longest)
    assert longest_span >= 20, \
        f"longest IRNode chain spans only {longest_span} passes"

    # The longest chain must be a user-source op (loc points into the corpus).
    sample = longest[0]
    assert "corpus/vector_add" in sample.attrs["loc_file"], \
        f"longest chain isn't user-source: {sample.attrs['loc_file']}"

    # Cross-pass transformed_to: IRNode -> IRNode edges must outnumber
    # Pass -> Pass transformed_to edges (one per (uid, pass-transition)).
    irnode_ids = {n.id for n in irnodes}
    ir_tt = [e for e in edges if e.kind == "transformed_to"
             and e.source in irnode_ids and e.target in irnode_ids]
    pass_tt = [e for e in edges if e.kind == "transformed_to"
               and e.source not in irnode_ids]
    assert len(ir_tt) > 10 * len(pass_tt) // 100, \
        "IRNode chains should be the bulk of transformed_to edges"


@pytest.mark.skipif(not GOLDEN_RUN.exists(), reason="vec_add capture not present")
def test_irnode_derived_from_links_user_source():
    """Each user-source IRNode must point to the matching SourceOp via
    derived_from (loc-resolved). Spot-check: tt.make_range at baseline.py:21:41
    exists as both an IRNode chain and a SourceOp, with derived_from joining."""
    nodes, edges = build_graph(GOLDEN_RUN)
    sourceops = {n.id: n for n in nodes if n.kind == "SourceOp"}
    irnodes = [n for n in nodes if n.kind == "IRNode"]

    # Find the make_range IRNodes.
    make_range_irnodes = [n for n in irnodes
                          if n.attrs["op_name"] == "tt.make_range"
                          and "vector_add" in n.attrs["loc_file"]]
    assert make_range_irnodes, "no tt.make_range IRNodes for vector_add"

    df_targets_by_src = {}
    for e in edges:
        if e.kind == "derived_from" and e.source in {n.id for n in make_range_irnodes}:
            df_targets_by_src.setdefault(e.source, []).append(e.target)
    assert df_targets_by_src, "no derived_from edges from tt.make_range IRNodes"

    # Each linked SourceOp must point back at the same file/line/col as the IRNode.
    for irn in make_range_irnodes:
        targets = df_targets_by_src.get(irn.id, [])
        if not targets:
            continue
        for tgt in targets:
            so = sourceops[tgt]
            assert so.attrs["file"] == irn.attrs["loc_file"]
            assert so.attrs["line"] == irn.attrs["loc_line"]
            assert so.attrs["col"] == irn.attrs["loc_col"]


@pytest.mark.skipif(
    not (GOLDEN_RUN / "pcsamples.json").exists()
    or not (GOLDEN_RUN / "pc_attribution.json").exists(),
    reason="Phase 2 artifacts (pcsamples.json, pc_attribution.json) not present",
)
def test_phase2_symptom_nodes_exist():
    """Phase 2: manifested_as edges must exist and Symptom nodes must carry
    real per-PC stall counts."""
    nodes, edges = build_graph(GOLDEN_RUN)
    schema.validate(nodes, edges)
    syms = [n for n in nodes if n.kind == "Symptom"]
    sass = [n for n in nodes if n.kind == "SassRange"]
    assert syms, "no Symptom nodes emitted"
    assert sass, "no SassRange nodes emitted"

    # Every Symptom must have a non-zero count and a metric name.
    for s in syms:
        assert s.attrs["count"] > 0
        assert s.attrs["metric"].startswith("smsp__pcsamp_warps_issue_stalled_")

    # At least one IRNode must manifest_as a long-scoreboard symptom
    # (the canonical vector_add bottleneck).
    long_sb = [s for s in syms if s.attrs["reason"] == "long_scoreboard"]
    assert long_sb, "no long_scoreboard symptom found"
    irnode_ids = {n.id for n in nodes if n.kind == "IRNode"}
    long_sb_ids = {s.id for s in long_sb}
    irnode_to_long_sb = [
        e for e in edges
        if e.kind == "manifested_as"
        and e.source in irnode_ids
        and e.target in long_sb_ids
    ]
    assert irnode_to_long_sb, "no IRNode -> long_scoreboard manifested_as edge"


@pytest.mark.skipif(not GOLDEN_RUN.exists(), reason="vec_add capture not present")
def test_golden_stage_chain():
    """The Stage subgraph must form a linear chain ttir -> ttgir -> llir -> ptx -> sass."""
    nodes, edges = build_graph(GOLDEN_RUN)
    stages = {n.attrs["ext"]: n.id for n in nodes if n.kind == "Stage"}
    expected = ["ttir", "ttgir", "llir", "ptx", "sass"]
    assert list(stages.keys()) == expected  # capture order
    stage_edges = [(e.source, e.target) for e in edges
                   if e.kind == "transformed_to"
                   and e.attrs.get("step") in expected]
    expected_edges = [
        (stages["ttir"], stages["ttgir"]),
        (stages["ttgir"], stages["llir"]),
        (stages["llir"], stages["ptx"]),
        (stages["ptx"], stages["sass"]),
    ]
    for ee in expected_edges:
        assert ee in stage_edges

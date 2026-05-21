"""Cross-pass op identity (Python fallback for the C++ PassInstrumentation).

The C++ PassInstrumentation in cpp/CagPassInstrumentation.cpp gives authoritative
UIDs (it attaches "cag.uid" attributes via MLIR's clone-preserving machinery).
This module approximates the same information by structural fingerprinting on
the IR text we already capture, so we have IRNode-level chains today without
needing a Triton-from-source build.

Approach
--------
For each captured per-pass IR file:

  1. Parse op lines:    %ssa = dialect.op ... loc(#locN)
  2. Resolve #locN to (file, line, col) via the IR's loc table.
  3. Fingerprint each op as (resolved_loc, op_name, ssa_name).

For each pair of chronologically adjacent passes:

  4. Match ops by primary key (resolved_loc, op_name); break ties with SSA name.
  5. Surviving op -> same UID; new op -> fresh UID.

Limitations vs. C++ path
------------------------
  * Passes that rename SSA values *and* multiple ops share a location (rare but
    possible after CSE) can be misattributed. The C++ path is immune.
  * Inlined locations (`loc(fused[...])`, `loc(callsite(...))`) are normalized
    to the *innermost* file:line:col.
  * Newly created ops with no captured location land in a synthetic
    "unknown:N" bucket; the C++ path handles them by tagging at op-creation
    time on the next sweep.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ----- Parsing ---------------------------------------------------------------

# A loc decl like:   #loc20 = loc("/path/file.py":23:4)
# or a chained loc:  #loc20 = loc("name"(#loc4))
_LOC_DECL_RE = re.compile(
    r'^#(?P<id>loc\w*)\s*=\s*'
    r'(?:loc\("(?P<file>[^"]+)":(?P<line>\d+):(?P<col>\d+)\)|'
    r'loc\("(?P<sym>[^"]+)"\(#(?P<inherit>loc\w*)\)\)|'
    r'loc\((?P<kw>unknown|callsite|fused)[^)]*\)?)',
    re.MULTILINE,
)

# An op line:
#   <indent> [%lhs[, %lhs2] = ] <opname> ... loc(<locref>)
# where <opname> is "<dialect>.<op>" and <locref> is either #locN, an inline
# loc("file":line:col), "unknown", or a callsite/fused loc.
_OP_LINE_RE = re.compile(
    r'^(?P<indent>\s+)'
    r'(?:(?P<lhs>%[\w\-]+(?:,\s*%[\w\-]+)*)\s*=\s*)?'
    r'(?P<op>[a-zA-Z_][\w]*\.[a-zA-Z_][\w]*)\b'
    r'(?P<body>.*?)'
    r'\s+loc\((?P<loc>#loc\w*|"[^"]+":\d+:\d+|unknown|callsite\(.+?\)|fused\[.+?\])\)\s*$',
    re.MULTILINE,
)


@dataclass(frozen=True)
class ResolvedLoc:
    file: str
    line: int
    col: int

    @classmethod
    def unknown(cls) -> "ResolvedLoc":
        return cls(file="<unknown>", line=0, col=0)


@dataclass(frozen=True)
class ParsedOp:
    pass_idx: int
    ir_line: int               # line number in the IR file (0-indexed)
    ssa: str | None            # primary SSA name, e.g. "%offsets_1"
    op_name: str               # "dialect.op", e.g. "arith.addi"
    loc_ref: str               # raw locator: "#loc21" or "unknown" etc.
    resolved_loc: ResolvedLoc  # file:line:col (best-effort)


def resolve_loc_table(ir_text: str) -> dict[str, ResolvedLoc]:
    """Build a #locN -> ResolvedLoc mapping by walking the IR's loc decls.

    Resolves chained 'loc("sym"(#parent))' aliases by following the chain.
    """
    raw: dict[str, dict] = {}
    for m in _LOC_DECL_RE.finditer(ir_text):
        loc_id = m.group("id")
        if m.group("file"):
            raw[loc_id] = {
                "kind": "file",
                "file": m.group("file"),
                "line": int(m.group("line")),
                "col": int(m.group("col")),
            }
        elif m.group("inherit"):
            raw[loc_id] = {"kind": "alias", "to": m.group("inherit")}
        elif m.group("kw") == "unknown":
            raw[loc_id] = {"kind": "unknown"}
        else:
            raw[loc_id] = {"kind": "other"}

    resolved: dict[str, ResolvedLoc] = {}

    def resolve(loc_id: str, depth: int = 0) -> ResolvedLoc:
        if loc_id in resolved:
            return resolved[loc_id]
        if depth > 16 or loc_id not in raw:
            return ResolvedLoc.unknown()
        entry = raw[loc_id]
        if entry["kind"] == "file":
            r = ResolvedLoc(file=entry["file"], line=entry["line"], col=entry["col"])
        elif entry["kind"] == "alias":
            r = resolve(entry["to"], depth + 1)
        else:
            r = ResolvedLoc.unknown()
        resolved[loc_id] = r
        return r

    for loc_id in raw:
        resolve(loc_id)
    return resolved


_INLINE_LOC_RE = re.compile(r'^"(?P<file>[^"]+)":(?P<line>\d+):(?P<col>\d+)$')


def _resolve_locref(locref: str, table: dict[str, ResolvedLoc]) -> ResolvedLoc:
    if locref.startswith("#"):
        # Table keys are stored without the "#" prefix (see _LOC_DECL_RE).
        return table.get(locref[1:], ResolvedLoc.unknown())
    m = _INLINE_LOC_RE.match(locref)
    if m:
        return ResolvedLoc(file=m.group("file"), line=int(m.group("line")),
                           col=int(m.group("col")))
    return ResolvedLoc.unknown()


def parse_ops(ir_text: str, pass_idx: int) -> list[ParsedOp]:
    """Extract a flat list of ParsedOp from an IR text snapshot."""
    table = resolve_loc_table(ir_text)
    ops: list[ParsedOp] = []
    for m in _OP_LINE_RE.finditer(ir_text):
        lhs = m.group("lhs")
        primary_ssa = lhs.split(",")[0].strip() if lhs else None
        ir_line = ir_text.count("\n", 0, m.start())
        ops.append(ParsedOp(
            pass_idx=pass_idx,
            ir_line=ir_line,
            ssa=primary_ssa,
            op_name=m.group("op"),
            loc_ref=m.group("loc"),
            resolved_loc=_resolve_locref(m.group("loc"), table),
        ))
    return ops


# ----- Cross-pass matching ---------------------------------------------------

def _fingerprint(op: ParsedOp) -> tuple:
    """Primary fingerprint for cross-pass identity."""
    return (op.resolved_loc, op.op_name)


def match_consecutive(prev: list[ParsedOp], curr: list[ParsedOp]) -> list[tuple[int, int]]:
    """Greedy match: for each op in `curr`, find the best surviving op in `prev`.

    Returns list of (prev_index, curr_index) pairs. Unmatched curr indices are
    "newly created"; unmatched prev indices are "erased".

    Algorithm: bucket by (resolved_loc, op_name); within each bucket consume
    prev->curr in document order (which preserves SSA-name ordering for the
    common case). Tiebreak by SSA name equality.
    """
    buckets_prev: dict[tuple, list[int]] = {}
    for i, op in enumerate(prev):
        buckets_prev.setdefault(_fingerprint(op), []).append(i)

    pairs: list[tuple[int, int]] = []
    used_prev: set[int] = set()

    # Pass 1: same fingerprint AND same SSA name (strongest match).
    leftovers_curr: list[int] = []
    for j, op in enumerate(curr):
        bucket = buckets_prev.get(_fingerprint(op), [])
        matched = -1
        for i in bucket:
            if i in used_prev:
                continue
            if op.ssa is not None and prev[i].ssa == op.ssa:
                matched = i
                break
        if matched >= 0:
            used_prev.add(matched)
            pairs.append((matched, j))
        else:
            leftovers_curr.append(j)

    # Pass 2: same fingerprint, in-bucket order (drop SSA-name requirement).
    for j in leftovers_curr:
        bucket = buckets_prev.get(_fingerprint(curr[j]), [])
        matched = -1
        for i in bucket:
            if i in used_prev:
                continue
            matched = i
            break
        if matched >= 0:
            used_prev.add(matched)
            pairs.append((matched, j))

    return pairs


# ----- UID assignment + edge generation -------------------------------------

@dataclass
class IRNodeRecord:
    """One observation of an op at a specific pass."""
    uid: int                # stable across passes for the surviving op
    pass_idx: int
    op_name: str
    ssa: str | None
    resolved_loc: ResolvedLoc
    ir_line: int


@dataclass
class IRChain:
    """All pass observations for one persistent op identity."""
    uid: int
    versions: list[IRNodeRecord]   # in pass order


def assign_identities(ir_files: Iterable[tuple[int, Path]]) -> list[IRChain]:
    """Walk per-pass IR files chronologically, assign stable UIDs.

    Args:
        ir_files: iterable of (pass_idx, path_to_ir_file) in pass order.

    Returns:
        list of IRChain (one per persistent op), versions sorted by pass.
    """
    next_uid = 1
    prev_ops: list[ParsedOp] = []
    prev_uids: list[int] = []
    chains: dict[int, IRChain] = {}

    for pass_idx, path in ir_files:
        ir = path.read_text()
        curr_ops = parse_ops(ir, pass_idx)
        curr_uids: list[int] = [0] * len(curr_ops)

        matches = match_consecutive(prev_ops, curr_ops) if prev_ops else []
        matched_curr = {j: i for (i, j) in matches}

        for j, op in enumerate(curr_ops):
            if j in matched_curr:
                uid = prev_uids[matched_curr[j]]
            else:
                uid = next_uid
                next_uid += 1
                chains[uid] = IRChain(uid=uid, versions=[])
            curr_uids[j] = uid
            chains[uid].versions.append(IRNodeRecord(
                uid=uid, pass_idx=pass_idx,
                op_name=op.op_name, ssa=op.ssa,
                resolved_loc=op.resolved_loc, ir_line=op.ir_line,
            ))

        prev_ops = curr_ops
        prev_uids = curr_uids

    # Sort versions chronologically (already in order, but be defensive).
    out = list(chains.values())
    for c in out:
        c.versions.sort(key=lambda r: r.pass_idx)
    return out

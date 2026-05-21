"""CAG schema: node/edge types, ID rules, validator.

Stable IDs are derived with blake2b over canonical content. Never use Python's
builtin hash() — it is salted across processes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal


NodeKind = Literal[
    "SourceOp", "Pass", "IRNode", "Parameter", "LayoutEncoding",
    "Stage", "SassRange", "Symptom",
]

EdgeKind = Literal[
    "transformed_to",   # Pass[i] -> Pass[i+1] or Stage[i] -> Stage[i+1]
    "governed_by",      # Parameter -> Pass (the pass that first observes the param)
    "derived_from",     # Pass -> SourceOp (pass touched this source line)
    "produced",         # Pass -> LayoutEncoding (encoding first appears in pass's out-IR)
    "manifested_as",    # IRNode|Stage|SassRange -> Symptom
    "lowered_to",       # Pass -> Stage (the pass belongs to / ends this stage)
]


def cid(kind: str, content: Any) -> str:
    """Content-addressed ID. 16-byte blake2b digest, hex-encoded, prefixed by kind."""
    payload = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=10).hexdigest()
    return f"{kind}:{digest}"


@dataclass
class Node:
    id: str
    kind: NodeKind
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"node": {"id": self.id, "kind": self.kind, "attrs": self.attrs}}


@dataclass
class Edge:
    source: str
    target: str
    kind: EdgeKind
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"edge": {"source": self.source, "target": self.target,
                         "kind": self.kind, "attrs": self.attrs}}


_VALID_NODE_KINDS = set(NodeKind.__args__)  # type: ignore[attr-defined]
_VALID_EDGE_KINDS = set(EdgeKind.__args__)  # type: ignore[attr-defined]


class CAGValidationError(Exception):
    pass


def validate(nodes: list[Node], edges: list[Edge]) -> None:
    """Raise CAGValidationError on any structural problem.

    Checks: unique node IDs, valid kinds, edge endpoints exist, no edge kinds
    crossing types they aren't defined for, no self-loops on derived_from.
    """
    id_to_kind: dict[str, str] = {}
    for n in nodes:
        if n.kind not in _VALID_NODE_KINDS:
            raise CAGValidationError(f"unknown node kind: {n.kind}")
        if n.id in id_to_kind:
            raise CAGValidationError(f"duplicate node id: {n.id}")
        id_to_kind[n.id] = n.kind

    expected_endpoints: dict[str, tuple[set[str], set[str]]] = {
        "transformed_to": ({"Pass", "Stage", "IRNode"}, {"Pass", "Stage", "IRNode"}),
        "governed_by":    ({"Parameter"}, {"Pass"}),
        "derived_from":   ({"Pass", "IRNode"}, {"SourceOp"}),
        "produced":       ({"Pass"}, {"LayoutEncoding"}),
        "manifested_as":  ({"IRNode", "Stage", "SassRange"}, {"Symptom"}),
        "lowered_to":     ({"Pass"}, {"Stage"}),
    }

    for e in edges:
        if e.kind not in _VALID_EDGE_KINDS:
            raise CAGValidationError(f"unknown edge kind: {e.kind}")
        src_kind = id_to_kind.get(e.source)
        tgt_kind = id_to_kind.get(e.target)
        if src_kind is None:
            raise CAGValidationError(f"edge source missing: {e.source}")
        if tgt_kind is None:
            raise CAGValidationError(f"edge target missing: {e.target}")
        allowed_src, allowed_tgt = expected_endpoints[e.kind]
        if src_kind not in allowed_src:
            raise CAGValidationError(
                f"edge {e.kind}: source kind {src_kind} not in {allowed_src}")
        if tgt_kind not in allowed_tgt:
            raise CAGValidationError(
                f"edge {e.kind}: target kind {tgt_kind} not in {allowed_tgt}")
        if e.kind == "derived_from" and e.source == e.target:
            raise CAGValidationError(f"self-loop derived_from: {e.source}")


def write_jsonl(path: str, nodes: list[Node], edges: list[Edge]) -> None:
    with open(path, "w") as fh:
        for n in nodes:
            fh.write(json.dumps(n.to_json()) + "\n")
        for e in edges:
            fh.write(json.dumps(e.to_json()) + "\n")

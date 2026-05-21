"""Extract per-SASS-PC stall samples from an NCU report.

This is Phase-2's "Symptom" capture path. NCU's CSV mode flattens per-PC
data into aggregated values; the binary .ncu-rep keeps the per-PC
samples. We read it with NCU's Python SDK (ships with Nsight Compute
under <install>/extras/python/ncu_report.py) and emit:

  {
    "kernel": "vector_add_kernel",
    "base_pc": "0x...",          # runtime addr of cubin offset 0x0
    "samples": [
      {"pc_abs": "0x...", "pc_rel": "0x190", "sass": "FADD R16,R16,R12",
       "stalls": {"long_scoreboard": 63, "membar": 0, ...}},
      ...
    ]
  }

The "pc_rel" offsets join directly to cag.sass_map (which uses cubin
offsets) and through it to the CAG's IRNode/SourceOp nodes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


# Canonical Nsight install layout. Lightning ships 2025.3.1; override with
# NCU_REPORT_PATH or NCU_INSTALL_DIR if needed.
_DEFAULT_SDK_DIRS = (
    os.environ.get("NCU_REPORT_PATH"),
    "/opt/nvidia/nsight-compute/2025.3.1/extras/python",
    "/opt/nvidia/nsight-compute/2025.2.0/extras/python",
    "/opt/nvidia/nsight-compute/2025.1.1/extras/python",
)


def _import_ncu_report():
    for d in _DEFAULT_SDK_DIRS:
        if d and Path(d, "ncu_report.py").is_file():
            if d not in sys.path:
                sys.path.insert(0, d)
            import ncu_report  # noqa: F401
            return ncu_report
    raise RuntimeError(
        "ncu_report module not found. Set NCU_REPORT_PATH to the directory "
        "containing ncu_report.py (typically "
        "/opt/nvidia/nsight-compute/<version>/extras/python)."
    )


# All per-PC warp-stall-reason metrics produced by NCU's PC sampling.
# We strip the smsp__pcsamp_warps_issue_stalled_ prefix when emitting.
STALL_METRIC_PREFIX = "smsp__pcsamp_warps_issue_stalled_"
_NON_PC_METRIC_SUFFIXES = ("_not_issued",)  # mirror counter, redundant


def find_kernel_base(act, candidate_pcs: Iterable[int]) -> int | None:
    """Find the lowest PC where sass_by_pc returns non-empty text.

    NCU per-PC values are absolute runtime addresses. To map back to cubin
    offsets we subtract this base.
    """
    candidates = sorted(set(candidate_pcs))
    if not candidates:
        return None
    base = candidates[0]
    # Walk backward until sass_by_pc returns empty -- the boundary is the base.
    step = 0x10
    while base - step >= 0:
        s = act.sass_by_pc(base - step)
        if not s:
            break
        base -= step
    return base


def extract(report_path: Path, action_index: int = 0) -> dict:
    ncr = _import_ncu_report()
    r = ncr.load_report(str(report_path))
    if r.num_ranges() == 0:
        return {"error": "report has no ranges"}
    rng = r.range_by_idx(0)
    if rng.num_actions() <= action_index:
        return {"error": f"range has {rng.num_actions()} actions, need {action_index+1}"}
    act = rng.action_by_idx(action_index)
    kernel = act.name(act.NameBase_DEMANGLED)

    # Discover all per-PC stall metrics actually in this report.
    metric_names = [
        n for n in act.metric_names()
        if n.startswith(STALL_METRIC_PREFIX)
        and not any(n.endswith(suf) for suf in _NON_PC_METRIC_SUFFIXES)
        and not n.startswith("group:")
    ]

    # First pass: collect every PC referenced by any stall metric.
    pcs_seen: set[int] = set()
    per_pc: dict[int, dict[str, int]] = {}
    for name in metric_names:
        m = act.metric_by_name(name)
        if not m.has_correlation_ids() or m.num_instances() == 0:
            continue
        cids = m.correlation_ids()
        short = name[len(STALL_METRIC_PREFIX):]
        for i in range(m.num_instances()):
            pc = cids.as_uint64(i)
            val = m.as_uint64(i)
            pcs_seen.add(pc)
            if val > 0:
                per_pc.setdefault(pc, {})[short] = val

    base = find_kernel_base(act, pcs_seen)
    samples: list[dict] = []
    for pc, stalls in sorted(per_pc.items()):
        rel = pc - base if base is not None else None
        sass = act.sass_by_pc(pc) or ""
        samples.append({
            "pc_abs": f"0x{pc:x}",
            "pc_rel": (f"0x{rel:x}" if rel is not None else None),
            "sass": sass.strip(),
            "stalls": stalls,
            "total_stalls": sum(stalls.values()),
        })

    # Top-N rollup so the consumer sees the bottleneck at a glance.
    top = sorted(samples, key=lambda s: s["total_stalls"], reverse=True)[:10]

    return {
        "report": str(report_path),
        "kernel": kernel,
        "base_pc": f"0x{base:x}" if base is not None else None,
        "stall_metrics_present": metric_names,
        "samples": samples,
        "top_by_total_stalls": [
            {"pc_rel": s["pc_rel"], "sass": s["sass"],
             "stalls": s["stalls"], "total": s["total_stalls"]}
            for s in top
        ],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ncu_rep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = extract(Path(args.ncu_rep))
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[+] wrote {args.out}")
    if "error" in out:
        print("[!]", out["error"])
        sys.exit(1)
    print(f"kernel={out['kernel']}  base={out['base_pc']}  "
          f"samples={len(out['samples'])}  metrics={len(out['stall_metrics_present'])}")
    for s in out["top_by_total_stalls"]:
        print(f"  rel={s['pc_rel']:<6}  total={s['total']:<5}  "
              f"{', '.join(f'{k}={v}' for k,v in s['stalls'].items()):<40s}  "
              f"{s['sass']}")

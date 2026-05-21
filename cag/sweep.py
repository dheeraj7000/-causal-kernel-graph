"""BLOCK_SIZE sweep: real NCU measurements across a knob.

Replaces Intervention_Engine._simulate_ncu_data. For each value of the named
knob:
  1. Generate a perturbed script.
  2. Run cag.capture to get the CAG.
  3. Run cag.ncu to get real hardware metrics.
  4. Write a row to sweep.csv with knob value + selected metrics.

Validity oracle: if Triton refuses to compile (e.g., BLOCK_SIZE > 1024 ×
num_warps × element_count, illegal layout) we mark the row INVALID and
continue. The point of the sweep is to map valid knob points to symptoms;
illegal points get skipped, not heuristically extrapolated.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .capture import CaptureConfig, capture
from .ncu import NcuConfig, collect as ncu_collect, parse_csv, summarize


REPORT_METRICS = (
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "smsp__average_warp_latency_per_inst_issued.ratio",
    "smsp__warp_cycles_per_issued_instruction.ratio",
    "smsp__inst_executed.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
)


def perturb_script(src: Path, knob: str, value: int, dst: Path) -> None:
    """Rewrite a single line `<knob>=<digits>` in src and write to dst."""
    content = src.read_text()
    new = re.sub(rf"\b{re.escape(knob)}\s*=\s*\d+", f"{knob}={value}", content)
    dst.write_text(new)


def run_one(script: Path, out_dir: Path, kernel_regex: str,
            launch_count: int) -> tuple[bool, dict[str, float | None], str]:
    """Run capture + ncu for a single script. Returns (valid, metrics_by_metric, note)."""
    # Capture (best-effort; if Triton refuses, mark invalid).
    cap_dir = out_dir / "capture"
    try:
        capture(CaptureConfig(script=script, out_dir=cap_dir))
    except subprocess.TimeoutExpired:
        return False, {}, "triton_compile_timeout"
    # If capture wrote an error event, treat as invalid.
    events = (cap_dir / "events.ndjson").read_text().splitlines()
    has_error = any(json.loads(l).get("type") == "error" for l in events if l.strip())
    if has_error:
        return False, {}, "triton_compile_error"

    # NCU
    ncu_dir = out_dir / "ncu"
    meta = ncu_collect(NcuConfig(
        script=script, out_dir=ncu_dir, label="run",
        kernel_regex=kernel_regex,
        launch_skip=0, launch_count=launch_count,
    ))
    rows = parse_csv(Path(meta["csv_path"]))
    summary = summarize(rows)
    # If the filter matched no kernel, summary is empty.
    if not summary:
        return False, {}, "ncu_no_kernel_matched"
    # Pick the kernel matching the regex (in case of false positives we still
    # get the right one).
    target = next(
        (k for k in summary if re.search(kernel_regex, k)),
        next(iter(summary.keys())),
    )
    metrics = {m: summary[target].get(m) for m in REPORT_METRICS}
    return True, metrics, target


def sweep(script: Path, knob: str, values: list[int], out_root: Path,
          kernel_regex: str, launch_count: int = 3) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    for v in values:
        label = f"{knob}_{v}"
        run_dir = out_root / label
        run_dir.mkdir(exist_ok=True)
        perturbed = run_dir / "perturbed.py"
        perturb_script(script, knob, v, perturbed)

        print(f"[*] {label}: capture + ncu …", flush=True)
        ok, metrics, target_kernel = run_one(perturbed, run_dir, kernel_regex,
                                             launch_count)
        row = {"knob": knob, "value": v, "valid": ok, "kernel": target_kernel}
        for m in REPORT_METRICS:
            row[m] = metrics.get(m) if ok else None
        summary_rows.append(row)
        print(f"    {'OK ' if ok else 'SKIP'} {target_kernel}", flush=True)

    sweep_csv = out_root / "sweep.csv"
    fieldnames = ["knob", "value", "valid", "kernel"] + list(REPORT_METRICS)
    with sweep_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[+] sweep summary -> {sweep_csv}")
    return sweep_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--knob", default="BLOCK_SIZE")
    ap.add_argument("--values", default="32,128,512,1024,2048",
                    help="comma-separated knob values")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", required=True,
                    help="kernel-name pattern for ncu filter")
    ap.add_argument("--launch-count", type=int, default=3)
    args = ap.parse_args()
    sweep(
        script=Path(args.script), knob=args.knob,
        values=[int(v) for v in args.values.split(",")],
        out_root=Path(args.out), kernel_regex=args.kernel,
        launch_count=args.launch_count,
    )

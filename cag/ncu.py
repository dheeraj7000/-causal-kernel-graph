"""Real NCU (Nsight Compute) measurement collector.

Replaces the synthetic _simulate_ncu_data heuristic that produced fake numbers
in paper.tex Table 1. Runs ncu as root (on machines where perfcounter access
is gated by NVreg_RestrictProfilingToAdminUsers=1, which is the default on
Lightning Studios and most stock NVIDIA driver installs).

Output for each run:
    runs/<k>/ncu/<label>.ncu-rep      binary report (replay later in ncu-ui)
    runs/<k>/ncu/<label>.csv          metrics in CSV form
    runs/<k>/ncu/<label>.meta.json    {script, sudo, ncu_version, returncode, cmd}

Metric set: NCU's named sections are more stable than hand-picked counter
names across NCU versions. We collect the four sections that map to the
CAG's HardwareSymptom node kinds:

    Occupancy            achieved_active_warps_per_sm, theoretical, ...
    WarpStateStats       per-stall-reason percentages
    MemoryWorkloadAnalysis  L1/L2/DRAM throughput, bank conflicts
    LaunchStats          grid, block, register count, SMEM, occupancy

Stall counters of interest:
    smsp__pcsamp_warps_issue_stalled_long_scoreboard            (memory latency)
    smsp__pcsamp_warps_issue_stalled_short_scoreboard           (register dep)
    smsp__pcsamp_warps_issue_stalled_membar                     (memory barrier)
    smsp__pcsamp_warps_issue_stalled_lg_throttle                (LSU throttle)
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum    (real bank conflicts)
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


# Curated, stable metric set. We avoid `--section` for two reasons: section
# bundles change across NCU minor versions, and they emit hundreds of rows
# per kernel which is wasteful for our use. Each metric here is documented
# in NVIDIA's Kernel Profiling Guide and has been stable since NCU 2022.x.
DEFAULT_METRICS: tuple[str, ...] = (
    # Launch / occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__cycles_active.avg",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    # Warp stall reasons (kernel-replay compatible). The smsp__pcsamp_* family
    # needs --replay-mode application; we stick with cycle-level stall pcts.
    "smsp__average_warp_latency_per_inst_issued.ratio",
    "smsp__warp_cycles_per_issued_instruction.ratio",
    "smsp__inst_executed.sum",
    # Memory: throughput, bank conflicts
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "smsp__sass_thread_inst_executed_op_shared_ld_sum.sum",
    "smsp__sass_thread_inst_executed_op_shared_st_sum.sum",
)


@dataclasses.dataclass
class NcuConfig:
    script: Path
    out_dir: Path
    label: str = "baseline"
    ncu_path: str = "/usr/local/cuda/bin/ncu"
    python: str = dataclasses.field(default_factory=lambda: sys.executable)
    kernel_regex: str | None = None       # --kernel-name regex; None = all
    launch_skip: int = 0                  # --launch-skip
    launch_count: int = 1                 # --launch-count
    replay_mode: str = "kernel"           # kernel|range|application
    cache_control: str = "none"           # none|all (none = invalidate L2 between replays)
    sudo: bool = True
    metrics: Iterable[str] = dataclasses.field(default_factory=lambda: DEFAULT_METRICS)
    extra_args: tuple[str, ...] = ()
    timeout_s: int = 600

    def __post_init__(self) -> None:
        self.script = Path(self.script).resolve()
        self.out_dir = Path(self.out_dir).resolve()


def _build_cmd(cfg: NcuConfig) -> tuple[list[str], Path, Path | None]:
    # NOTE: --export <file> suppresses CSV stdout output in NCU 2025.3+, so we
    # use it only in a second pass to materialize the binary report. The CSV
    # is what the linker consumes anyway.
    csv_path = cfg.out_dir / f"{cfg.label}.csv"
    cmd: list[str] = []
    if cfg.sudo:
        cmd += ["sudo", "-n", "env", f"PATH={os.environ.get('PATH', '')}"]
    cmd.append(cfg.ncu_path)
    cmd += ["--csv"]
    cmd += ["--replay-mode", cfg.replay_mode]
    cmd += ["--cache-control", cfg.cache_control]
    cmd += ["--launch-skip", str(cfg.launch_skip)]
    cmd += ["--launch-count", str(cfg.launch_count)]
    cmd += ["--metrics", ",".join(cfg.metrics)]
    if cfg.kernel_regex is not None:
        # NCU's --kernel-name treats the value as a regex when it contains
        # metacharacters; passing the literal "regex:" prefix is rejected by
        # 2025.3+ versions. We always pass the bare pattern.
        cmd += ["--kernel-name", cfg.kernel_regex]
    cmd += list(cfg.extra_args)
    cmd += [cfg.python, str(cfg.script)]
    return cmd, csv_path, None


def collect(cfg: NcuConfig) -> dict:
    """Run NCU once. Returns a dict with cmd, returncode, csv_path, rep_path, meta."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cmd, csv_path, _ = _build_cmd(cfg)
    rep_path = cfg.out_dir / f"{cfg.label}.ncu-rep"

    env = os.environ.copy()
    # NCU itself doesn't need TRITON_KERNEL_DUMP; but if user-script imports
    # cag.capture, leave their env alone.
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=cfg.timeout_s)
    csv_path.write_text(proc.stdout)

    # NCU version (best-effort)
    ncu_ver = "unknown"
    try:
        v = subprocess.run([cfg.ncu_path, "--version"], capture_output=True,
                           text=True, timeout=10).stdout.splitlines()
        for line in v:
            if line.strip().startswith("Version"):
                ncu_ver = line.strip()
                break
    except Exception:
        pass

    meta = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-2000:],
        "ncu_version": ncu_ver,
        "csv_path": str(csv_path),
        "rep_path": str(rep_path),
        "script": str(cfg.script),
        "label": cfg.label,
        "kernel_regex": cfg.kernel_regex,
    }
    (cfg.out_dir / f"{cfg.label}.meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def parse_csv(csv_path: Path) -> list[dict]:
    """Parse NCU CSV into a list of {kernel, metric_name, metric_unit, metric_value}.

    NCU CSV columns (since 2022): ID, Process ID, Process Name, Host Name,
    Kernel Name, Context, Stream, Block Size, Grid Size, Device, CC, Section
    Name, Metric Name, Metric Unit, Metric Value.
    """
    import csv
    rows: list[dict] = []
    # NCU 2025.3+ interleaves status lines like '==WARNING==' and '==PROF=='
    # into stdout before the CSV body. Strip everything until the real header.
    raw = csv_path.read_text().splitlines()
    header_idx = next(
        (i for i, line in enumerate(raw)
         if line.startswith('"ID"') or line.startswith("ID,")),
        None,
    )
    if header_idx is None:
        return rows
    csv_body = "\n".join(raw[header_idx:])
    import io
    reader = csv.reader(io.StringIO(csv_body))
    try:
        header = next(reader)
    except StopIteration:
        return rows
    if True:
        # Build index by best-known column names.
        idx = {name: i for i, name in enumerate(header)}
        for row in reader:
            if not row or len(row) < 13:
                continue
            try:
                kernel = row[idx["Kernel Name"]]
                metric_name = row[idx["Metric Name"]]
                metric_unit = row[idx.get("Metric Unit", -1)] if "Metric Unit" in idx else ""
                metric_value = row[idx["Metric Value"]]
            except (KeyError, IndexError):
                continue
            # NCU writes "n/a" when a metric is unavailable; keep these so
            # downstream code can flag them, but coerce a numeric column too.
            value_num = None
            try:
                value_num = float(metric_value.replace(",", ""))
            except ValueError:
                pass
            rows.append({
                "kernel": kernel,
                "metric": metric_name,
                "unit": metric_unit,
                "value_raw": metric_value,
                "value": value_num,
            })
    return rows


def summarize(rows: list[dict]) -> dict[str, dict[str, float | None]]:
    """Group rows by kernel and metric, return {kernel: {metric: value}}."""
    out: dict[str, dict] = {}
    for r in rows:
        out.setdefault(r["kernel"], {})[r["metric"]] = r["value"]
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--kernel", default=None, help="kernel-name regex filter")
    ap.add_argument("--no-sudo", action="store_true")
    ap.add_argument("--launch-skip", type=int, default=0)
    ap.add_argument("--launch-count", type=int, default=1)
    args = ap.parse_args()

    cfg = NcuConfig(
        script=Path(args.script),
        out_dir=Path(args.out),
        label=args.label,
        kernel_regex=args.kernel,
        sudo=not args.no_sudo,
        launch_skip=args.launch_skip,
        launch_count=args.launch_count,
    )
    meta = collect(cfg)
    rows = parse_csv(Path(meta["csv_path"]))
    print(f"[+] ncu rc={meta['returncode']}  metrics={len(rows)}  -> {meta['csv_path']}")
    if meta["returncode"] != 0:
        print("[!] stderr tail:")
        print(meta["stderr_tail"])

"""Phase-1 capture: per-pass IR snapshots + per-stage end IRs + listener metadata.

Drives a Triton script as a subprocess with MLIR debug printing and kernel dump
enabled. Emits one NDJSON event stream and writes IR blobs to disk so the stream
stays small.

Event kinds:
  kernel_meta   one per compiled kernel: src_hash, target, options, env_vars
  pass_before   one per MLIR pass: idx, pass_name, pass_id, op_type, op_sym, ir_sha256, ir_path
  stage_end     one per Triton stage (ttir/ttgir/llir/ptx/cubin): ext, path, copied_to
  sass_dump     SASS disassembly path
  error         capture-side problem (subprocess failure, no dumps, etc.)

This module does no graph-building. It produces raw structured events; the
linker (cag/link.py) consumes them.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator


# Header grammar emitted by mlir::PassManager::enableIRPrinting (Triton's
# pm.enable_debug()). The optional ': @sym' suffix appears for func-like ops.
#   // -----// IR Dump Before <PassName> (<pass_id>) ('<op>' operation[: @<sym>]) //----- //
_HEADER_RE = re.compile(
    r"// -----// IR Dump (Before|After) "
    r"(?P<pass_name>[^\s(]+) "
    r"\((?P<pass_id>[^)]+)\) "
    r"\('(?P<op_type>[^']+)' operation(?:: @(?P<op_sym>[^)]+))?\) "
    r"//----- //"
)

# Stages whose end-of-stage IR Triton drops into ~/.triton/dump/<hash>/<name>.<ext>
_STAGE_EXTS = ("ttir", "ttgir", "llir", "ptx", "amdgcn")
_BINARY_EXTS = ("cubin", "hsaco")
_SASS_EXT = "sass"


@dataclasses.dataclass
class CaptureConfig:
    script: Path
    out_dir: Path                  # all artifacts (events.ndjson + ir/*.mlir) go here
    triton_cache_root: Path | None = None  # default ~/.triton
    extra_env: dict[str, str] | None = None
    python: str = sys.executable
    timeout_s: int = 600

    def __post_init__(self) -> None:
        self.script = Path(self.script).resolve()
        self.out_dir = Path(self.out_dir).resolve()
        if self.triton_cache_root is None:
            self.triton_cache_root = Path.home() / ".triton"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _parse_passes(stderr_text: str) -> Iterator[dict]:
    """Yield one record per pass header in stderr, with ir_text between headers."""
    matches = list(_HEADER_RE.finditer(stderr_text))
    for i, m in enumerate(matches):
        ir_start = m.end()
        ir_end = matches[i + 1].start() if i + 1 < len(matches) else len(stderr_text)
        ir_text = stderr_text[ir_start:ir_end].strip("\n")
        yield {
            "idx": i,
            "phase": m.group(1),                # "Before" or "After"
            "pass_name": m.group("pass_name"),
            "pass_id": m.group("pass_id"),
            "op_type": m.group("op_type"),
            "op_sym": m.group("op_sym"),
            "ir_text": ir_text,
        }


def _find_dump_dirs(triton_root: Path, before_mtime: float) -> list[Path]:
    """Return dump dirs created/modified since before_mtime."""
    dump_root = triton_root / "dump"
    if not dump_root.is_dir():
        return []
    fresh: list[Path] = []
    for d in dump_root.iterdir():
        if not d.is_dir():
            continue
        try:
            if d.stat().st_mtime >= before_mtime - 1:
                fresh.append(d)
        except FileNotFoundError:
            continue
    return fresh


def _kernel_meta_from_dump(dump_dir: Path) -> dict:
    """Best-effort: find the metadata JSON Triton writes alongside the IR files."""
    for p in dump_dir.iterdir():
        if p.suffix == ".json":
            try:
                return json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def capture(cfg: CaptureConfig) -> Path:
    """Run cfg.script, capture per-pass IRs and per-stage end IRs, write events.ndjson.

    Returns path to events.ndjson.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ir_dir = cfg.out_dir / "ir"
    ir_dir.mkdir(exist_ok=True)
    stage_dir = cfg.out_dir / "stages"
    stage_dir.mkdir(exist_ok=True)
    events_path = cfg.out_dir / "events.ndjson"
    stderr_path = cfg.out_dir / "raw_stderr.log"

    env = os.environ.copy()
    env["MLIR_ENABLE_DUMP"] = "1"
    env["TRITON_ALWAYS_COMPILE"] = "1"
    env["TRITON_KERNEL_DUMP"] = "1"
    # Stable hash seed so layout-attr hashes are reproducible across runs.
    env.setdefault("PYTHONHASHSEED", "0")
    if cfg.extra_env:
        env.update(cfg.extra_env)

    t0 = (cfg.triton_cache_root / "dump").stat().st_mtime if (
        cfg.triton_cache_root / "dump").is_dir() else 0.0

    proc = subprocess.run(
        [cfg.python, str(cfg.script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=cfg.timeout_s,
    )
    stderr_path.write_text(proc.stderr)

    events: list[dict] = []
    if proc.returncode != 0:
        events.append({
            "type": "error",
            "stage": "subprocess",
            "returncode": proc.returncode,
            "tail": proc.stderr[-2000:],
        })

    # Layer A: per-pass events from stderr
    for rec in _parse_passes(proc.stderr):
        sha = _sha256_text(rec["ir_text"])
        ir_path = ir_dir / f"{rec['idx']:04d}_{rec['pass_id']}.mlir"
        ir_path.write_text(rec["ir_text"])
        events.append({
            "type": "pass_before" if rec["phase"] == "Before" else "pass_after",
            "idx": rec["idx"],
            "pass_name": rec["pass_name"],
            "pass_id": rec["pass_id"],
            "op_type": rec["op_type"],
            "op_sym": rec["op_sym"],
            "ir_sha256": sha,
            "ir_path": str(ir_path.relative_to(cfg.out_dir)),
            "ir_bytes": len(rec["ir_text"]),
        })

    # Layer B: per-stage end-of-stage IRs from the Triton dump dir
    fresh_dumps = _find_dump_dirs(cfg.triton_cache_root, t0)
    for d in fresh_dumps:
        meta = _kernel_meta_from_dump(d)
        events.append({
            "type": "kernel_meta",
            "dump_dir": str(d),
            "src_hash": meta.get("hash"),
            "triton_version": meta.get("triton_version"),
            "target": meta.get("target"),
            "num_warps": meta.get("num_warps"),
            "num_stages": meta.get("num_stages"),
            "num_ctas": meta.get("num_ctas"),
            "shared": meta.get("shared"),
            "name": meta.get("name"),
            "options": {k: v for k, v in meta.items()
                        if k not in {"hash", "target", "triton_version"}},
        })
        for p in sorted(d.iterdir()):
            ext = p.suffix.lstrip(".")
            if ext in _STAGE_EXTS or ext == _SASS_EXT or ext in _BINARY_EXTS:
                dst = stage_dir / p.name
                shutil.copy2(p, dst)
                text_sha = None
                if ext not in _BINARY_EXTS:
                    try:
                        text_sha = _sha256_text(dst.read_text())
                    except UnicodeDecodeError:
                        pass
                events.append({
                    "type": "stage_end" if ext != _SASS_EXT else "sass_dump",
                    "ext": ext,
                    "name": p.name,
                    "path": str(dst.relative_to(cfg.out_dir)),
                    "size_bytes": dst.stat().st_size,
                    "sha256": text_sha,
                    "dump_dir": str(d),
                })

    if not fresh_dumps:
        events.append({
            "type": "error",
            "stage": "dump_dir",
            "msg": "no fresh dump dirs found; check TRITON_KERNEL_DUMP and dump root",
            "triton_cache_root": str(cfg.triton_cache_root),
            "t0_mtime": t0,
        })

    with events_path.open("w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")

    return events_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="Triton Python script to run")
    ap.add_argument("--out", required=True, help="output directory for events.ndjson")
    args = ap.parse_args()
    p = capture(CaptureConfig(script=Path(args.script), out_dir=Path(args.out)))
    print(f"[+] events -> {p}")

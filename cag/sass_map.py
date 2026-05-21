"""PC -> source-line back-mapping for Triton cubins.

Pipeline:
  1. cuobjdump --dump-sass <cubin>     -> SASS instructions with PC offsets.
  2. objdump --dwarf=decodedline       -> (file, line, starting_address) records
     for each PC range. Triton ships `ptxas -lineinfo` by default; the resulting
     line-program is what DWARF tools parse.
  3. Join: for each SASS PC, find the largest PC <= it in the DWARF table, attach
     (file, line). Then look up SourceOp ID in the CAG.

Limitations:
  * NCU 2025 / CUDA 13 use "Std ELF" cubin format. nvdisasm warns about this and
    falls back to "Old format" but cuobjdump --dump-sass still works.
  * No column information from DWARF — only (file, line). Our SourceOp nodes
    have (file, line, col); we match on (file, line) and aggregate over cols.
  * ptxas reorders instructions aggressively; line info is best-effort and may
    skip lines that got hoisted. Coverage is reported per kernel.
"""
from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from pathlib import Path
from typing import Sequence


SASS_LINE_RE = re.compile(
    r"^\s*/\*(?P<pc>[0-9a-f]+)\*/\s+(?P<inst>.+?)\s*;\s*/\*\s*(?P<enc>0x[0-9a-f]+)\s*\*/"
)

# objdump --dwarf=decodedline lines look like:
#   baseline.py                                6                   0               x
# where col 1 = file, col 2 = line, col 3 = starting_address (hex with optional 0x).
DWARF_DECODED_RE = re.compile(
    r"^(?P<file>\S+\.py)\s+(?P<line>\d+|-)\s+(?P<addr>0x[0-9a-fA-F]+|\d+)\s*"
)


@dataclasses.dataclass(frozen=True)
class SassInst:
    pc: int                # byte offset into the kernel function
    text: str              # disassembled mnemonic + operands
    enc: str               # hex-encoded instruction word


@dataclasses.dataclass(frozen=True)
class LineRow:
    file: str              # basename in DWARF (we resolve to full path separately)
    line: int              # source line (1-indexed). 0 == terminal "-" record.
    addr: int              # PC where this (file, line) takes effect


@dataclasses.dataclass(frozen=True)
class PcAttribution:
    pc: int
    sass: str
    file: str | None
    line: int | None


def dump_sass(cubin: Path, cuobjdump: str = "cuobjdump") -> list[SassInst]:
    """Run cuobjdump --dump-sass and parse PC + mnemonic for each instruction."""
    proc = subprocess.run(
        [cuobjdump, "--dump-sass", str(cubin)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cuobjdump failed: {proc.stderr[:500]}")
    insts: list[SassInst] = []
    for line in proc.stdout.splitlines():
        m = SASS_LINE_RE.match(line)
        if m:
            insts.append(SassInst(
                pc=int(m.group("pc"), 16),
                text=m.group("inst").strip(),
                enc=m.group("enc"),
            ))
    return insts


def dump_line_program(cubin: Path, objdump: str = "objdump") -> list[LineRow]:
    """Parse `objdump --dwarf=decodedline` output into LineRow records.

    The decoded-line table is one row per source position with the PC at which
    it takes effect. Terminal rows have line == "-" indicating end-of-function.
    """
    proc = subprocess.run(
        [objdump, "--dwarf=decodedline", str(cubin)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"objdump failed: {proc.stderr[:500]}")
    rows: list[LineRow] = []
    for line in proc.stdout.splitlines():
        m = DWARF_DECODED_RE.match(line)
        if not m:
            continue
        line_num_str = m.group("line")
        addr_str = m.group("addr")
        addr = int(addr_str, 16) if addr_str.startswith("0x") else int(addr_str)
        line_num = 0 if line_num_str == "-" else int(line_num_str)
        rows.append(LineRow(
            file=m.group("file"),
            line=line_num,
            addr=addr,
        ))
    rows.sort(key=lambda r: r.addr)
    return rows


def resolve_file_path(rows: Sequence[LineRow], cubin: Path) -> str:
    """objdump prints just the basename (e.g. 'baseline.py'). Look in the header
    of the decodedline output for the full path. As a fallback we use the first
    longer-than-basename path we see in the cubin DWARF strings."""
    proc = subprocess.run(
        ["objdump", "--dwarf=decodedline", str(cubin)],
        capture_output=True, text=True, timeout=60,
    )
    # The header has a line like:
    #   /full/path/to/baseline.py:
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.endswith(":") and line.startswith("/") and line.endswith(".py:"):
            return line[:-1]
    # Last resort: scan the cubin for the longest .py path.
    raw = Path(cubin).read_bytes()
    candidates = re.findall(rb"/[A-Za-z0-9_/.\-]+\.py", raw)
    if candidates:
        return max(candidates, key=len).decode("utf-8", "ignore")
    return ""


def attribute_pcs(sass: list[SassInst], rows: list[LineRow]) -> list[PcAttribution]:
    """For each SASS instruction, find the line row that covers its PC.

    DWARF semantics: a line row at address A is in effect until the next row's
    address. We sort rows by addr and step a pointer.
    """
    out: list[PcAttribution] = []
    j = 0
    # Skip leading "-" rows (no line yet).
    for inst in sass:
        # Advance j as long as next row starts at or before this PC.
        while j + 1 < len(rows) and rows[j + 1].addr <= inst.pc:
            j += 1
        if j < len(rows) and rows[j].addr <= inst.pc:
            row = rows[j]
            out.append(PcAttribution(
                pc=inst.pc, sass=inst.text,
                file=row.file if row.line else None,
                line=row.line if row.line else None,
            ))
        else:
            out.append(PcAttribution(pc=inst.pc, sass=inst.text, file=None, line=None))
    return out


def coverage(attrs: list[PcAttribution]) -> dict[str, float]:
    """Fraction of SASS instructions that resolved to a (file, line)."""
    total = len(attrs)
    resolved = sum(1 for a in attrs if a.line is not None)
    return {
        "total_inst": total,
        "resolved": resolved,
        "coverage_pct": (100.0 * resolved / total) if total else 0.0,
    }


def build_map(cubin: Path) -> dict:
    """Top-level: dump SASS + DWARF, join, return a JSON-able report."""
    sass = dump_sass(cubin)
    rows = dump_line_program(cubin)
    full_path = resolve_file_path(rows, cubin)
    attrs = attribute_pcs(sass, rows)
    cov = coverage(attrs)
    return {
        "cubin": str(cubin),
        "full_path": full_path,
        "sass_inst_count": len(sass),
        "dwarf_rows": len(rows),
        "coverage": cov,
        "attributions": [
            {"pc": f"0x{a.pc:x}", "sass": a.sass, "file": a.file, "line": a.line}
            for a in attrs
        ],
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cubin")
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args()
    rep = build_map(Path(args.cubin))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"[+] wrote {args.out}  coverage={rep['coverage']['coverage_pct']:.1f}%")
    else:
        print(f"sass={rep['sass_inst_count']} dwarf_rows={rep['dwarf_rows']} "
              f"coverage={rep['coverage']['coverage_pct']:.1f}% file={rep['full_path']}")
        for a in rep["attributions"][:10]:
            print(f"  {a['pc']:>6}  L{a['line']}  {a['sass']}")
        if len(rep["attributions"]) > 10:
            print(f"  ... ({len(rep['attributions']) - 10} more)")

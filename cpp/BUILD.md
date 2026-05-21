# Building and integrating the CAG MLIR PassInstrumentation

This shared library implements `mlir::PassInstrumentation` and emits one
NDJSON event per pass/op transition. It is the C++ backbone for IRNode-level
cross-pass identity in the Causal Attribution Graph (the `IRNode ->
[transformed_to] -> IRNode` edges).

The pure-Python fallback in `cag/identity.py` approximates the same
information by structural fingerprinting on the captured IR text files. Use
the C++ path when you need authoritative identity (i.e., for paper-grade
results) and the Python path during day-to-day iteration where you don't want
to maintain a Triton-from-source build.

---

## Prerequisites

This library cannot be built against the pip Triton wheel. It needs LLVM/MLIR
development headers, and those must match Triton's pinned LLVM commit
(otherwise C++ ABI mismatch causes silent corruption at runtime).

1. Clone Triton from source: `git clone https://github.com/triton-lang/triton`.
2. Read `triton/cmake/llvm-hash.txt` (or wherever the current Triton release
   keeps it) to find the pinned LLVM commit.
3. Build LLVM/MLIR at that commit:
   ```
   git clone https://github.com/llvm/llvm-project
   cd llvm-project && git checkout <pinned-sha>
   cmake -S llvm -B build \
       -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DCMAKE_INSTALL_PREFIX=$HOME/llvm-mlir-cag
   cmake --build build -j
   cmake --install build
   ```
4. Build Triton against the same LLVM (`LLVM_SYSPATH=$HOME/llvm-mlir-cag pip
   install -e python/`). Confirm `python -c "import triton; print(triton.__file__)"`
   points at the editable install.

Total cost: 30-60 min for LLVM, 10-20 min for Triton, on a 16-core box.

## Build

```
cmake -S cpp -B cpp/build \
    -DLLVM_DIR=$HOME/llvm-mlir-cag/lib/cmake/llvm \
    -DMLIR_DIR=$HOME/llvm-mlir-cag/lib/cmake/mlir \
    -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
```

Output: `cpp/build/libCagPassInstrumentation.so` exporting a single C symbol:

```c
int cag_attach_to_pass_manager(mlir::PassManager *pm);
```

## Wiring into Triton

Triton's `python/triton/backends/nvidia/compiler.py` already has an
extension point. Two integration patterns:

### Pattern A: shim via `CUDABackend.instrumentation` (lightest touch)

Triton's `CUDABackend` has a class-level `instrumentation = None` field.
When set, Triton calls `.load_dialects(ctx)` and `.patch(stage_name, pm,
ctx)`. We write a tiny Python wrapper that, in `.patch()`, calls our C
symbol via `ctypes`:

```python
# cag/triton_hook.py
import ctypes, os
from pathlib import Path
import triton.backends.nvidia.compiler as nv

class CagInstrumentation:
    def __init__(self):
        so = Path(__file__).parent.parent / "cpp/build/libCagPassInstrumentation.so"
        self._lib = ctypes.CDLL(str(so))
        self._lib.cag_attach_to_pass_manager.argtypes = [ctypes.c_void_p]
        self._lib.cag_attach_to_pass_manager.restype = ctypes.c_int

    def load_dialects(self, ctx):
        # Nothing dialect-specific; we observe ops generically.
        pass

    def patch(self, stage_name, pm, ctx):
        # pm here is a Triton ir.pass_manager; expose its underlying C++ pointer.
        # Triton's pm exposes .get_capsule() in recent versions; fall back to
        # ctypes_handle if available.
        addr = getattr(pm, "get_capsule", lambda: None)()
        if addr is not None:
            self._lib.cag_attach_to_pass_manager(addr)

nv.CUDABackend.instrumentation = CagInstrumentation()
```

Then run any Triton script with the event sink configured:

```
TRITON_CAG_EVENTS_PATH=/tmp/cag_cpp_events.ndjson \
TRITON_INSTRUMENTATION_MODE=cag \
python3 -c "import cag.triton_hook; import my_kernel"
```

> Note: `pm.get_capsule()` is not yet exposed by every Triton release. If
> missing, we have to patch Triton: see Pattern B below.

### Pattern B: patch `triton/backends/nvidia/compiler.py` directly

Add three lines after each `pm = ir.pass_manager(...)` construction:

```python
from cag.triton_hook import attach
attach(pm)
```

where `attach()` extracts the underlying `mlir::PassManager*` from the
pybind capsule and calls `cag_attach_to_pass_manager`. This requires
maintaining a Triton patch but is the most robust path. We ship the patch as
`cpp/triton.patch` (TODO).

## Output schema

Events emitted to the sink (one JSON object per line):

```json
{"type":"pass_before_cpp","pass_idx":0,"pass_name":"Inliner","pass_arg":"inline","root_op":"builtin.module","root_uid":1,"newly_tagged":42}
{"type":"op_before_cpp","pass_idx":0,"uid":2,"parent_uid":1,"op_name":"tt.func","loc":"loc(\"k.py\":6:0)","num_operands":0,"num_results":0}
{"type":"op_after_cpp","pass_idx":0,"uid":2,"parent_uid":1,"op_name":"tt.func","loc":"loc(\"k.py\":6:0)"}
{"type":"pass_after_cpp","pass_idx":0,"pass_name":"Inliner","newly_tagged":0}
```

The linker (`cag/link.py`) accepts these events in addition to the
stderr-derived `pass_before` records. When both are present the C++ records
take precedence because they carry authoritative UIDs.

## Limitations

* MLIR's `Operation::clone` propagates discardable attributes (which
  `cag.uid` is), so identity survives most passes. Some passes that build
  ops from scratch via `OpBuilder` will *not* copy the UID; for those ops
  we'll assign a new UID on the next sweep, which is the right behaviour:
  it's genuinely a new op.

* The instrumentation attaches to one PassManager at a time. Triton
  constructs several PassManagers per compile (`make_ttir`, `make_ttgir`,
  `make_llir`, `make_ptx`). Pattern A above attaches at each stage by
  hooking `CUDABackend.instrumentation.patch`. The UID counter is
  process-wide and monotonic, so UIDs don't collide across stages.

* Pass parallelism: MLIR may run nested passes in parallel
  (`MLIR_DISABLE_THREADING=0`). The sink is mutex-guarded. To disable
  threading for deterministic output set `MLIR_DISABLE_THREADING=1` (Triton
  does this by default for its main pipelines).

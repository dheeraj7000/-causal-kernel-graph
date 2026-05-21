# Regression Corpus Catalog (Gold Labels)

This document tracks the seeded regressions in the `/corpus` directory, identifying the "True Cause" for validation of the Attribution Graph.

| Kernel | Variant | Seeded Regression (True Cause) | Expected Symptom | Target Knob |
| :--- | :--- | :--- | :--- | :--- |
| **Vector Add** | `regression_block_size.py` | `BLOCK_SIZE=32` (Too small) | High kernel launch overhead, poor coalescing. | `BLOCK_SIZE` |
| **Matmul** | `regression_num_stages.py` | `num_stages=1` (Disabled pipelining) | Increased memory latency stalls. | `num_stages` |
| **Softmax** | `regression_num_warps.py` | `num_warps=1` (Too few warps) | Low occupancy, poor latency hiding. | `num_warps` |

## Future Additions
- **Matmul (Bank Conflicts):** Perturbing the `#shared` layout swizzling parameters.
- **LayerNorm (Reduction):** Low vectorization factor in reduction loops.
- **FlashAttention:** Imbalanced tiling causing occupancy loss.

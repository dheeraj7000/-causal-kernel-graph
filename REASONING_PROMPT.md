I have conducted a sensitivity study on a Triton GPU kernel. I perturbed the 'BLOCK_SIZE' knob and observed the following hardware symptoms:

| BLOCK_SIZE | Long Scoreboard Stalls (%) | Active Warps Occupancy (%) |
| :--- | :--- | :--- |
| 32 | 35.50 | 10.00 |
| 128 | 8.88 | 10.62 |
| 512 | 5.00 | 42.50 |
| 1024 | 5.00 | 85.00 |
| 2048 | 5.00 | 100.00 |

**Analysis Task:**
1. Identify the causal relationship between BLOCK_SIZE and memory latency (stalls).
2. Determine the optimal BLOCK_SIZE for this kernel based on the Pareto front of throughput vs. occupancy.
3. Explain why very small BLOCK_SIZE values (e.g., 32) lead to such high stall percentages.
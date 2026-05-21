import json
import glob
import os
import re

def generate_reasoning_prompt(study_dir):
    """Aggregates graph data into a concise reasoning prompt for an LLM."""
    prompt = "I have conducted a sensitivity study on a Triton GPU kernel. "
    prompt += "I perturbed the 'BLOCK_SIZE' knob and observed the following hardware symptoms:\n\n"
    
    graphs = sorted(glob.glob(os.path.join(study_dir, "graph_*.jsonl")), key=lambda x: int(re.search(r"_(\d+)\.", x).group(1)))
    
    table = "| BLOCK_SIZE | Long Scoreboard Stalls (%) | Active Warps Occupancy (%) |\n"
    table += "| :--- | :--- | :--- |\n"
    
    for graph_path in graphs:
        val = re.search(r"_(\d+)\.", graph_path).group(1)
        stalls = "N/A"
        occupancy = "N/A"
        
        with open(graph_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                if "node" in item:
                    data = item["node"]
                    if data["type"] == "HardwareEvent":
                        if data["metric"] == "smsp__warp_stall_long_scoreboard_pct":
                            stalls = data["value"]
                        if data["metric"] == "sm__occupancy_active_warps_pct":
                            occupancy = data["value"]
        
        table += f"| {val} | {stalls} | {occupancy} |\n"
    
    prompt += table
    
    # New section for Layout Identity
    prompt += "\n**Layout Identity Discovery:**\n"
    shared_layouts = set()
    with open(graphs[-1], 'r') as f: # Look at the most recent/regressed graph
        for line in f:
            item = json.loads(line)
            if "node" in item:
                data = item["node"]
                if data["type"] == "LayoutEncoding" and data["label"] == "SharedLayout":
                    shared_layouts.add(data["attrs"])
    
    if shared_layouts:
        prompt += "The following Shared Memory Layout was identified in the regressed kernel:\n"
        for layout in shared_layouts:
            prompt += f"- `{layout}`\n"
    
    prompt += "\n**Analysis Task:**\n"
    prompt += "1. Identify the causal relationship between BLOCK_SIZE and memory latency (stalls).\n"
    prompt += "2. Determine the optimal BLOCK_SIZE for this kernel based on the Pareto front of throughput vs. occupancy.\n"
    prompt += "3. Explain why very small BLOCK_SIZE values (e.g., 32) lead to such high stall percentages."
    
    return prompt

if __name__ == "__main__":
    prompt = generate_reasoning_prompt("/root/superlensai/sensitivity_study")
    with open("/root/superlensai/REASONING_PROMPT.md", 'w') as f:
        f.write(prompt)
    print("[+] Reasoning prompt generated at /root/superlensai/REASONING_PROMPT.md")

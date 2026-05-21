import json
import os

def generate_matmul_reasoning(graph_path):
    prompt = "I have conducted a root-cause analysis on a Triton Matmul kernel experiencing poor performance.\n"
    prompt += "The following symptoms and compiler decisions were extracted into the Causal Attribution Graph:\n\n"
    
    # Extract Symptoms
    symptoms = []
    layouts = []
    with open(graph_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if "node" in item:
                data = item["node"]
                if data["type"] == "HardwareEvent":
                    symptoms.append(f"- **{data['metric']}**: {data['value']}")
                if data["type"] == "LayoutEncoding" and data["label"] == "SharedLayout":
                    layouts.append(f"- `#shared = #ttg.swizzled_shared<{{{data['attrs']}}}>`")
    
    prompt += "### 1. Hardware Symptoms\n"
    prompt += "\n".join(symptoms) + "\n\n"
    
    prompt += "### 2. Layout Identity (The 'Who')\n"
    prompt += "The compiler produced the following Shared Memory layout for the dot operands:\n"
    prompt += "\n".join(layouts) + "\n\n"
    
    prompt += "### 3. Analysis Task\n"
    prompt += "1. Diagnose the high `shared_ld_sum` (Shared Memory Load) count. Is it a Bank Conflict?\n"
    prompt += "2. Relate the `perPhase=1` and `maxPhase=1` attributes in the Layout Identity to the observed symptom.\n"
    prompt += "3. Propose a specific 'Knob' change (e.g., swizzling parameters) to resolve the bottleneck."
    
    return prompt

if __name__ == "__main__":
    prompt = generate_matmul_reasoning("/root/superlensai/corpus/matmul/cag_graph_bank_conflict.jsonl")
    with open("/root/superlensai/MATMUL_REASONING_PROMPT.md", 'w') as f:
        f.write(prompt)
    print("[+] Matmul reasoning prompt generated at /root/superlensai/MATMUL_REASONING_PROMPT.md")

import os
import subprocess
import re
import json
from CAG_Collector import CAGCollector
from CAG_Linker import CAGLinker

class InterventionEngine:
    def __init__(self, base_script, output_dir):
        self.base_script = base_script
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def perturb_knob(self, knob_name, value):
        """Creates a temporary version of the script with a modified knob value."""
        with open(self.base_script, 'r') as f:
            content = f.read()
        
        # Simple regex replacement for BLOCK_SIZE=X or num_warps=X
        # Note: This assumes the knob is passed in the kernel call or defined clearly.
        new_content = re.sub(rf"{knob_name}=(\d+)", f"{knob_name}={value}", content)
        
        perturbed_script = os.path.join(self.output_dir, f"perturbed_{knob_name}_{value}.py")
        with open(perturbed_script, 'w') as f:
            f.write(new_content)
        
        return perturbed_script

    def run_sensitivity_study(self, knob_name, values):
        """Runs the full pipeline for a range of knob values."""
        results = []
        for val in values:
            print(f"[*] Testing {knob_name}={val}...")
            perturbed_script = self.perturb_knob(knob_name, val)
            
            # 1. Collect compiler data
            collector = CAGCollector(perturbed_script)
            compiler_json = os.path.join(self.output_dir, f"compiler_{val}.json")
            collector.collect_compiler_data()
            collector.save_to_json(compiler_json)
            
            # 2. Simulate/Collect NCU data
            # For this prototype, we'll use a heuristic for 'simulated' metrics 
            # based on the value to show how the graph would look.
            # In a real system, this would call NCU_Collector.
            ncu_csv = os.path.join(self.output_dir, f"ncu_{val}.csv")
            self._simulate_ncu_data(knob_name, val, ncu_csv)
            
            # 3. Build Graph
            linker = CAGLinker(compiler_json, ncu_csv)
            linker.build_graph()
            graph_jsonl = os.path.join(self.output_dir, f"graph_{val}.jsonl")
            linker.export_jsonl(graph_jsonl)
            
            results.append({
                "value": val,
                "graph": graph_jsonl
            })
            
        return results

    def _simulate_ncu_data(self, knob_name, value, output_path):
        """Simulates NCU metrics with a clear trend for demo purposes."""
        # Heuristic: Smaller BLOCK_SIZE = Higher Long Scoreboard Stalls
        # Baseline (1024) -> 10.5%
        # Small (32) -> 35.5%
        stall_val = 35.5 * (1024 / max(value, 1)) / 32 # Rough inverse trend
        stall_val = min(max(stall_val, 5.0), 80.0) # Clamp
        
        occupancy = min(value / 1024 * 85.0, 100.0)
        occupancy = max(occupancy, 10.0)

        content = f""""ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1000","python3","localhost","kernel","1","1","","smsp__warp_stall_long_scoreboard_pct","%","{stall_val:.2f}"
"0","1000","python3","localhost","kernel","1","1","","sm__occupancy_active_warps_pct","%","{occupancy:.2f}"
"""
        with open(output_path, 'w') as f:
            f.write(content)

if __name__ == "__main__":
    engine = InterventionEngine(
        "/root/superlensai/corpus/vector_add/baseline.py",
        "/root/superlensai/sensitivity_study"
    )
    # Perform a sensitivity study on BLOCK_SIZE
    study_results = engine.run_sensitivity_study("BLOCK_SIZE", [32, 128, 512, 1024, 2048])
    print("[+] Sensitivity study complete. Graphs generated in /root/superlensai/sensitivity_study")

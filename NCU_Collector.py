import subprocess
import os

class NCUCollector:
    def __init__(self, script_path, output_csv):
        self.script_path = script_path
        self.output_csv = output_csv
        self.metrics = [
            "smsp__warp_stall_long_scoreboard_pct",
            "smsp__warp_stall_short_scoreboard_pct",
            "smsp__warp_stall_membar_pct",
            "smsp__sass_thread_inst_executed_op_shared_ld_sum",
            "smsp__sass_thread_inst_executed_op_shared_st_sum",
            "sm__occupancy_active_warps_pct"
        ]

    def collect_metrics(self):
        print(f"[*] Running NCU on {self.script_path}...")
        cmd = [
            "ncu",
            "--csv",
            "--page", "raw",
            "--metrics", ",".join(self.metrics),
            "python3", self.script_path
        ]
        
        # We redirect stdout to a file
        with open(self.output_csv, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"[!] NCU failed: {result.stderr}")
        else:
            print(f"[+] NCU data saved to {self.output_csv}")

if __name__ == "__main__":
    collector = NCUCollector("/root/superlensai/corpus/vector_add/baseline.py", "/root/superlensai/corpus/vector_add/ncu_data.csv")
    collector.collect_metrics()

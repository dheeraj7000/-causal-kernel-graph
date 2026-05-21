import os
import subprocess
import re
import json

class CAGCollector:
    def __init__(self, script_path):
        self.script_path = script_path
        self.ir_dumps = []

    def collect_compiler_data(self):
        """Runs the script with IR dumping enabled and parses the output."""
        env = os.environ.copy()
        env["TRITON_ALWAYS_COMPILE"] = "1"
        env["MLIR_ENABLE_DUMP"] = "1"
        
        print(f"[*] Running {self.script_path} with MLIR dumping...")
        result = subprocess.run(
            ["python3", self.script_path],
            env=env,
            capture_output=True,
            text=True
        )
        
        # Parse IR dumps from stderr
        raw_output = result.stderr
        passes = re.split(r"// -----// IR Dump (Before|After) (.+?) \((.+?)\) \('(.+?)' operation\) //----- //", raw_output)
        
        # re.split with multiple groups can be tricky. Let's use a more robust regex.
        pass_pattern = r"// -----// IR Dump (Before|After) (.+?) \((.+?)\) \('(.+?)' operation\) //----- //"
        matches = list(re.finditer(pass_pattern, raw_output))
        
        for i in range(len(matches)):
            start = matches[i].end()
            end = matches[i+1].start() if i+1 < len(matches) else len(raw_output)
            
            self.ir_dumps.append({
                "type": matches[i].group(1),
                "pass_desc": matches[i].group(2),
                "pass_id": matches[i].group(3),
                "op_type": matches[i].group(4),
                "ir": raw_output[start:end].strip()
            })
            
        print(f"[+] Captured {len(self.ir_dumps)} IR snapshots.")
        return self.ir_dumps

    def save_to_json(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.ir_dumps, f, indent=2)

if __name__ == "__main__":
    collector = CAGCollector("/root/superlensai/corpus/vector_add/baseline.py")
    collector.collect_compiler_data()
    collector.save_to_json("/root/superlensai/corpus/vector_add/compiler_data.json")

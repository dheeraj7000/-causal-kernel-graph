import json
import re

class CAGLinker:
    def __init__(self, compiler_json, ncu_csv):
        with open(compiler_json, 'r') as f:
            self.compiler_data = json.load(f)
        self.ncu_csv = ncu_csv
        self.nodes = []
        self.edges = []

    def _extract_knobs(self, ir_text):
        """Accurate knob extraction from Triton GPU IR."""
        knobs = {}
        # num_warps is in the module attributes
        num_warps_match = re.search(r"\"ttg.num-warps\" = (\d+) : i32", ir_text)
        if num_warps_match:
            knobs["num_warps"] = int(num_warps_match.group(1))
        
        # BLOCK_SIZE is the 'end' of tt.make_range
        block_size_match = re.search(r"tt.make_range \{end = (\d+) : i32, start = 0 : i32\}", ir_text)
        if block_size_match:
            knobs["BLOCK_SIZE"] = int(block_size_match.group(1))
            
        return knobs

    def _extract_layouts(self, ir_text):
        """Extracts LayoutEncoding nodes from the IR."""
        layouts = []
        # Match #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
        shared_matches = re.finditer(r"#shared = #ttg\.swizzled_shared<{(.+?)}>", ir_text)
        for match in shared_matches:
            attrs = match.group(1)
            layouts.append({
                "type": "LayoutEncoding",
                "label": "SharedLayout",
                "attributes": attrs
            })
            
        # Match #blocked = #ttg.blocked<{...}>
        blocked_matches = re.finditer(r"#blocked\d* = #ttg\.blocked<{(.+?)}>", ir_text)
        for match in blocked_matches:
            attrs = match.group(1)
            layouts.append({
                "type": "LayoutEncoding",
                "label": "BlockedLayout",
                "attributes": attrs
            })
        return layouts

    def build_graph(self):
        # 1. Add Pass Nodes and IR Nodes
        for idx, snapshot in enumerate(self.compiler_data):
            pass_node_id = f"pass_{idx}"
            self.nodes.append({
                "id": pass_node_id,
                "type": "Pass",
                "label": snapshot["pass_id"],
                "desc": snapshot["pass_desc"]
            })
            
            # Extract Knobs
            knobs = self._extract_knobs(snapshot["ir"])
            for knob_name, knob_val in knobs.items():
                knob_id = f"knob_{knob_name}_{knob_val}"
                if not any(n["id"] == knob_id for n in self.nodes):
                    self.nodes.append({
                        "id": knob_id,
                        "type": "Parameter",
                        "name": knob_name,
                        "value": knob_val
                    })
                self.edges.append({
                    "source": knob_id,
                    "target": pass_node_id,
                    "type": "governed_by"
                })

            # Extract Layouts
            layouts = self._extract_layouts(snapshot["ir"])
            for layout_data in layouts:
                layout_id = f"layout_{hash(layout_data['attributes'])}"
                if not any(n["id"] == layout_id for n in self.nodes):
                    self.nodes.append({
                        "id": layout_id,
                        "type": "LayoutEncoding",
                        "label": layout_data["label"],
                        "attrs": layout_data["attributes"]
                    })
                self.edges.append({
                    "source": pass_node_id,
                    "target": layout_id,
                    "type": "produced"
                })

        # 2. Add Hardware Symptom Nodes from CSV
        with open(self.ncu_csv, 'r') as f:
            lines = f.readlines()[1:] # Skip header
            for i, line in enumerate(lines):
                parts = line.strip().split('","')
                metric_name = parts[8].strip('"')
                metric_val = parts[10].strip('"')
                
                symptom_id = f"symptom_{i}"
                self.nodes.append({
                    "id": symptom_id,
                    "type": "HardwareEvent",
                    "metric": metric_name,
                    "value": metric_val
                })
                
                # Link last pass to symptoms (as a simplification for the prototype)
                self.edges.append({
                    "source": f"pass_{len(self.compiler_data)-1}",
                    "target": symptom_id,
                    "type": "manifested_as"
                })

    def export_jsonl(self, output_path):
        with open(output_path, 'w') as f:
            for node in self.nodes:
                f.write(json.dumps({"node": node}) + "\n")
            for edge in self.edges:
                f.write(json.dumps({"edge": edge}) + "\n")

if __name__ == "__main__":
    linker = CAGLinker(
        "/root/superlensai/corpus/vector_add/compiler_data.json",
        "/root/superlensai/corpus/vector_add/ncu_data_baseline.csv"
    )
    linker.build_graph()
    linker.export_jsonl("/root/superlensai/corpus/vector_add/cag_graph.jsonl")
    print("[+] Graph construction complete.")

import os
import json
from CAG_Collector import CAGCollector
from CAG_Linker import CAGLinker

# 1. Collect compiler data for the regression kernel
collector = CAGCollector("/root/superlensai/corpus/vector_add/regression_block_size.py")
collector.collect_compiler_data()
collector.save_to_json("/root/superlensai/corpus/vector_add/compiler_data_regression.json")

# 2. Link with the synthesized regression NCU data
linker = CAGLinker(
    "/root/superlensai/corpus/vector_add/compiler_data_regression.json",
    "/root/superlensai/corpus/vector_add/ncu_data_regression.csv"
)
linker.build_graph()
linker.export_jsonl("/root/superlensai/corpus/vector_add/cag_graph_regression.jsonl")

print("[+] Regression graph construction complete.")

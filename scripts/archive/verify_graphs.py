import os, json, glob

ROOT = "data/graphs_raw"

files = sorted(glob.glob(f"{ROOT}/*.jsonl"))
print("Total graph files:", len(files))

total_graphs = 0
for f in files:
    for line in open(f):
        total_graphs += 1

print("Total graph items:", total_graphs)

# check minimum structure
bad = 0
for f in files:
    for line in open(f):
        j = json.loads(line)
        if not all(k in j for k in ["ast", "cfg", "dfg"]):
            bad += 1

print("Graphs missing keys:", bad)

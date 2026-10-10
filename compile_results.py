"""Merge the per-experiment JSON files into results_all.json (the file behind every number in the paper)."""
import json
parts = ["sbox", "diff", "kav", "coll", "lin", "att", "dist", "func", "nist", "extra"]
allr = {p: json.load(open(f"res_{p}.json")) for p in parts}
allr["v1_experiments"] = json.load(open("results_v1.json"))
allr["key_derivation_check"] = json.load(open("results_kd.json"))
allr["dh_cost"] = json.load(open("dhtime.json"))
json.dump(allr, open("results_all.json", "w"), indent=1)
print("wrote results_all.json")

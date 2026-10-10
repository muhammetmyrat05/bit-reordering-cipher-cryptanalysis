import sys, json, time
import exp_v2 as e
name = sys.argv[1]
fn = {"lin": e.linear_bias, "att": e.attacks_on_v2, "dist": e.distinguishers_v2,
      "func": e.functional_checks, "nist": e.nist_experiment,
      "sbox": e.sbox_properties, "diff": e.diffusion, "kav": e.key_avalanche,
      "coll": e.differential_collisions}[name]
t0 = time.time()
res = fn()
json.dump(res, open(f"res_{name}.json", "w"), indent=2)
print(name, "done in", round(time.time() - t0, 1), "s", flush=True)

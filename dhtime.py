import time, json, secrets, sys
from cryptography.hazmat.primitives.asymmetric import dh
out = {}
# (1) 2048-bit modular exponentiation cost (what each exchange costs per side)
p = secrets.randbits(2048) | (1 << 2047) | 1
g = 2
reps = 20
for label, bits in (("exp256", 256), ("exp2048", 2048)):
    t0 = time.perf_counter()
    for _ in range(reps):
        a = secrets.randbits(bits) | 1
        pow(g, a, p)
    out[label + "_ms"] = (time.perf_counter() - t0) / reps * 1000
json.dump(out, open("dhtime_modexp.json", "w"))
print("modexp done", out, flush=True)
# (2) the call used in [1]: generate_parameters(generator=2, key_size=2048)
t0 = time.perf_counter()
try:
    params = dh.generate_parameters(generator=2, key_size=2048)
    out["generate_parameters_s"] = time.perf_counter() - t0
except Exception as e:
    out["generate_parameters_error"] = repr(e)
json.dump(out, open("dhtime.json", "w"))
print("done", out, flush=True)

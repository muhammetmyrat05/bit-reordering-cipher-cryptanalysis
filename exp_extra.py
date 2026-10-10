import json, math
import numpy as np
from ciphers import v1_keys, v1_source_index, v1_encrypt
rng = np.random.default_rng(777)
# HD preservation for ARBITRARY message pairs under one key (consequence of the avalanche proposition)
n, ok, T = 128, 0, 2000
for _ in range(T):
    R, K, X = v1_keys(n, rng); src = v1_source_index(R, K)
    P = rng.integers(0, 2, (2, n), dtype=np.uint8)
    C = v1_encrypt(P, src, X)
    ok += int(np.sum(P[0] ^ P[1]) == np.sum(C[0] ^ C[1]))
out = {"HD_preserved_arbitrary_pairs": f"{ok}/{T}"}
for n in (64, 128, 256, 512, 1024):
    l2f = math.lgamma(n + 1) / math.log(2)
    out[f"n={n}"] = {"log2_n_factorial": l2f, "log2_effective_keyspace_nfact_2n": l2f + n,
                     "log2_claimed_nfact_squared_2n": 2 * l2f + n}
json.dump(out, open("res_extra.json", "w"), indent=2)
print(json.dumps(out, indent=1))

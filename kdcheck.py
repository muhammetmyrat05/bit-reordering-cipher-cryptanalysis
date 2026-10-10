"""
kdcheck.py -- statistical check of the key-derivation step.

The original paper derives (R, K, X) from HKDF-SHA256 output but does not
say how derived bytes become permutations. We instantiate the most
direct reading: unbiased Fisher-Yates driven by the HKDF byte stream for R
and K, then raw bits for X, with the same salt=None / info=b'crypto_keys'
parameters as in the paper. The shared secret is replaced by 32 uniformly
random bytes (a DH shared secret is, for this purpose, a high-entropy
string). We test: uniformity of R and K position/value matrices,
balance and pairwise independence of X bits, and independence between the
three components.
"""
import json
import math

import numpy as np
from scipy.stats import chi2

from ciphers import hkdf_extract, ByteStream, fisher_yates, bytes_to_bits

rng = np.random.default_rng(424242)
n = 32
S = 40000


def derive(secret):
    prk = hkdf_extract(b"", secret)                 # salt=None
    st = ByteStream(prk, b"crypto_keys")
    R = fisher_yates(st, n)
    K = fisher_yates(st, n)
    X = bytes_to_bits(st.take(n // 8))
    return R, K, X


Rs = np.empty((S, n), dtype=np.int64)
Ks = np.empty((S, n), dtype=np.int64)
Xs = np.empty((S, n), dtype=np.uint8)
for s in range(S):
    Rs[s], Ks[s], Xs[s] = derive(rng.bytes(32))


def uniformity_p(P):
    counts = np.zeros((n, n))
    for i in range(n):
        counts[i] = np.bincount(P[:, i], minlength=n)
    e = S / n
    stat = np.sum((counts - e) ** 2 / e)
    return float(chi2.sf(stat, (n - 1) ** 2))


def contingency_p(a, b, ka, kb):
    t = np.zeros((ka, kb))
    np.add.at(t, (a, b), 1)
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    stat = np.sum((t - e) ** 2 / e)
    return float(chi2.sf(stat, (ka - 1) * (kb - 1)))


res = {"sessions": S, "n": n}
res["R_position_value_uniformity_p"] = uniformity_p(Rs)
res["K_position_value_uniformity_p"] = uniformity_p(Ks)

# X: bit balance (min over bits, Bonferroni) and max pairwise correlation
ones = Xs.mean(axis=0)
z = (ones - 0.5) / (0.5 / math.sqrt(S))
res["X_max_abs_z_bit_balance"] = float(np.max(np.abs(z)))
corr = np.corrcoef(Xs.T.astype(float))
np.fill_diagonal(corr, 0)
res["X_max_abs_pairwise_corr"] = float(np.max(np.abs(corr)))
res["X_corr_expected_noise_4.5sigma"] = 4.5 / math.sqrt(S)

# independence between components (first position of each + a pooled statistic)
res["R0_vs_K0_independence_p"] = contingency_p(Rs[:, 0], Ks[:, 0], n, n)
res["R0_vs_X0_independence_p"] = contingency_p(Rs[:, 0], Xs[:, 0].astype(int), n, 2)
res["K0_vs_X0_independence_p"] = contingency_p(Ks[:, 0], Xs[:, 0].astype(int), n, 2)
# pooled: p-values for all (R[i], K[i]) pairs at each position
ps = [contingency_p(Rs[:, i], Ks[:, i], n, n) for i in range(n)]
res["R_i_vs_K_i_min_p_over_positions"] = float(min(ps))
res["R_i_vs_K_i_positions_below_0.01"] = int(sum(p < 0.01 for p in ps))
res["expected_positions_below_0.01"] = n * 0.01
# composite permutation T = K o R-structure is uniform? check via fixed points of K^-1 R
res["note"] = "no structure expected: HKDF-SHA256 output is a PRF; this checks the instantiation, not HKDF itself"
json.dump(res, open("results_kd.json", "w"), indent=2)
print(json.dumps(res, indent=2))

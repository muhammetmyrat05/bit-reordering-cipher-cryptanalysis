import json
import math
import time

import numpy as np

from ciphers import v1_keys, v1_source_index, v1_encrypt

rng = np.random.default_rng(20261009)
results = {}


class Oracle:
    """Fixed-key encryption oracle with a query counter (rows = queries)."""

    def __init__(self, n, rng):
        self.n = n
        R, K, X = v1_keys(n, rng)
        self.src = v1_source_index(R, K)
        self.X = X
        self.queries = 0

    def __call__(self, P):
        P = np.atleast_2d(P)
        self.queries += P.shape[0]
        return v1_encrypt(P, self.src, self.X)


# ----------------------------------------------------------------------
# E1: deterministic avalanche -- implementation validation of the proof
# ----------------------------------------------------------------------
def e1(trials=2000, n=128):
    bad = 0
    for _ in range(trials):
        o = Oracle(n, rng)
        P = rng.integers(0, 2, n, dtype=np.uint8)
        i = int(rng.integers(n))
        P2 = P.copy(); P2[i] ^= 1
        C = o(np.stack([P, P2]))
        if int(np.sum(C[0] ^ C[1])) != 1:
            bad += 1
    return {"trials": trials, "n": n, "violations_of_HD_equals_1": bad}


# ----------------------------------------------------------------------
# E2: chosen-plaintext attacks
# ----------------------------------------------------------------------
def naive_attack(o):
    n = o.n
    Q = np.zeros((n + 1, n), dtype=np.uint8)
    Q[np.arange(1, n + 1), np.arange(n)] = 1
    C = o(Q)
    X = C[0]
    pos = np.argmax(C[1:] ^ X, axis=1)            # input i -> output pos[i]
    inv = np.empty(n, dtype=np.int64); inv[pos] = np.arange(n)
    return inv, X


def log_attack(o):
    n = o.n
    m = math.ceil(math.log2(n))
    idx = np.arange(n)
    Q = np.zeros((m + 1, n), dtype=np.uint8)
    for k in range(m):
        Q[k + 1] = (idx >> k) & 1
    C = o(Q)
    X = C[0]
    D = C[1:] ^ X                                  # D[k][j] = bit k of inv[j]
    inv = np.zeros(n, dtype=np.int64)
    for k in range(m):
        inv |= D[k].astype(np.int64) << k
    return inv, X


def decrypt_with(inv, X, C):
    un = C ^ X
    P = np.zeros_like(un)
    P[inv] = un
    return P


def lower_bound_queries(n):
    # key space of the effective transform: n! * 2^n ; each query returns n bits
    return math.ceil((math.lgamma(n + 1) / math.log(2) + n) / n)


def e2(sizes=(64, 128, 256, 512, 1024), trials=100):
    rows = []
    for n in sizes:
        row = {"n": n, "theory_naive_q": n + 1, "theory_log_q": math.ceil(math.log2(n)) + 1,
               "info_theoretic_lower_bound_q": lower_bound_queries(n)}
        for name, fn in (("naive", naive_attack), ("log", log_attack)):
            ok, qs, ts = 0, [], []
            for _ in range(trials):
                o = Oracle(n, rng)
                t0 = time.perf_counter()
                inv, X = fn(o)
                dt = time.perf_counter() - t0
                held = rng.integers(0, 2, n, dtype=np.uint8)
                ct = v1_encrypt(held[None, :], o.src, o.X)[0]       # not counted as a query
                ok += int(np.array_equal(decrypt_with(inv, X, ct), held))
                qs.append(o.queries); ts.append(dt * 1000)
            row[name] = {"queries_measured": int(np.max(qs)), "mean_time_ms": float(np.mean(ts)),
                         "held_out_decryptions_correct": ok, "trials": trials}
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# E3: known-plaintext (random pairs) -- how many pairs identify T and X?
# ----------------------------------------------------------------------
def kp_unique_probability(n, q, trials):
    """P[all n columns of the (q-1) x n difference matrix are distinct]."""
    weights = (1 << np.arange(q - 1, dtype=np.int64)) if q - 1 < 62 else None
    hits = 0
    for _ in range(trials):
        P = rng.integers(0, 2, (q, n), dtype=np.uint8)
        D = (P[1:] ^ P[0]).astype(np.int64)
        keys = weights @ D
        hits += int(len(np.unique(keys)) == n)
    return hits / trials


def kp_attack(P, C, held_ct):
    q, n = P.shape
    w = 1 << np.arange(q - 1, dtype=np.int64)
    kin = w @ (P[1:] ^ P[0]).astype(np.int64)
    kout = w @ (C[1:] ^ C[0]).astype(np.int64)
    if len(np.unique(kin)) != n:
        return None
    where_out = {int(k): j for j, k in enumerate(kout)}
    inv = np.empty(n, dtype=np.int64)               # output j <- input inv[j]
    for i, k in enumerate(kin):
        inv[where_out[int(k)]] = i
    X = C[0] ^ P[0][inv]
    return inv, X


def e3(sizes=(64, 128, 256, 512), trials=1500):
    rows = []
    for n in sizes:
        curve, q50, q99 = {}, None, None
        for q in range(2, 64):
            p = kp_unique_probability(n, q, trials)
            curve[q] = p
            if q50 is None and p >= 0.5: q50 = q
            if q99 is None and p >= 0.99: q99 = q
            if p >= 0.999: break
        # end-to-end check at q99
        ok = 0
        E2E = 200
        for _ in range(E2E):
            o = Oracle(n, rng)
            P = rng.integers(0, 2, (q99, n), dtype=np.uint8)
            C = o(P)
            held = rng.integers(0, 2, n, dtype=np.uint8)
            ct = v1_encrypt(held[None, :], o.src, o.X)[0]
            out = kp_attack(P, C, ct)
            if out is not None:
                ok += int(np.array_equal(decrypt_with(out[0], out[1], ct), held))
        # first-order approximation: P ~ exp(-n(n-1)/2 / 2^(q-1))
        approx_q99 = math.ceil(1 + math.log2(n * (n - 1) / 2 / -math.log(0.99)))
        rows.append({"n": n, "pairs_for_50pct": q50, "pairs_for_99pct": q99,
                     "approx_formula_pairs_for_99pct": approx_q99,
                     "end_to_end_success_at_q99": f"{ok}/{E2E}"})
    return rows


# ----------------------------------------------------------------------
# E4: distinguishers against v1
# ----------------------------------------------------------------------
def e4(trials=20000, n=128):
    # (a) PRP-type distinguisher: two queries x, x xor e_i; answer "cipher" iff HD == 1
    wins = 0
    for _ in range(trials):
        o = Oracle(n, rng)
        x = rng.integers(0, 2, n, dtype=np.uint8)
        i = int(rng.integers(n)); x2 = x.copy(); x2[i] ^= 1
        C = o(np.stack([x, x2]))
        wins += int(np.sum(C[0] ^ C[1]) == 1)
    # (b) IND-CPA: one oracle query. m0=0^n, m1=1^n; guess b by comparing the
    # challenge ciphertext with Enc(0^n) obtained from the oracle.
    correct = 0
    for _ in range(trials):
        o = Oracle(n, rng)
        b = int(rng.integers(2))
        chall = o(np.full(n, b, dtype=np.uint8))[0]            # challenger encrypts m_b
        ref = o(np.zeros(n, dtype=np.uint8))[0]                 # adversary's 1 query
        guess = 0 if np.array_equal(chall, ref) else 1
        correct += int(guess == b)
    return {"trials": trials, "n": n,
            "prp_distinguisher_P_says_cipher_given_v1": wins / trials,
            "prp_random_permutation_baseline": n / (2 ** n - 1),
            "prp_advantage_v1": wins / trials - n / (2 ** n - 1),
            "ind_cpa_one_query_success_rate": correct / trials,
            "ind_cpa_advantage": 2 * correct / trials - 1}


if __name__ == "__main__":
    results["E1_avalanche_validation"] = e1()
    print("E1", results["E1_avalanche_validation"], flush=True)
    results["E2_chosen_plaintext"] = e2()
    for r in results["E2_chosen_plaintext"]:
        print("E2", json.dumps(r), flush=True)
    results["E3_known_plaintext"] = e3()
    for r in results["E3_known_plaintext"]:
        print("E3", json.dumps(r), flush=True)
    results["E4_distinguishers_v1"] = e4()
    print("E4", results["E4_distinguishers_v1"], flush=True)
    json.dump(results, open("results_v1.json", "w"), indent=2)
    print("saved results_v1.json")

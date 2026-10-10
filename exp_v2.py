import json
import math
import sys
import time

import numpy as np
from scipy.stats import binom

import nist
from ciphers import (v1_keys, v1_source_index, v1_encrypt, v2_round_keys, v2_encrypt_blocks,
                     v2_encrypt, v2_decrypt, PRESENT_SBOX, BLOCK, bits_from_int, NONCE_LEN)
import base64
from exp_v1 import log_attack, naive_attack, decrypt_with

rng = np.random.default_rng(20261010)
N_BITS = BLOCK
RS = [1, 2, 3, 4, 5, 6, 8, 12, 16]
R_FINAL = 16
results = {}


def enc_chunked(P, masks, perms, chunk=8192):
    return np.concatenate([v2_encrypt_blocks(P[i:i + chunk], masks, perms) for i in range(0, len(P), chunk)])


def new_keys(r):
    master, nonce = rng.bytes(32), rng.bytes(NONCE_LEN)
    return v2_round_keys(master, nonce, r)


# ----------------------------------------------------------------------
def sbox_properties():
    S = PRESENT_SBOX
    ddt = np.zeros((16, 16), dtype=int)
    for a in range(16):
        for x in range(16):
            ddt[a, S[x] ^ S[x ^ a]] += 1
    lat = np.zeros((16, 16), dtype=int)
    par = lambda v: bin(v).count("1") & 1
    for a in range(16):
        for b in range(16):
            lat[a, b] = sum((-1) ** (par(a & x) ^ par(b & int(S[x]))) for x in range(16))
    return {"max_DDT_entry_nonzero_a": int(ddt[1:].max()),
            "max_differential_probability": float(ddt[1:].max() / 16),
            "max_abs_LAT_nonzero_b": int(np.abs(lat[:, 1:]).max()),
            "max_linear_bias": float(np.abs(lat[:, 1:]).max() / 32)}


# ----------------------------------------------------------------------
def diffusion(K=20, N=1000):
    out = {}
    cdf_ref = binom.cdf(np.arange(N_BITS + 1), N_BITS, 0.5)
    for r in RS:
        hds = []
        for _ in range(K):
            masks, perms = new_keys(r)
            P = rng.integers(0, 2, (N, N_BITS), dtype=np.uint8)
            P2 = P.copy()
            P2[np.arange(N), rng.integers(0, N_BITS, N)] ^= 1
            hds.append(np.sum(v2_encrypt_blocks(P, masks, perms) ^ v2_encrypt_blocks(P2, masks, perms), axis=1))
        h = np.concatenate(hds)
        emp = np.cumsum(np.bincount(h, minlength=N_BITS + 1)) / len(h)
        out[r] = {"mean_fraction": float(h.mean() / N_BITS), "std_fraction": float(h.std() / N_BITS),
                  "min_HD": int(h.min()), "KS_distance_vs_Binomial": float(np.max(np.abs(emp - cdf_ref))),
                  "samples": int(len(h))}
    out["ideal_std_fraction"] = math.sqrt(N_BITS) / 2 / N_BITS
    out["KS_critical_99pct"] = 1.63 / math.sqrt(K * N)
    return out


def key_avalanche(trials=600):
    res = {}
    for what in ("master", "nonce"):
        hds = []
        for _ in range(trials):
            master, nonce = bytearray(rng.bytes(32)), bytearray(rng.bytes(NONCE_LEN))
            tgt = master if what == "master" else nonce
            m2, n2 = bytearray(master), bytearray(nonce)
            t2 = m2 if what == "master" else n2
            pos = int(rng.integers(len(tgt) * 8)); t2[pos // 8] ^= 1 << (pos % 8)
            blk = bits_from_int(0, N_BITS)[None, :]
            c1 = v2_encrypt_blocks(blk, *v2_round_keys(bytes(master), bytes(nonce), R_FINAL))
            c2 = v2_encrypt_blocks(blk, *v2_round_keys(bytes(m2), bytes(n2), R_FINAL))
            hds.append(int(np.sum(c1 ^ c2)))
        res[what] = {"mean_fraction": float(np.mean(hds) / N_BITS), "std_fraction": float(np.std(hds) / N_BITS),
                     "trials": trials}
    return res


# ----------------------------------------------------------------------
def pack_rows(A):
    return np.packbits(A, axis=1)


def differential_collisions(K=10, N=8192):
    out = {}
    for r in RS:
        fr = []
        for _ in range(K):
            masks, perms = new_keys(r)
            P = rng.integers(0, 2, (N, N_BITS), dtype=np.uint8)
            P2 = P.copy(); P2[:, int(rng.integers(N_BITS))] ^= 1
            D = enc_chunked(P, masks, perms) ^ enc_chunked(P2, masks, perms)
            _, inv, cnt = np.unique(pack_rows(D), axis=0, return_inverse=True, return_counts=True)
            fr.append(float(np.mean(cnt[inv.reshape(-1)] > 1)))
        out[r] = float(np.mean(fr))
    return out


def linear_bias(K=5, N=65536):
    out = {}
    noise = 4.5 / (2 * math.sqrt(N))
    for r in RS:
        mx = []
        for _ in range(K):
            masks, perms = new_keys(r)
            P = rng.integers(0, 2, (N, N_BITS), dtype=np.uint8)
            C = enc_chunked(P, masks, perms)
            A = (1 - 2 * P.astype(np.float32)); B = (1 - 2 * C.astype(np.float32))
            corr = (A.T @ B) / N
            mx.append(float(np.max(np.abs(corr)) / 2))
        out[r] = float(np.mean(mx))
    # v1 reference (single key)
    Rk, Kk, Xk = v1_keys(N_BITS, rng)
    src = v1_source_index(Rk, Kk)
    P = rng.integers(0, 2, (N, N_BITS), dtype=np.uint8)
    C = v1_encrypt(P, src, Xk)
    corr = ((1 - 2 * P.astype(np.float32)).T @ (1 - 2 * C.astype(np.float32))) / N
    return {"v2_max_single_bit_bias_by_rounds": out, "v1_max_single_bit_bias": float(np.max(np.abs(corr)) / 2),
            "noise_level_4.5_sigma": noise, "N": N, "K": K}


# ----------------------------------------------------------------------
class RawOracleV2:
    def __init__(self, r):
        self.n = N_BITS
        self.masks, self.perms = new_keys(r)
        self.queries = 0

    def __call__(self, P):
        P = np.atleast_2d(P); self.queries += P.shape[0]
        return v2_encrypt_blocks(P, self.masks, self.perms)


def attacks_on_v2(trials=200):
    out = {}
    for r in (1, 2, 4, R_FINAL):
        acc, exact, valid = [], 0, 0
        for _ in range(trials):
            o = RawOracleV2(r)
            inv, X = log_attack(o)
            valid += int(len(np.unique(inv)) == N_BITS)
            held = rng.integers(0, 2, (1, N_BITS), dtype=np.uint8)
            ct = v2_encrypt_blocks(held, o.masks, o.perms)[0]
            rec = decrypt_with(inv, X, ct)
            acc.append(float(np.mean(rec == held[0])))
            exact += int(np.array_equal(rec, held[0]))
        out[r] = {"mean_bit_accuracy_on_held_out_block": float(np.mean(acc)),
                  "exact_recoveries": f"{exact}/{trials}",
                  "recovered_map_is_valid_permutation": f"{valid}/{trials}"}
    return out


def distinguishers_v2(trials=20000, ind_trials=1500):
    masks, perms = new_keys(R_FINAL)
    wins = 0
    for _ in range(trials // 1000):
        P = rng.integers(0, 2, (1000, N_BITS), dtype=np.uint8)
        P2 = P.copy(); P2[np.arange(1000), rng.integers(0, N_BITS, 1000)] ^= 1
        hd = np.sum(v2_encrypt_blocks(P, masks, perms) ^ v2_encrypt_blocks(P2, masks, perms), axis=1)
        wins += int(np.sum(hd == 1))
    # IND-CPA game through the full scheme, with fresh nonces vs reused nonce
    master = rng.bytes(32)
    msg0, msg1 = b"\x00" * 16, b"\xff" * 16

    def body(tok):
        return base64.b85decode(tok)[NONCE_LEN:-16]

    res = {}
    for label, fixed_nonce in (("fresh_nonce", False), ("reused_nonce", True)):
        correct = 0
        for _ in range(ind_trials):
            nonce = rng.bytes(NONCE_LEN) if fixed_nonce else None
            b = int(rng.integers(2))
            ref = body(v2_encrypt(master, msg0, nonce=nonce))
            chall = body(v2_encrypt(master, msg1 if b else msg0, nonce=nonce))
            guess = 0 if chall == ref else 1
            correct += int(guess == b)
        res[label] = {"success_rate": correct / ind_trials, "advantage": 2 * correct / ind_trials - 1,
                      "trials": ind_trials}
    return {"prp_distinguisher_HD1_hits": f"{wins}/{trials}", "ind_cpa_game": res}


def functional_checks(msgs=500, flips=2000):
    master = rng.bytes(32)
    ok = 0
    for _ in range(msgs):
        m = rng.bytes(int(rng.integers(0, 300)))
        ok += int(v2_decrypt(master, v2_encrypt(master, m)) == m)
    rejected = 0
    for _ in range(flips):
        m = rng.bytes(int(rng.integers(1, 100)))
        raw = bytearray(base64.b85decode(v2_encrypt(master, m)))
        pos = int(rng.integers(len(raw) * 8)); raw[pos // 8] ^= 1 << (pos % 8)
        try:
            v2_decrypt(master, base64.b85encode(bytes(raw)).decode())
        except Exception:
            rejected += 1
    t0 = time.perf_counter()
    for _ in range(20):
        v2_encrypt(master, b"x" * 1024)
    ms_1k = (time.perf_counter() - t0) / 20 * 1000
    t0 = time.perf_counter()
    for _ in range(50):
        v2_round_keys(master, b"n" * NONCE_LEN, R_FINAL)
    ms_ks = (time.perf_counter() - t0) / 50 * 1000
    key_bits = R_FINAL * (N_BITS * math.log2(N_BITS) + N_BITS) + N_BITS
    return {"roundtrip_ok": f"{ok}/{msgs}", "tamper_rejected": f"{rejected}/{flips}",
            "python_ms_per_1KiB_message": ms_1k, "python_ms_key_schedule": ms_ks,
            "derived_key_material_bits_per_message_approx": key_bits}


# ----------------------------------------------------------------------
# NIST data sets (one key per sequence unless stated otherwise)
# ----------------------------------------------------------------------
L_BLOCKS = 1024


def plaintext_sets():
    ldp = [np.zeros(N_BITS, dtype=np.uint8)]
    for i in range(N_BITS):
        v = np.zeros(N_BITS, dtype=np.uint8); v[i] = 1; ldp.append(v)
    for i in range(N_BITS):
        for j in range(i + 1, N_BITS):
            if len(ldp) >= L_BLOCKS: break
            v = np.zeros(N_BITS, dtype=np.uint8); v[i] = v[j] = 1; ldp.append(v)
        if len(ldp) >= L_BLOCKS: break
    ldp = np.stack(ldp[:L_BLOCKS])
    return {"low_density": ldp, "high_density": 1 - ldp,
            "counter": np.stack([bits_from_int(i, N_BITS) for i in range(L_BLOCKS)])}


def nist_experiment(m=200):
    sets = plaintext_sets()
    names = ["low_density", "high_density", "counter", "random_plaintext"]
    out = {}

    def summarise(P):
        ev, lo = nist.evaluate(np.array(P))
        bad = [e["test"] for e in ev if not (e["prop_ok"] and e["unif_ok"])]
        return {"series_passing": len(ev) - len(bad), "of": len(ev),
                "min_proportion": min(e["proportion"] for e in ev), "failing": bad,
                "accept_lower_bound": lo}

    # v1 and v2, one key per sequence
    for scheme in ("v1", "v2"):
        for name in names:
            P = []
            for _ in range(m):
                pts = sets[name] if name != "random_plaintext" else rng.integers(0, 2, (L_BLOCKS, N_BITS), dtype=np.uint8)
                if scheme == "v1":
                    R, K, X = v1_keys(N_BITS, rng)
                    C = v1_encrypt(pts, v1_source_index(R, K), X)
                else:
                    C = v2_encrypt_blocks(pts, *new_keys(R_FINAL))
                P.append(nist.run_all(C.reshape(-1)))
            out[f"{scheme}/{name}"] = summarise(P)
            print(scheme, name, out[f"{scheme}/{name}"], flush=True)
    # v1 with a fresh key for EVERY block and random plaintext
    P = []
    for _ in range(m):
        blocks = []
        for _b in range(L_BLOCKS):
            R, K, X = v1_keys(N_BITS, rng)
            blocks.append(v1_encrypt(rng.integers(0, 2, (1, N_BITS), dtype=np.uint8), v1_source_index(R, K), X))
        P.append(nist.run_all(np.concatenate(blocks).reshape(-1)))
    out["v1/random_plaintext_fresh_key_per_block"] = summarise(P)
    print("v1 fresh-key", out["v1/random_plaintext_fresh_key_per_block"], flush=True)
    out["sequences_per_dataset"] = m
    out["bits_per_sequence"] = L_BLOCKS * N_BITS
    return out


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if part == "a2":
        results["linear_bias"] = linear_bias(); print("linear", results["linear_bias"], flush=True)
        results["attacks_on_v2"] = attacks_on_v2(); print("attacks", json.dumps(results["attacks_on_v2"]), flush=True)
        results["distinguishers_v2"] = distinguishers_v2(); print("dist", results["distinguishers_v2"], flush=True)
        results["functional"] = functional_checks(); print("func", results["functional"], flush=True)
        json.dump(results, open("results_v2a2.json", "w"), indent=2)
    if part in ("all", "a"):
        results["sbox"] = sbox_properties(); print("sbox", results["sbox"], flush=True)
        results["diffusion"] = diffusion(); print("diffusion", json.dumps(results["diffusion"]), flush=True)
        results["key_avalanche"] = key_avalanche(); print("key_avalanche", results["key_avalanche"], flush=True)
        results["differential_collisions"] = differential_collisions(); print("diffcoll", results["differential_collisions"], flush=True)
        results["linear_bias"] = linear_bias(); print("linear", results["linear_bias"], flush=True)
        results["attacks_on_v2"] = attacks_on_v2(); print("attacks", json.dumps(results["attacks_on_v2"]), flush=True)
        results["distinguishers_v2"] = distinguishers_v2(); print("dist", results["distinguishers_v2"], flush=True)
        results["functional"] = functional_checks(); print("func", results["functional"], flush=True)
        json.dump(results, open("results_v2a.json", "w"), indent=2)
    if part in ("all", "b"):
        results = {"nist": nist_experiment()}
        json.dump(results, open("results_v2b.json", "w"), indent=2)
    print("elapsed", round(time.time() - t0, 1), "s")

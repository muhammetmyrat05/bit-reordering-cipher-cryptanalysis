"""
nist.py -- re-implementation of eight p-value series from a subset of the
NIST SP 800-22 Rev.1a tests (Frequency, Block Frequency, Runs, Longest Run
of Ones, Cumulative Sums (forward), Approximate Entropy, Serial (2 p-values)). Validated against the worked examples of the specification
(see validate()).  Input: numpy uint8 array of 0/1.
"""
import math

import numpy as np
from scipy.special import erfc, gammaincc
from scipy.stats import norm

TEST_NAMES = ["Frequency", "BlockFrequency", "Runs", "LongestRun",
              "CumulativeSums", "ApproxEntropy", "Serial1", "Serial2"]
# NOTE: dft() below is kept for reference but is NOT part of run_all(): its
# output for the worked example of SP 800-22 did not match the specification
# and we could not resolve the discrepancy.


def frequency(bits):
    n = len(bits)
    s = np.sum(2.0 * bits - 1)
    return float(erfc(abs(s) / math.sqrt(n) / math.sqrt(2)))


def block_frequency(bits, M=128):
    n = len(bits)
    N = n // M
    blocks = bits[:N * M].reshape(N, M)
    pi = blocks.mean(axis=1)
    chi = 4.0 * M * np.sum((pi - 0.5) ** 2)
    return float(gammaincc(N / 2.0, chi / 2.0))


def runs(bits):
    n = len(bits)
    pi = bits.mean()
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0
    v = 1 + int(np.sum(bits[1:] != bits[:-1]))
    return float(erfc(abs(v - 2.0 * n * pi * (1 - pi)) /
                      (2.0 * math.sqrt(2.0 * n) * pi * (1 - pi))))


def longest_run(bits):
    n = len(bits)
    if n < 128:
        raise ValueError("n too small")
    if n < 6272:
        M, K, N = 8, 3, 16
        pis = [0.2148, 0.3672, 0.2305, 0.1875]
        classes = lambda L: 0 if L <= 1 else (1 if L == 2 else (2 if L == 3 else 3))
    elif n < 750000:
        M, K, N = 128, 5, 49
        pis = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        classes = lambda L: 0 if L <= 4 else (min(L, 9) - 4)
    else:
        raise ValueError("n too large for implemented variants")
    v = np.zeros(K + 1)
    for b in range(N):
        blk = bits[b * M:(b + 1) * M]
        best = cur = 0
        for x in blk:
            if x:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        v[classes(best)] += 1
    chi = sum((v[i] - N * pis[i]) ** 2 / (N * pis[i]) for i in range(K + 1))
    return float(gammaincc(K / 2.0, chi / 2.0))


def cumulative_sums(bits):
    n = len(bits)
    s = np.cumsum(2 * bits.astype(np.int64) - 1)
    z = int(np.max(np.abs(s)))
    if z == 0:
        return 1.0
    sn = math.sqrt(n)
    k1 = np.arange(math.floor((-n / z + 1) / 4), math.floor((n / z - 1) / 4) + 1)
    k2 = np.arange(math.floor((-n / z - 3) / 4), math.floor((n / z - 1) / 4) + 1)
    t1 = np.sum(norm.cdf((4 * k1 + 1) * z / sn) - norm.cdf((4 * k1 - 1) * z / sn))
    t2 = np.sum(norm.cdf((4 * k2 + 3) * z / sn) - norm.cdf((4 * k2 + 1) * z / sn))
    return float(1.0 - t1 + t2)


def _pattern_counts(bits, m):
    n = len(bits)
    if m == 0:
        return np.array([n])
    ext = np.concatenate([bits, bits[:m - 1]]) if m > 1 else bits
    idx = np.zeros(n, dtype=np.int64)
    for j in range(m):
        idx = (idx << 1) | ext[j:j + n]
    return np.bincount(idx, minlength=2 ** m)


def approximate_entropy(bits, m=2):
    n = len(bits)

    def phi(mm):
        c = _pattern_counts(bits, mm).astype(float) / n
        c = c[c > 0]
        return float(np.sum(c * np.log(c)))

    apen = phi(m) - phi(m + 1)
    chi = 2.0 * n * (math.log(2) - apen)
    return float(gammaincc(2 ** (m - 1), chi / 2.0))


def serial(bits, m=3):
    n = len(bits)

    def psi(mm):
        if mm <= 0:
            return 0.0
        c = _pattern_counts(bits, mm).astype(float)
        return (2 ** mm / n) * float(np.sum(c ** 2)) - n

    p1 = psi(m) - psi(m - 1)
    p2 = psi(m) - 2 * psi(m - 1) + psi(m - 2)
    return (float(gammaincc(2 ** (m - 2), p1 / 2.0)),
            float(gammaincc(2 ** (m - 3), p2 / 2.0)))


def dft(bits):
    n = len(bits)
    x = 2.0 * bits - 1
    mag = np.abs(np.fft.fft(x))[:n // 2]
    T = math.sqrt(math.log(1 / 0.05) * n)
    N0 = 0.95 * n / 2
    N1 = float(np.sum(mag < T))
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    return float(erfc(abs(d) / math.sqrt(2)))


def run_all(bits, apen_m=2, serial_m=3):
    bits = np.asarray(bits, dtype=np.uint8)
    s1, s2 = serial(bits, serial_m)
    return [frequency(bits), block_frequency(bits), runs(bits), longest_run(bits),
            cumulative_sums(bits), approximate_entropy(bits, apen_m), s1, s2]


def evaluate(pvals_matrix, alpha=0.01):
    """pvals_matrix: (m sequences, 9 tests). Returns per-test dict following
    SP 800-22 section 4.2: proportion of passes (in acceptance interval?) and
    uniformity of p-values (P_T >= 0.0001)."""
    P = np.asarray(pvals_matrix)
    m = P.shape[0]
    lo = (1 - alpha) - 3 * math.sqrt(alpha * (1 - alpha) / m)
    out = []
    for t in range(P.shape[1]):
        p = P[:, t]
        prop = float(np.mean(p >= alpha))
        hist = np.histogram(p, bins=10, range=(0, 1))[0]
        chi = np.sum((hist - m / 10.0) ** 2 / (m / 10.0))
        pt = float(gammaincc(9 / 2.0, chi / 2.0))
        out.append({"test": TEST_NAMES[t], "proportion": prop,
                    "prop_ok": prop >= lo, "uniformity_p": pt, "unif_ok": pt >= 0.0001})
    return out, lo


def _b(s):
    return np.array([int(c) for c in s], dtype=np.uint8)


def validate():
    eps = _b("1100100100001111110110101010001000100001011010001100001000110100"
             "110001001100011001100010100010111000")
    checks = [
        ("Frequency n=100 (spec 0.109599)", frequency(eps), 0.109599),
        ("Runs n=100 (spec 0.500798)", runs(eps), 0.500798),
        ("BlockFrequency M=10 n=100 (spec 0.706438)", block_frequency(eps, 10), 0.706438),
        ("CumSums fwd n=100 (spec 0.219194)", cumulative_sums(eps), 0.219194),
        ("ApproxEntropy m=2 n=100 (spec 0.235301)", approximate_entropy(eps, 2), 0.235301),
    ]
    s1, s2 = serial(_b("0011011101"), 3)
    checks += [("Serial m=3 p1, n=10 (spec 0.808792)", s1, 0.808792),
               ("Serial m=3 p2, n=10 (spec 0.670320)", s2, 0.670320)]
    ok = True
    for name, got, want in checks:
        flag = abs(got - want) < 2e-5
        ok &= flag
        print(f"  {'OK ' if flag else 'DIFF'} {name}: got {got:.6f}")
    return ok


if __name__ == "__main__":
    print("Validation against SP 800-22 worked examples:")
    allok = validate()
    print("ALL MATCH" if allok else "MISMATCH (see above)")


def statistical_validation(m=300, n=131072, seed=7):
    """Sanity check on data of known quality: good (os.urandom-like) data
    must pass the SP 800-22 proportion/uniformity criteria; biased and
    periodic data must fail."""
    rng = np.random.default_rng(seed)
    sets = {
        "uniform random": lambda: rng.integers(0, 2, n, dtype=np.uint8),
        "biased (P[1]=0.505)": lambda: (rng.random(n) < 0.505).astype(np.uint8),
        "period-128 pattern": lambda: np.tile(rng.integers(0, 2, 128, dtype=np.uint8), n // 128),
    }
    res = {}
    for name, gen in sets.items():
        P = np.array([run_all(gen()) for _ in range(m)])
        ev, lo = evaluate(P)
        res[name] = {"tests_passing_both_criteria": sum(e["prop_ok"] and e["unif_ok"] for e in ev),
                     "of": len(ev), "min_proportion": min(e["proportion"] for e in ev),
                     "accept_lower_bound": lo}
        print(f"  {name:22s}: {res[name]['tests_passing_both_criteria']}/{res[name]['of']} tests pass; "
              f"min proportion {res[name]['min_proportion']:.3f} (accept >= {lo:.4f})")
    return res

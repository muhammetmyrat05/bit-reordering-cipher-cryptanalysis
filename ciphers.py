"""
ciphers.py -- (a) vectorised re-implementation of the original cipher "v1"
and (b) the revised construction "v2" evaluated in the paper.

v1 follows the Python fragments of the original paper:
    c[2i]   = P[R[i]]            c[2i+1] = P[R[n-1-i]]      (i < n/2)
    D[i]    = c[K[i]]            (gather)
    C[i]    = D[i] XOR X[i]
All arrays are 0-indexed here; bit vectors are numpy uint8 arrays.

v2 is a minimal repair of v1 that adds the three missing ingredients named
in the discussion section of the paper:
  * a non-linear layer (PRESENT 4-bit S-box) so that permutations can
    actually diffuse,
  * r rounds of (round-key XOR -> S-box layer -> keyed bit permutation),
  * per-message round keys derived with HKDF from (master key, nonce),
    used in counter mode, plus an HMAC-SHA256 tag (encrypt-then-MAC).
v2 is NOT claimed to be secure: it is evaluated empirically only.
"""
import base64
import hashlib
import hmac
import math
import os

import numpy as np

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def bits_from_int(value: int, n: int) -> np.ndarray:
    return np.array([(value >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


# ----------------------------------------------------------------------
# v1 (original scheme)
# ----------------------------------------------------------------------
def v1_keys(n: int, rng: np.random.Generator):
    R = rng.permutation(n)
    K = rng.permutation(n)
    X = rng.integers(0, 2, n, dtype=np.uint8)
    return R, K, X


def v1_source_index(R: np.ndarray, K: np.ndarray) -> np.ndarray:
    """src[i] = index of the plaintext bit that ends up at ciphertext
    position i (before the XOR mask)."""
    n = len(R)
    h = n // 2
    src_c = np.empty(n, dtype=np.int64)
    src_c[0::2] = R[:h]
    src_c[1::2] = R[::-1][:h]          # R[n-1-i], i < h
    return src_c[K]


def v1_encrypt(P: np.ndarray, src: np.ndarray, X: np.ndarray) -> np.ndarray:
    """P: (N, n) uint8 -> (N, n) uint8"""
    return P[:, src] ^ X


# ----------------------------------------------------------------------
# HKDF (RFC 5869) with HMAC-SHA256
# ----------------------------------------------------------------------
def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    if not salt:
        salt = b"\x00" * 32
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    if length > 255 * 32:
        raise ValueError("HKDF-Expand output too long")
    okm, t, i = b"", b"", 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1
    return okm[:length]


class ByteStream:
    """Deterministic byte stream from HKDF-Expand (single call)."""

    def __init__(self, prk: bytes, info: bytes, length: int = 255 * 32):
        self.buf = hkdf_expand(prk, info, length)
        self.pos = 0

    def take(self, k: int) -> bytes:
        if self.pos + k > len(self.buf):
            raise RuntimeError("byte stream exhausted")
        out = self.buf[self.pos:self.pos + k]
        self.pos += k
        return out

    def uniform_below(self, m: int) -> int:
        """Unbiased integer in [0, m) by rejection sampling on bytes (m<=256)."""
        limit = 256 - (256 % m)
        while True:
            b = self.take(1)[0]
            if b < limit:
                return b % m


def fisher_yates(stream: ByteStream, n: int) -> np.ndarray:
    a = list(range(n))
    for i in range(n - 1, 0, -1):
        j = stream.uniform_below(i + 1)
        a[i], a[j] = a[j], a[i]
    return np.array(a, dtype=np.int64)


# ----------------------------------------------------------------------
# v2 block function (SPN) -- 128-bit state
# ----------------------------------------------------------------------
PRESENT_SBOX = np.array([0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
                         0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2], dtype=np.uint8)
SBOX_BITS = np.array([[(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1]
                      for v in PRESENT_SBOX], dtype=np.uint8)           # (16, 4)
NIB_W = np.array([8, 4, 2, 1], dtype=np.int64)

BLOCK = 128


def sbox_layer(S: np.ndarray) -> np.ndarray:
    N, n = S.shape
    nib = S.reshape(N, n // 4, 4).astype(np.int64) @ NIB_W              # (N, n/4)
    return SBOX_BITS[nib].reshape(N, n)


def v2_round_keys(master: bytes, nonce: bytes, rounds: int, n: int = BLOCK):
    """Per-message round keys: rounds+1 XOR masks and `rounds` bit permutations."""
    prk = hkdf_extract(nonce, master)
    stream = ByteStream(prk, b"v2-round-keys")
    perms = [fisher_yates(stream, n) for _ in range(rounds)]
    masks = [bytes_to_bits(stream.take(n // 8)) for _ in range(rounds + 1)]
    return masks, perms


def v2_encrypt_blocks(S: np.ndarray, masks, perms) -> np.ndarray:
    """S: (N, n) bit array. Round: XOR mask -> S-box layer -> permutation."""
    r = len(perms)
    for i in range(r):
        S = S ^ masks[i]
        S = sbox_layer(S)
        S = S[:, perms[i]]
    return S ^ masks[r]


def counter_blocks(start: int, count: int, n: int = BLOCK) -> np.ndarray:
    return np.stack([bits_from_int(start + i, n) for i in range(count)])


DEFAULT_ROUNDS = 16
TAG_LEN = 16
NONCE_LEN = 12


def v2_encrypt(master: bytes, plaintext: bytes, aad: bytes = b"", rounds: int = DEFAULT_ROUNDS,
               nonce: bytes = None) -> str:
    """CTR mode over the SPN + HMAC-SHA256 (encrypt-then-MAC). Returns Base85 text."""
    if nonce is None:
        nonce = os.urandom(NONCE_LEN)
    masks, perms = v2_round_keys(master, nonce, rounds)
    nblocks = max(1, math.ceil(len(plaintext) / (BLOCK // 8)))
    ks = v2_encrypt_blocks(counter_blocks(0, nblocks), masks, perms)
    keystream = bits_to_bytes(ks.reshape(-1))[:len(plaintext)]
    ct = bytes(a ^ b for a, b in zip(plaintext, keystream))
    mac_key = hkdf_expand(hkdf_extract(nonce, master), b"v2-mac-key", 32)
    tag = hmac.new(mac_key, nonce + aad + ct, hashlib.sha256).digest()[:TAG_LEN]
    return base64.b85encode(nonce + ct + tag).decode()


def v2_decrypt(master: bytes, token: str, aad: bytes = b"", rounds: int = DEFAULT_ROUNDS) -> bytes:
    raw = base64.b85decode(token)
    nonce, ct, tag = raw[:NONCE_LEN], raw[NONCE_LEN:-TAG_LEN], raw[-TAG_LEN:]
    mac_key = hkdf_expand(hkdf_extract(nonce, master), b"v2-mac-key", 32)
    expect = hmac.new(mac_key, nonce + aad + ct, hashlib.sha256).digest()[:TAG_LEN]
    if not hmac.compare_digest(tag, expect):
        raise ValueError("authentication failed")
    masks, perms = v2_round_keys(master, nonce, rounds)
    nblocks = max(1, math.ceil(len(ct) / (BLOCK // 8)))
    ks = v2_encrypt_blocks(counter_blocks(0, nblocks), masks, perms)
    keystream = bits_to_bytes(ks.reshape(-1))[:len(ct)]
    return bytes(a ^ b for a, b in zip(ct, keystream))


if __name__ == "__main__":
    # --- sanity checks -------------------------------------------------
    # HKDF against `cryptography`
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    ikm, salt, info = os.urandom(32), os.urandom(12), b"check"
    mine = hkdf_expand(hkdf_extract(salt, ikm), info, 100)
    ref = HKDF(algorithm=hashes.SHA256(), length=100, salt=salt, info=info).derive(ikm)
    assert mine == ref, "HKDF mismatch"
    print("HKDF matches cryptography library")

    # S-box is a permutation
    assert sorted(PRESENT_SBOX.tolist()) == list(range(16))

    # v1 round trip structure: composite map is a permutation
    rng = np.random.default_rng(1)
    R, K, X = v1_keys(128, rng)
    src = v1_source_index(R, K)
    assert sorted(src.tolist()) == list(range(128))
    print("v1 composite transform is a permutation")

    # v2 round trip + tamper detection
    master = os.urandom(32)
    msg = b"hello constrained world, this is a v2 test message!"
    tok = v2_encrypt(master, msg)
    assert v2_decrypt(master, tok) == msg
    raw = bytearray(base64.b85decode(tok)); raw[20] ^= 1
    try:
        v2_decrypt(master, base64.b85encode(bytes(raw)).decode()); raise SystemExit("tamper NOT detected")
    except ValueError:
        pass
    print("v2 round trip OK, tamper detected; sample token:", tok[:40], "...")

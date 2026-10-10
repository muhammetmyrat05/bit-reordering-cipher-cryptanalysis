#!/bin/sh
# Reproduces every number in the paper. Run from this directory. Total time: about 3 minutes on one core (175 s on our machine).
set -e
python3 ciphers.py                 # self-checks: HKDF vs the `cryptography` library, round trips, tamper detection
python3 nist.py                    # validation of the NIST SP 800-22 subset against the specification's worked examples
python3 exp_v1.py                  # original cipher: avalanche validation, chosen/known-plaintext attacks, distinguishers -> results_v1.json
python3 exp_extra.py               # Hamming-distance preservation, key-space sizes -> res_extra.json
python3 kdcheck.py                 # HKDF-derived R, K, X structure check -> results_kd.json
python3 dhtime.py                  # Diffie-Hellman cost (can take ~20 s or more for parameter generation) -> dhtime.json
for p in sbox diff kav coll lin att dist func nist; do python3 run_part.py $p; done   # repaired construction v2 -> res_<part>.json
python3 compile_results.py         # -> results_all.json

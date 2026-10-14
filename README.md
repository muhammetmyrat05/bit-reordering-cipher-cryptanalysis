# Cryptanalysis of a Hybrid Bit-Reordering and Permutation-Based Cipher

Code and data accompanying the paper *"Cryptanalysis of a Hybrid Bit-Reordering and Permutation-Based
Cipher: A Red-Team Evaluation of Our Own Construction"* (submitted to CSAI 2026).

**Authors:** Muhammetmyrat Annamammedov, Yslam Orazov, Ayjemal Ussayeva, Ayjemal Abdyrahymova, Ayjeren Sahetlyyeva
Department of Cyberphysical Systems, Oguz han Engineering and Technology University of Turkmenistan.

The paper attacks a lightweight cipher we proposed earlier (called **v1** here), and then tests a
conventional repair (**v2**). Every number in the paper comes from `results_all.json`, which these
scripts produce.

> **Warning.** v1 is broken and must not be used. v2 is a diagnostic repair that was evaluated
> empirically only; it is **not** claimed to be secure. For real systems use a standardised
> authenticated cipher such as Ascon-AEAD128 (NIST SP 800-232) or AES-GCM.

## Quick start
```
pip install -r requirements.txt
sh run_all.sh
```
`run_all.sh` runs every experiment and writes `results_all.json` (about 3 minutes on one core). In our from-scratch test run, every value except timings was identical to the published `results_all.json`. Smaller checks:
```
python3 ciphers.py                                   # self-checks (HKDF, round trips, tamper detection)
python3 nist.py                                      # NIST test implementation vs. the specification's examples
python3 -c "import nist; nist.statistical_validation()"
```

## Files
| File | Purpose |
|---|---|
| `ciphers.py` | Vectorised v1 (written from the code fragments of the original paper), v2, and an HKDF (RFC 5869) helper checked against the `cryptography` library |
| `nist.py` | Eight p-value series of NIST SP 800-22 (Frequency, Block Frequency, Runs, Longest Run, Cumulative Sums forward, Approximate Entropy, Serial x2) plus validation code |
| `exp_v1.py` | Attacks on v1: avalanche validation, single-bit probing, position-code attack (ceil(log2 n)+1 queries), known-plaintext attack, distinguishers |
| `exp_v2.py`, `run_part.py` | Evaluation of v2: diffusion, differential collisions, linear bias, attacks, distinguishers, NIST data sets |
| `exp_extra.py` | Hamming-distance preservation and key-space sizes |
| `kdcheck.py` | Statistical check of the HKDF-based derivation of R, K, X (our instantiation) |
| `dhtime.py` | Diffie-Hellman cost measurements |
| `run_all.sh`, `compile_results.py` | Run everything / merge the results |
| `results_all.json` | The values reported in the paper |

## Notes for reproducers
* Tested with Python 3.12, NumPy 2.4, SciPy 1.17 and cryptography 46. Seeds are fixed in each script.
* Run the parts of the v2 evaluation through `run_part.py`, as `run_all.sh` does. Each part uses its own random stream; running `exp_v2.py` as a whole draws random numbers in a different order and gives slightly different values (same conclusions).
* Timings (milliseconds, seconds) and the Diffie-Hellman measurements depend on the machine and are indicative only.
* The NIST data sets are shortened versions of those in Soto (NISTIR 6390): 1024 blocks and 200 sequences per data set. The counter data set is our own addition.
* The original paper does not say how HKDF output becomes permutations; `kdcheck.py` tests our own instantiation only.

## How to cite
If you use this code, please cite the paper (venue details to be added once published):

> Annamammedov, M.: Cryptanalysis of a Hybrid Bit-Reordering and Permutation-Based Cipher:
> A Red-Team Evaluation of Our Own Construction. CSAI 2026 (under review).

## License
MIT, see `LICENSE`.

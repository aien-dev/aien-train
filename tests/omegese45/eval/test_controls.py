#!/usr/bin/env python3
"""Independent-control machinery tests; no checkpoint needed."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(ROOT, "eval/omegese45"))
from oe_controls import nearest, split_corpus
from oe_wilson import wilson


def main():
    failures = []
    def check(name, ok, detail=""):
        print("C %s %s %s" % (name, "PASS" if ok else "FAIL", detail))
        if not ok:
            failures.append(name)
    # Wolfram query and full outputs are in evidence/.../wilson-verification.txt.
    for k, n, expected in [(0, 10, [0, 0.27753279986288925]),
                            (5, 10, [0.23659309051256398, 0.76340690948743602]),
                            (10, 10, [0.72246720013711075, 1])]:
        actual = wilson(k, n)
        check("wilson_%d_%d" % (k, n), all(abs(a - b) < 1e-12 for a, b in zip(actual, expected)), repr(actual))
    check("wilson_zero_denominator", wilson(0, 0) is None)
    try:
        wilson(11, 10)
        bad = False
    except ValueError:
        bad = True
    check("wilson_bad_counts", bad)
    check("retrieval_tie_first", nearest([1, 2], [(8, [1, 3]), (9, [4, 2])]) == (8, 1))
    check("retrieval_length_penalty", nearest([1, 2], [(3, [1, 2, 3]), (4, [1, 2])]) == (4, 0))
    check("corpus_split", split_corpus("oe45 v1\na\noe45 v1\nb\n") == [["a"], ["b"]])
    with tempfile.TemporaryDirectory(prefix="oe45-controls-negative-") as tmp:
        sealed = os.path.join(tmp, "sealed")
        seed = os.path.join(tmp, "seed")
        open(sealed, "w").write("tampered\n")
        open(seed, "w").write("00\n")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "eval/omegese45/oe_controls.py"),
                            sealed, seed, "-", "-", "-", "-", "-", os.path.join(tmp, "out")],
                           capture_output=True, text=True)
        check("commitment_mismatch_refused_before_scoring", r.returncode == 1
              and "INVALID commitment hash mismatch" in r.stdout
              and not os.path.exists(os.path.join(tmp, "out", "counts.json")))
    print("OE45_CONTROL_TESTS %s" % ("FAIL" if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

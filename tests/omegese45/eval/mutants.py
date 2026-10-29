#!/usr/bin/env python3
"""Mutation controls for the eval suite (aien-train #45). Standard library only.

Each mutant copies eval/omegese45 to a temporary directory, disables exactly
one check by a literal source replacement, and reruns test_eval.py against
the copy (OE45_EVAL_DIR). A mutant is KILLED when the suite fails AND the
named test is among the failures. Prints `M <name> KILLED|SURVIVED` and
`OE45_EVAL_MUTANTS PASS|FAIL`.
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, "..", "..", "..", "eval", "omegese45"))

# (name, file, literal old, literal new, test that must fail)
MUTANTS = [
    ("skip_independent_recompute", "oe_an0.py",
     'rec("recompute.output_digest", r["output_digest"] == want,',
     'rec("recompute.output_digest", True,',
     "checkdemo.neg.output_digest_resigned"),
    ("accept_missing_response", "check-demo",
     'rep.ok("english.present", False,',
     'rep.ok("english.present", True,',
     "checkdemo.neg.missing_response"),
    ("accept_high_s", "oe_an0.py",
     "    if s > N // 2:\n",
     "    if False:\n",
     "real.neg.high_s_refused"),
    ("ignore_english_seeds", "oe_en.py",
     'rec("english.seeds", claim["seeds"] == tuple(goal.seeds),',
     'rec("english.seeds", True,',
     "checkdemo.neg.english_number"),
    ("ignore_facts_order", "check-demo",
     '            return "row%d job %d after its ok" % (i + 1, r["job"])\n',
     '            pass\n',
     "checkdemo.neg.facts_row_after_ok"),
    ("accept_malformed_nonok_dag", "check-demo",
     "            if r[c] != \"-\" and not _hex(r[c], 64):\n",
     "            if False:\n",
     "checkdemo.neg.facts_nonok_dag_short"),
]


def main():
    fails = []
    for name, fname, old, new, must_fail in MUTANTS:
        with tempfile.TemporaryDirectory(prefix="oe45-mutant-") as tmp:
            d = os.path.join(tmp, "omegese45")
            shutil.copytree(EVAL, d, ignore=shutil.ignore_patterns("__pycache__"))
            p = os.path.join(d, fname)
            src = open(p).read()
            if src.count(old) != 1:
                print("M %s SURVIVED mutation site not unique (count=%d)" % (name, src.count(old)))
                fails.append(name)
                continue
            open(p, "w").write(src.replace(old, new))
            env = dict(os.environ, OE45_EVAL_DIR=d, PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run([sys.executable, os.path.join(HERE, "test_eval.py")],
                               capture_output=True, text=True, env=env)
            failed = [l.split()[1] for l in r.stdout.splitlines() if l.startswith("T ") and l.split()[2] == "FAIL"]
            killed = r.returncode != 0 and must_fail in failed
            print("M %s %s rc=%d failed=%s" % (name, "KILLED" if killed else "SURVIVED", r.returncode,
                                               ",".join(failed)))
            if not killed:
                fails.append(name)
    print("OE45_EVAL_MUTANTS %s %s" % ("FAIL" if fails else "PASS", " ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

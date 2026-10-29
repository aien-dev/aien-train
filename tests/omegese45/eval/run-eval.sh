#!/bin/sh
# Eval-lane suite (aien-train #45). Prints OE45_EVAL_PASS when the eval
# generator matches the hand-written spec cases, the audit rejects every
# hand-written violation, the readers verify the real Aienet-0 baseline
# artifacts, and check-demo catches every negative control.
# Usage: tests/omegese45/eval/run-eval.sh [CONTRACT_VOCAB_TSV]
set -eu
here=$(cd "$(dirname "$0")" && pwd)
(cd "$here/fixtures/baseline" && sha256sum -c --quiet SHA256SUMS)
if python3 "$here/test_eval.py" "$@"; then
  echo OE45_EVAL_PASS
else
  echo OE45_EVAL_FAIL
  exit 1
fi

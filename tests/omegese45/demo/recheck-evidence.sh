#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# aien-train #45 demo: re-run the independent checker over committed demo receipts.
#
#   sh tests/omegese45/demo/recheck-evidence.sh [RUN ...]
#
# RUN  a directory under evidence/OMEGESE45/demo/ (default: the current receipts
#      run-reference-v13 run-trained-v13, docs/omegese45/DEMO.md "Files").
# For every checked case of every RUN (each cases/*/ and cases/tampered/{receipt,english}/
# that holds check.out), the unchanged eval/omegese45/check-demo runs on the committed
# case directory; its stdout+stderr must equal the committed check.out byte for byte and
# its exit code must equal the committed check.rc. The committed runs were produced with
# the runtime identity pinned in check-demo, so a fresh clone can verify them without
# that runtime or any private key (EVAL-CONTRACT.md §10).
# Two negative controls run on scratch copies and must be detected:
#   flip   byte 200 of happy-mm's MJR1 receipt: check-demo must fail row1.mjr1.signature;
#   edit   one byte of a committed check.out: the byte comparison must report a difference.
# This re-verifies recorded receipts; it does not execute the demo (run-demo.sh does).
# Exits: 0 OE45_RECHECK_PASS, 1 OE45_RECHECK_FAIL, 68 checker not runnable, 2 usage.
set -u
export PYTHONDONTWRITEBYTECODE=1   # the checker leaves no bytecode in the tree
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
EV=$REPO/evidence/OMEGESE45/demo
CHECK=$REPO/eval/omegese45/check-demo
[ $# -gt 0 ] || set -- run-reference-v13 run-trained-v13
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null \
  || { echo "OE45_RECHECK_NOT_RUN python3 >= 3.8 missing"; exit 68; }
[ -f "$CHECK" ] || { echo "OE45_RECHECK_NOT_RUN $CHECK missing"; exit 68; }
S=$(mktemp -d) || exit 68
trap 'rm -rf "$S"' EXIT

same() { # case dir -> 0 when check-demo reproduces check.out and check.rc
  python3 "$CHECK" "$1" > "$S/out" 2>&1; r=$?
  [ -f "$1/check.out" ] && [ -f "$1/check.rc" ] && [ "$r" = "$(cat "$1/check.rc")" ] && cmp -s "$S/out" "$1/check.out"
}

n=0; bad=""
for run in "$@"; do
  case $run in */*|.*|"") echo "recheck-evidence: RUN must be a directory name under $EV" >&2; exit 2 ;; esac
  [ -d "$EV/$run/cases" ] || { echo "recheck-evidence: $EV/$run/cases missing" >&2; exit 2; }
  m=0
  for out in "$EV/$run"/cases/*/check.out "$EV/$run"/cases/tampered/*/check.out; do
    [ -f "$out" ] || continue
    c=$(dirname "$out"); id=${c#"$EV/"}
    m=$((m + 1)); n=$((n + 1))
    if same "$c"; then echo "RECHECK SAME $id rc=$(cat "$c/check.rc") $(tail -n 1 "$c/check.out")"
    else echo "RECHECK DIFF $id rc=$(cat "$c/check.rc" 2>/dev/null)"; diff "$c/check.out" "$S/out" | sed 's/^/  /'; bad="$bad $id"; fi
  done
  [ "$m" -gt 0 ] || { echo "RECHECK EMPTY $run (no checked case)"; bad="$bad $run"; }
done

# negative control 1: a flipped receipt byte must break the signature check
first=$EV/$1/cases/happy-mm
cp -R "$first" "$S/flip"
mj=$(ls "$S/flip/artifacts/"*.mjr1 2>/dev/null | head -n 1)
if [ -n "$mj" ]; then
  b=$(od -An -tu1 -j 200 -N 1 "$mj" | tr -d ' ')
  printf "$(printf '\\%03o' $(( (b + 1) % 256 )))" | dd of="$mj" bs=1 seek=200 conv=notrunc 2>/dev/null
  python3 "$CHECK" "$S/flip" > "$S/flip.out" 2>&1; r=$?
  if [ $r -eq 1 ] && grep -q '^CHECK row1.mjr1.signature FAIL' "$S/flip.out"; then echo "CONTROL flip CAUGHT rc=$r $(tail -n 1 "$S/flip.out")"
  else echo "CONTROL flip MISSED rc=$r"; bad="$bad control-flip"; fi
else echo "CONTROL flip MISSED (no MJR1 receipt in $1/cases/happy-mm)"; bad="$bad control-flip"; fi

# negative control 2: an edited committed check.out must not compare equal
cp -R "$first" "$S/edit"
sed '1s/PASS/PAS5/' "$S/edit/check.out" > "$S/edit.t" && mv "$S/edit.t" "$S/edit/check.out"
if cmp -s "$S/edit/check.out" "$first/check.out"; then echo "CONTROL edit MISSED (edit did not change check.out)"; bad="$bad control-edit"
elif same "$S/edit"; then echo "CONTROL edit MISSED (comparison accepted an edited check.out)"; bad="$bad control-edit"
else echo "CONTROL edit CAUGHT"; fi

if [ -z "$bad" ]; then echo "OE45_RECHECK_PASS runs=$# cases=$n controls=2"; exit 0; fi
echo "OE45_RECHECK_FAIL$bad"
exit 1

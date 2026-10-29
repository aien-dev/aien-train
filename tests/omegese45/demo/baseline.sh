#!/bin/sh
# Reproduce the Aienet-0 baseline (aien-protocols e705fdc9 evidence/aienet0-20261028/exercised/baseline.cmd)
# on this Spark over loopback: one VERIFIED matmul_u32 m=k=n=64 seed=2716205057, then the offline check.
# usage: tests/omegese45/demo/baseline.sh RT EVIDENCE_DIR [PORT]
# Exit 0 only when run=0, serve=0, a VERIFIED line is printed and `aienet0-run check` passes.
set -u
RT=${1:?usage: baseline.sh RT EVIDENCE_DIR [PORT]}
E=${2:?usage: baseline.sh RT EVIDENCE_DIR [PORT]}
PORT=${3:-14330}
. "$RT/pins.env"
HEAVY="taskset -c 10-14 nice -n 19 ionice -c3"
mkdir -p "$E"
W=$(mktemp -d)
OUT=$W/out
scmd="$HEAVY $AN0_SERVE --connect 127.0.0.1:$PORT --keys $WKEYS --pins $WPINS --state $W/wstate --connect-timeout 60"
rcmd="$HEAVY $AN0_RUN run --listen 127.0.0.1:$PORT --keys $KEYS --pins $PINS --out $OUT --recompute $RECOMPUTE --recompute-sha256 $RECOMPUTE_SHA256 --m 64 --k 64 --n 64 --seed 2716205057 --accept-timeout 90"
printf 'serve: %s\nrun:   %s\n' "$scmd" "$rcmd" | sed "s#$W#\$W#g" > "$E/baseline.cmd"
$scmd > "$E/baseline.serve.out" 2> "$E/baseline.serve.err" < /dev/null &
sp=$!
$rcmd > "$E/baseline.run.out" 2> "$E/baseline.run.err"
rr=$?
wait "$sp"; sr=$?
fail=0
[ "$rr" = 0 ] && [ "$sr" = 0 ] && grep -q '^VERIFIED ' "$E/baseline.run.out" || fail=1
dag=$(awk '/^VERIFIED /{print $2}' "$E/baseline.run.out")
cr=NOT_RUN
if [ -n "$dag" ]; then
  "$AN0_RUN" check --bundle "$OUT/$dag.an0b" --mjr1 "$OUT/$dag.mjr1" --mjv "$OUT/$dag.mjv" \
    --keys "$KEYS" --pins "$PINS" > "$E/baseline.check.out" 2>&1
  cr=$?
  [ "$cr" = 0 ] || fail=1
  mkdir -p "$E/artifacts"
  cp "$OUT/$dag.an0b" "$OUT/$dag.mjr1" "$OUT/$dag.mjv" "$E/artifacts/"
  (cd "$E/artifacts" && sha256sum "$dag.an0b" "$dag.mjr1" "$dag.mjv") > "$E/baseline.artifacts.sha256"
fi
cp "$W/wstate/seen" "$E/worker-seen.after" 2>/dev/null || true
sed -i "s#$W#\$W#g" "$E"/baseline.*.out "$E"/baseline.*.err 2>/dev/null || true
cat "$RT/pins.env" | sed "s#$RT#\$RT#g" > "$E/pins.env"
cp "$RT/coordinator-public-keys.txt" "$RT/worker-public-keys.txt" "$E/"
printf 'run_rc=%s serve_rc=%s check_rc=%s port=%s host=%s date=%s\n' "$rr" "$sr" "$cr" "$PORT" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$E/baseline.rc"
cat "$E/baseline.rc" "$E/baseline.run.out" "$E/baseline.check.out" 2>/dev/null
rm -rf "$W"
[ "$fail" = 0 ] && echo BASELINE_VERIFIED || echo BASELINE_FAIL
exit "$fail"

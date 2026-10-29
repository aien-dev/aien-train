#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# aien-train #45 demo (demo lane): nine end-to-end cases, docs/omegese45/DEMO.md.
#
#   sh tests/omegese45/demo/run-demo.sh RT OUT
#
# RT   runtime dir prepared by tests/omegese45/demo/setup-runtime.sh (built if absent).
# OUT  fresh output dir: bin/, work/, cases/<case>/ receipts, run-demo.summary.
# Env:
#   OE45_CORE=model|reference   model (default): the pipeline core oe45core with CKPT.
#                               reference: the eval-owned reference policy
#                               (eval/omegese45/oe45lib.py policy_next) as a CONTROL
#                               core; it exercises every non-model step, not the model.
#   CKPT=DIR                    checkpoint dir (ck.a/ck.b) for OE45_CORE=model (required).
#   MODE=G|U                    first-decode mode (default G, interface §9).
#   PORTBASE=N                  first loopback port (default 14340; never 14323).
#   OMC, OMEGA                  omc bf54b19f and its omega-mv checkout (default /tmp/om-main).
# This is test harness glue (interface §0): it only calls binaries, copies files,
# compares and flips bytes. Product logic is in native/omegese45/**.om.
# Exits: 0 every case met its expectation (OE45_DEMO_PASS), 60 setup/build,
# 67 at least one case missed its expectation (OE45_DEMO_FAIL), 68 eval checker
# not runnable, 2 usage. Driver exits 61..66, 69: native/omegese45/hive/hive.om.
set -u
export PYTHONDONTWRITEBYTECODE=1   # the eval checker and the control core leave no bytecode in the tree
[ $# -eq 2 ] || { echo "usage: run-demo.sh RT OUT" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
RT=$1
case $2 in /*) OUT=$2 ;; *) OUT=$(pwd)/$2 ;; esac
CORE_KIND=${OE45_CORE:-model}
MODE=${MODE:-G}
PORTBASE=${PORTBASE:-14340}
OMC=${OMC:-/tmp/om-main/seed/out/omc}
OMEGA=${OMEGA:-/tmp/om-main}
OMC_SHA=62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b
HEAVY="taskset -c 10-14 nice -n 19 ionice -c3"
N=$REPO/native/omegese45
T=$N/translate
CHECK=$REPO/eval/omegese45/check-demo

[ "$PORTBASE" -ne 14323 ] 2>/dev/null || { echo "run-demo: bad PORTBASE" >&2; exit 2; }
[ -e "$OUT" ] && { echo "run-demo: $OUT exists; give a fresh OUT" >&2; exit 2; }
case $CORE_KIND in
  model) [ -n "${CKPT:-}" ] && [ -d "${CKPT:-}" ] || { echo "run-demo: OE45_CORE=model needs CKPT=DIR (NOT_RUN)" >&2; exit 60; } ;;
  reference) CKPT=- ;;
  *) echo "run-demo: OE45_CORE must be model or reference" >&2; exit 2 ;;
esac
case $MODE in G|U) ;; *) echo "run-demo: MODE must be G or U" >&2; exit 2 ;; esac
case $CKPT in /*|-) ;; *) CKPT=$(cd "$CKPT" && pwd) ;; esac

BIN=$OUT/bin; W=$OUT/work; CS=$OUT/cases
mkdir -p "$BIN" "$W/wrap" "$W/conf" "$CS" || exit 60
SUM=$OUT/run-demo.summary
say() { echo "$*"; echo "$*" >> "$SUM"; }

# ---- 1. pins and builds -------------------------------------------------------
got=$(sha256sum "$OMC" 2>/dev/null | cut -d' ' -f1)
[ "$got" = "$OMC_SHA" ] || { say "SETUP FAIL omc sha256 $got != $OMC_SHA"; exit 60; }
say "PIN omc $OMC sha256 $OMC_SHA (omega-mv bf54b19f)"
say "PIN aien-train $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
build() { # name files...
  b=$1; shift
  cat "$@" | $HEAVY "$OMC" > "$BIN/$b" 2> "$BIN/$b.build.err" || { say "SETUP FAIL build $b"; exit 60; }
  chmod +x "$BIN/$b"
}
build hive $N/vocab.om $N/msg.om $N/mjr1.om $N/hive/hive.om
build en2goal $T/otio.om $T/en2goal.om $T/otfrag.om $T/main-en2goal.om
build goal2en $T/otio.om $N/vocab.om $N/msg.om $T/otmsg.om $T/goal2en.om $T/otfrag.om $T/main-goal2en.om
build rep2en $T/otio.om $N/vocab.om $N/msg.om $T/otmsg.om $T/rep2en.om $T/otfrag.om $T/main-rep2en.om
if [ "$CORE_KIND" = model ]; then
  $HEAVY env OMC="$OMC" OMEGA="$OMEGA" OUT="$BIN/core" sh "$REPO/tests/omegese45/core/build.sh" > "$BIN/core.build.out" 2>&1 \
    || { say "SETUP FAIL core build (tests/omegese45/core/build.sh)"; exit 60; }
  CORE=$BIN/core/oe45core
  say "PIN ckpt $CKPT $( (cd "$CKPT" && sha256sum ck.a ck.b 2>/dev/null) | tr '\n' ' ')"
  $CORE info "$CKPT" > "$OUT/ckpt.info" 2>&1; say "CKPT info rc=$? $(cat "$OUT/ckpt.info")"
else
  CORE=$W/wrap/refcore
  cat > "$CORE" <<EOF
#!/bin/sh
# CONTROL core (harness only): the eval-owned reference policy, not a model.
[ "\$1" = decode ] && [ \$# -eq 4 ] || exit 2
exec /usr/bin/python3 - "\$4" <<'PY'
import sys
sys.path.insert(0, "$REPO/eval/omegese45")
import oe45lib as L
try:
    lines = L.parse_episode(open(sys.argv[1]).read())
    msgs = [L.parse_line(x) for x in lines]
    out = L.policy_next(msgs)
except Exception:
    sys.exit(40)
sys.stdout.write(out + "\n")
PY
EOF
  chmod +x "$CORE"
  say "CORE reference-policy CONTROL ($REPO/eval/omegese45/oe45lib.py policy_next); no model is exercised"
fi
(cd "$BIN" && sha256sum hive en2goal goal2en rep2en; [ "$CORE_KIND" = model ] && sha256sum core/oe45core) > "$OUT/binaries.sha256"
while read -r l; do say "BIN $l"; done < "$OUT/binaries.sha256"

$HEAVY sh "$HERE/setup-runtime.sh" "$RT" > "$OUT/setup-runtime.out" 2>&1
rc=$?
[ $rc -eq 0 ] || { say "SETUP setup-runtime rc=$rc (3 = NOT_RUN)"; exit 60; }
. "$RT/pins.env"
for v in AN0_RUN AN0_SERVE RECOMPUTE; do
  eval "p=\$$v; want=\$${v}_SHA256"
  got=$(sha256sum "$p" | cut -d' ' -f1)
  [ "$got" = "$want" ] || { say "SETUP FAIL $v sha256 $got != $want"; exit 60; }
  say "PIN $v $p sha256 $got"
done
say "PIN open-humanity $OH_COMMIT aien-protocols $PROTO_COMMIT sha-unit omega-mv $SHA_UNIT_COMMIT"
VERIFIER=$(grep '^verifier.p256 ' "$RT/coordinator-public-keys.txt" | cut -d' ' -f2)
[ -n "$VERIFIER" ] || { say "SETUP FAIL no verifier key"; exit 60; }
python3 "$CHECK" > /dev/null 2>&1; [ $? -eq 2 ] || { say "SETUP eval checker not runnable"; exit 68; }

WSTATE=$W/wstate; mkdir -p "$WSTATE"   # one worker identity and replay state per run

# ---- 2. fault wrappers (test dir only; the driver sees them as RUN/SERVE/CORE) ---
FLAKY=$W/wrap/serve-down; SDOWN=$W/serve-down.flag
cat > "$FLAKY" <<EOF
#!/bin/sh
# Injected transport fault: while the flag exists the worker is down for one dispatch.
if [ -e "$SDOWN" ]; then rm -f "$SDOWN"; echo "fault: worker down (injected)" >&2; exit 0; fi
exec "$AN0_SERVE" "\$@"
EOF
BADCORE=$W/wrap/core-bad; CBAD=$W/core-bad.flag
cat > "$BADCORE" <<EOF
#!/bin/sh
# Injected invalid action: while the flag exists, one decode loses its final \`end\`.
if [ -e "$CBAD" ]; then
  rm -f "$CBAD"
  o=\$("$CORE" "\$@"); r=\$?
  printf '%s\n' "\${o% end}"
  exit \$r
fi
exec "$CORE" "\$@"
EOF
HANGRUN=$W/wrap/run-hang; HMARK=$W/run-hang.mark
cat > "$HANGRUN" <<EOF
#!/bin/sh
# Restart fault: after a dispatched run returns, record its rc and hang until killed.
if [ "\$1" = run ] && [ ! -e "$HMARK" ]; then
  "$AN0_RUN" "\$@"; r=\$?
  echo "\$r" > "$HMARK"
  exec sleep 600
fi
exec "$AN0_RUN" "\$@"
EOF
chmod +x "$FLAKY" "$BADCORE" "$HANGRUN"

# ---- 3. one case ---------------------------------------------------------------
PORT=$PORTBASE
mkconf() { # out run serve core accept
  printf '%s\n' "$2" "$3" "$KEYS" "$PINS" "$WKEYS" "$WPINS" "$WSTATE" "$RECOMPUTE" \
    "$RECOMPUTE_SHA256" "127.0.0.1:$PORT" "$4" "$CKPT" "$MODE" "$VERIFIER" "$5" > "$1"
}
seen_sha() { if [ -e "$WSTATE/seen" ]; then sha256sum < "$WSTATE/seen" | cut -d' ' -f1; else echo none; fi; }

receipts() { # case dir: static receipts the eval checker reads
  c=$1
  mkdir -p "$c/keys" "$c/pins" "$c/artifacts" "$c/worker"
  cp "$RT/coordinator-public-keys.txt" "$RT/worker-public-keys.txt" "$c/keys/"
  cp "$PINS/volunteer.ed25519" "$PINS/executor.p256" "$c/pins/"
  printf 'AN0_RUN=%s\nAN0_RUN_SHA256=%s\nAN0_SERVE=%s\nAN0_SERVE_SHA256=%s\nRECOMPUTE=%s\nRECOMPUTE_SHA256=%s\nCORE_KIND=%s\nCKPT=%s\nMODE=%s\n' \
    "$AN0_RUN" "$AN0_RUN_SHA256" "$AN0_SERVE" "$AN0_SERVE_SHA256" "$RECOMPUTE" "$RECOMPUTE_SHA256" "$CORE_KIND" "$CKPT" "$MODE" > "$c/runtime.env"
  for f in "$c"/out/*.an0b "$c"/out/*.mjr1 "$c"/out/*.mjv; do [ -e "$f" ] && cp "$f" "$c/artifacts/"; done
  for f in "$c"/logs/*-serve.err; do [ -e "$f" ] && cp "$f" "$c/worker/serve-$(basename "$f")"; done
  [ -e "$WSTATE/seen" ] && cp "$WSTATE/seen" "$c/worker/seen"
  [ -d "$c/artifacts" ] && [ -z "$(ls "$c/artifacts")" ] && rmdir "$c/artifacts"
}

front() { # case dir request: English -> G (+ review English); rc of en2goal
  c=$1
  printf '%s\n' "$2" > "$c/request.en"
  "$BIN/en2goal" < "$c/request.en" > "$c/en2goal.out" 2> "$c/en2goal.err"
  r=$?; echo $r > "$c/en2goal.rc"
  [ $r -eq 0 ] || return $r
  cp "$c/en2goal.out" "$c/goal.oe45"
  "$BIN/goal2en" < "$c/goal.oe45" > "$c/goal.review.en" 2> "$c/goal2en.err"; echo $? > "$c/goal2en.rc"
  return 0
}

hive() { # case dir conf -> hive rc
  (cd "$1" && exec $HEAVY "$BIN/hive" "$2") > "$1/hive.out" 2> "$1/hive.err"
}

back() { # case dir: P + facts -> English, then the independent checker
  c=$1
  if [ -e "$c/facts.tsv" ] && [ -e "$c/episode.oe45" ]; then
    { head -n 2 "$c/episode.oe45" | tail -n 1; tail -n 1 "$c/episode.oe45"; } > "$c/rep2en.in"
    "$BIN/rep2en" "$c/facts.tsv" < "$c/rep2en.in" > "$c/rep2en.out" 2> "$c/rep2en.err"
    r=$?; echo $r > "$c/rep2en.rc"
    [ $r -eq 0 ] && cp "$c/rep2en.out" "$c/response.en"
  fi
  receipts "$c"
}

check() { # case dir -> checker rc
  python3 "$CHECK" "$1" > "$1/check.out" 2>&1; r=$?; echo $r > "$1/check.rc"; return $r
}

FAILED=""
verdict() { # case ok(0/1) detail
  if [ "$2" -eq 0 ]; then say "CASE $1 PASS $3"; else say "CASE $1 FAIL $3"; FAILED="$FAILED $1"; fi
}
rcof() { cat "$1" 2>/dev/null || echo none; }
lastline() { tail -n 1 "$1" 2>/dev/null; }
newcase() { C=$CS/$1; mkdir -p "$C"; PORT=$((PORT + 1)); }

ok_flow() { # case conf-run conf-serve conf-core accept request -> sets HR, CR
  newcase "$1"
  front "$C" "$6"; ER=$?
  mkconf "$W/conf/$1.conf" "$2" "$3" "$4" "$5"; cp "$W/conf/$1.conf" "$C/hive.conf"
  HR=none; CR=none
  if [ $ER -eq 0 ]; then hive "$C" "$W/conf/$1.conf"; HR=$?; echo $HR > "$C/hive.rc"
  else # the driver never ran: an empty facts table (header only) states "nothing dispatched"
    printf 'job\tattempt\tm\tk\tn\tseed\texit\tclass\tcode\tdag_id\tmjr1_sha256\tmjr1_outcome\tverifier\n' > "$C/facts.tsv"; fi
  back "$C"; check "$C"; CR=$?
}

# Case 1 happy-mm
REQ_MM="please multiply a 16 by 32 matrix by a 32 by 8 matrix with seed 7"
ok_flow happy-mm "$AN0_RUN" "$AN0_SERVE" "$CORE" 120 "$REQ_MM"
P=$(lastline "$C/episode.oe45")
[ $HR -eq 0 ] && [ $CR -eq 0 ] && [ "$P" = "rep ok mm ushort 0x010 ushort 0x020 ushort 0x008 ulong 0x0007 end" ]
verdict happy-mm $? "hive=$HR check=$CR P=[$P]"
HAPPY=$C

# Case 2 happy-pair
ok_flow happy-pair "$AN0_RUN" "$AN0_SERVE" "$CORE" 120 "run two matmuls of size 4 x 4 x 4 with seeds 1 and 2"
P=$(lastline "$C/episode.oe45")
[ $HR -eq 0 ] && [ $CR -eq 0 ] && [ "$P" = "rep ok pair ushort 0x004 ushort 0x004 ushort 0x004 ulong 0x0001 ulong 0x0002 end" ]
verdict happy-pair $? "hive=$HR check=$CR P=[$P]"

# Case 3 bound: core refusal, nothing dispatched, worker state unchanged
before=$(seen_sha)
ok_flow bound "$AN0_RUN" "$AN0_SERVE" "$CORE" 120 "run a matmul of size 300 x 8 x 8 with seed 5"
after=$(seen_sha)
P=$(lastline "$C/episode.oe45")
printf 'seen_sha256_before %s\nseen_sha256_after %s\n' "$before" "$after" > "$C/worker-state.txt"
[ $HR -eq 0 ] && [ $CR -eq 0 ] && [ "$P" = "rep refused bound end" ] && [ "$before" = "$after" ] && [ ! -e "$C/disp/j1.an0b" ]
verdict bound $? "hive=$HR check=$CR P=[$P] seen_before=$before seen_after=$after"

# Case 4 translator refusal: no goal, no dispatch, no English success
before=$(seen_sha)
ok_flow translator "$AN0_RUN" "$AN0_SERVE" "$CORE" 120 "make me a sandwich"
after=$(seen_sha)
ER=$(rcof "$C/en2goal.rc")
printf 'seen_sha256_before %s\nseen_sha256_after %s\n' "$before" "$after" > "$C/worker-state.txt"
[ "$ER" = 20 ] && [ "$HR" = none ] && [ ! -e "$C/goal.oe45" ] && [ ! -e "$C/response.en" ] && [ "$before" = "$after" ] \
  && [ "$CR" = 0 ] && [ -z "$(sed 1d "$C/facts.tsv")" ]
verdict translator $? "en2goal=$ER($(cat "$C/en2goal.out")) hive=$HR check=$CR facts_rows=0"

# Case 5 transport: worker down once -> res fail transport -> identical retry -> ok
touch "$SDOWN"
ok_flow transport "$AN0_RUN" "$FLAKY" "$CORE" 15 "run a matmul of size 8 x 8 x 8 with seed 11"
P=$(lastline "$C/episode.oe45")
X=$(cut -f1,2,7,8,9 "$C/facts.tsv" 2>/dev/null | tr '\t\n' ' ;')
[ $HR -eq 0 ] && [ $CR -eq 0 ] && [ ! -e "$SDOWN" ] && grep -q '^res fail transport end$' "$C/episode.oe45" \
  && [ "$P" = "rep ok mm ushort 0x008 ushort 0x008 ushort 0x008 ulong 0x000b end" ] && grep -q ' retry=1 ' "$C/dispatch.log"
verdict transport $? "hive=$HR check=$CR facts=[$X] P=[$P]"
rm -f "$SDOWN"

# Case 6 replay: resubmit happy-mm's signed bundle -> worker AN0_REPLAY -> refused replay
newcase replay
mkdir -p "$C/disp"; cp "$HAPPY/disp/j1.an0b" "$C/disp/j1.an0b"
dag=$(cut -f10 "$HAPPY/facts.tsv" | tail -n 1)
ex0=$(cat "$CS"/*/logs/*-serve.err 2>/dev/null | grep -c "executing dag_id=$dag")
front "$C" "$REQ_MM"
mkconf "$W/conf/replay.conf" "$AN0_RUN" "$AN0_SERVE" "$CORE" 120; cp "$W/conf/replay.conf" "$C/hive.conf"
hive "$C" "$W/conf/replay.conf"; HR=$?; echo $HR > "$C/hive.rc"
back "$C"; check "$C"; CR=$?
ex1=$(cat "$CS"/*/logs/*-serve.err 2>/dev/null | grep -c "executing dag_id=$dag")
dn=$(grep -c "^done $dag " "$WSTATE/seen" 2>/dev/null)
P=$(lastline "$C/episode.oe45")
printf 'dag %s\nexecutions_before %s\nexecutions_after %s\nseen_done_lines %s\n' "$dag" "$ex0" "$ex1" "$dn" > "$C/worker-state.txt"
[ $HR -eq 0 ] && [ $CR -eq 0 ] && [ "$P" = "rep refused replay end" ] && [ "$ex0" = 1 ] && [ "$ex1" = 1 ] && [ "$dn" = 1 ]
verdict replay $? "hive=$HR check=$CR P=[$P] executions=$ex0->$ex1 seen_done=$dn"

# Case 7 repair: one injected invalid action -> rejected -> grammar-guided re-decode -> executed
touch "$CBAD"
ok_flow repair "$AN0_RUN" "$AN0_SERVE" "$BADCORE" 120 "run a matmul of size 12 x 6 x 3 with seed 21"
P=$(lastline "$C/episode.oe45")
[ $HR -eq 0 ] && [ $CR -eq 0 ] && [ ! -e "$CBAD" ] && grep -q 'hive: repair' "$C/hive.log" \
  && [ "$P" = "rep ok mm ushort 0x00c ushort 0x006 ushort 0x003 ulong 0x0015 end" ]
verdict repair $? "hive=$HR check=$CR repair_logged=$(grep -c 'hive: repair' "$C/hive.log" 2>/dev/null) P=[$P]"
rm -f "$CBAD"

# Case 8 restart: kill -9 the whole driver group after dispatch, restart from the journal
newcase restart
front "$C" "run a matmul of size 32 x 16 x 8 with seed 33"
mkconf "$W/conf/restart-1.conf" "$HANGRUN" "$AN0_SERVE" "$CORE" 120; cp "$W/conf/restart-1.conf" "$C/hive-1.conf"
mkconf "$W/conf/restart-2.conf" "$AN0_RUN" "$AN0_SERVE" "$CORE" 120; cp "$W/conf/restart-2.conf" "$C/hive.conf"
(cd "$C" && exec setsid $HEAVY "$BIN/hive" "$W/conf/restart-1.conf" > "$C/hive-1.out" 2> "$C/hive-1.err") &
grp=$!
i=0; while [ ! -e "$HMARK" ] && [ $i -lt 600 ]; do sleep 1; i=$((i + 1)); done
sleep 1
kill -9 "-$grp"; KR=$?; wait $grp 2>/dev/null; K=$?
echo "kill -9 -$grp rc=$KR; driver wait rc=$K (137 = SIGKILL); aienet0-run rc before kill=$(cat "$HMARK" 2>/dev/null)" > "$C/kill.txt"
cp "$C/episode.oe45" "$C/episode.killed.oe45" 2>/dev/null
cp "$C/dispatch.log" "$C/dispatch.killed.log" 2>/dev/null
[ -e "$C/facts.tsv" ] && echo "facts.tsv present before restart" >> "$C/kill.txt"
hive "$C" "$W/conf/restart-2.conf"; HR=$?; echo $HR > "$C/hive.rc"
back "$C"; check "$C"; CR=$?
dag=$(cut -f10 "$C/facts.tsv" 2>/dev/null | tail -n 1)
ex=$(cat "$C"/logs/*-serve.err 2>/dev/null | grep -c "executing dag_id=$dag")
dn=$(grep -c "^done $dag " "$WSTATE/seen" 2>/dev/null)
P=$(lastline "$C/episode.oe45")
printf 'dag %s\nworker_executions %s\nseen_done_lines %s\n' "$dag" "$ex" "$dn" > "$C/worker-state.txt"
[ -e "$HMARK" ] && [ "$KR" = 0 ] && [ "$K" = 137 ] && [ "$(lastline "$C/episode.killed.oe45" | cut -d' ' -f1-2)" = "act job" ] \
  && grep -q 'hive: reconcile' "$C/hive.log" && [ $HR -eq 0 ] && [ $CR -eq 0 ] \
  && [ "$ex" = 1 ] && [ "$dn" = 1 ] && [ "$P" = "rep ok mm ushort 0x020 ushort 0x010 ushort 0x008 ulong 0x0021 end" ]
verdict restart $? "killed_after=$(lastline "$C/episode.killed.oe45" | cut -d' ' -f1-2) hive=$HR check=$CR executions=$ex seen_done=$dn P=[$P]"

# Case 9 tampered: negative controls on copies of happy-mm (receipt byte, English fact)
newcase tampered
cp -R "$HAPPY" "$C/receipt"; cp -R "$HAPPY" "$C/english"
m=$(ls "$C/receipt/artifacts/"*.mjr1 2>/dev/null | head -n 1)
r1=none; r2=none
if [ -n "$m" ]; then
  b=$(od -An -tu1 -j 200 -N 1 "$m" | tr -d ' ')
  printf "$(printf '\\%03o' $(( (b + 1) % 256 )))" | dd of="$m" bs=1 seek=200 conv=notrunc 2>/dev/null
  check "$C/receipt"; r1=$?
fi
if [ -e "$C/english/response.en" ]; then
  sed 's/seed 7/seed 8/' "$C/english/response.en" > "$C/english/response.en.t" && mv "$C/english/response.en.t" "$C/english/response.en"
  cmp -s "$C/english/response.en" "$HAPPY/response.en"; same=$?
  check "$C/english"; r2=$?
  [ $same -eq 1 ] || r2=same
fi
[ "$r1" = 1 ] && [ "$r2" = 1 ]
verdict tampered $? "receipt_byte200_check=$r1 [$(lastline "$C/receipt/check.out")] english_check=$r2 [$(lastline "$C/english/check.out")]"

# ---- 4. summary ----------------------------------------------------------------
if [ -z "$FAILED" ]; then
  say "OE45_DEMO_PASS core=$CORE_KIND mode=$MODE ckpt=$CKPT"
  exit 0
fi
say "OE45_DEMO_FAIL core=$CORE_KIND mode=$MODE ckpt=$CKPT failed:$FAILED"
exit 67

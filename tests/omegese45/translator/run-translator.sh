#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Translator suite for aien-train #45: builds en2goal, goal2en and rep2en
# from .om with the pinned omc (refuses any other omc), then checks the dev
# cases, the held-out set (raw counts against the committed file, then with
# the dated errata in heldout-errata.tsv), every refusal code, boundaries,
# goal2en/en2goal stability, rep2en per outcome class, contradictions and
# the interface v1.2 facts rules, cost, and negative controls: a k<->n
# en2goal mutant and one rep2en mutant per facts check (each must be caught).
# Loads no model or checkpoint. Shell only builds, runs and compares bytes.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OMC=${OMC:-/tmp/om-main/seed/out/omc}
PIN=62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b
T=$ROOT/native/omegese45/translate
C=$ROOT/native/omegese45
D=$ROOT/tests/omegese45/translator
W=${WORK:-$(mktemp -d)}
RUN="taskset -c 10-14 nice -n 19 ionice -c3"
FAIL=0
fail() { echo "FAIL $*"; FAIL=1; }

got=$(sha256sum "$OMC" 2>/dev/null | cut -d' ' -f1)
echo "omc sha256 $got"
[ "$got" = "$PIN" ] || { echo "OMC_PIN_MISMATCH want $PIN ($OMC)"; echo OE45_TRANSLATOR_FAIL; exit 1; }

build() { # name units...
  n=$1; shift
  cat "$@" > "$W/$n.unit.om" || exit 1
  $RUN "$OMC" < "$W/$n.unit.om" > "$W/$n" 2> "$W/$n.err" || { cat "$W/$n.err"; echo "BUILD_FAIL $n"; exit 1; }
  [ "$(od -An -c -N4 "$W/$n" | tr -d ' ')" = '177ELF' ] || { echo "BUILD_FAIL $n not ELF"; exit 1; }
  chmod +x "$W/$n"
}
REPU="$T/otio.om $C/vocab.om $C/msg.om $T/otmsg.om"
build en2goal "$T/otio.om" "$T/en2goal.om" "$T/otfrag.om" "$T/main-en2goal.om"
build goal2en $REPU "$T/goal2en.om" "$T/otfrag.om" "$T/main-goal2en.om"
build rep2en $REPU "$T/rep2en.om" "$T/otfrag.om" "$T/main-rep2en.om"
# Mutant: k and n swapped in the emitted G (negative control).
sed -e 's/^kv = get otrd 3$/kv = get otrd 4/' -e 's/^nv = get otrd 4$/nv = get otrd 3/' "$T/en2goal.om" > "$W/en2goal-mut.om"
cmp -s "$T/en2goal.om" "$W/en2goal-mut.om" && { echo "MUTANT_NOT_APPLIED"; exit 1; }
build en2goal-mut "$T/otio.om" "$W/en2goal-mut.om" "$T/otfrag.om" "$T/main-en2goal.om"

# score BIN FILE -> "exact refused_differently wrong total"
score() {
  ex=0; rd=0; wr=0; tot=0
  TAB=$(printf '\t')
  while IFS="$TAB" read -r exp req; do
    tot=$((tot+1))
    out=$(printf '%s\n' "$req" | "$1"); rc=$?
    case $exp in
      refuse*) want=${exp#refuse }
        if [ "$rc" = "$want" ]; then ex=$((ex+1)); elif [ "$rc" = 0 ]; then wr=$((wr+1)); echo "  wrong: [$req] -> $out" >&2; else rd=$((rd+1)); echo "  refused-differently: [$req] rc=$rc want=$want" >&2; fi ;;
      *) if [ "$rc" = 0 ] && [ "$out" = "$exp" ]; then ex=$((ex+1)); elif [ "$rc" = 0 ]; then wr=$((wr+1)); echo "  wrong: [$req] -> $out" >&2; else rd=$((rd+1)); echo "  refused-differently: [$req] rc=$rc" >&2; fi ;;
    esac
  done < "$2"
  echo "$ex $rd $wr $tot"
}

set -- $(score "$W/en2goal" "$D/cases.tsv")
echo "DEV exact=$1 refused_differently=$2 wrong=$3 total=$4"
[ "$1" = "$4" ] || fail "dev cases must all be exact"

# Held-out. The primary number is raw, against the committed heldout.tsv
# (faeb7ad, never rewritten). heldout-errata.tsv lists dated corrections
# where the interface changed after the file was written; each erratum's
# original must equal the committed expectation. PASS: raw wrong is at most
# the number of errata rows and the corrected file has wrong 0.
set -- $(score "$W/en2goal" "$D/heldout.tsv")
echo "HELDOUT exact=$1 refused_differently=$2 wrong=$3 total=$4 (raw, committed heldout.tsv)"
rawwr=$3
ner=$(($(wc -l < "$D/heldout-errata.tsv") - 1))
awk -F'\t' -v OFS='\t' '
  NR == FNR { if (FNR > 1) { want[$1] = $2; fix[$1] = $3 } next }
  (FNR in fix) { if ($1 != want[FNR]) bad = 1; $1 = fix[FNR]; n++ }
  { print }
  END { if (bad) exit 3; print n > "/dev/stderr" }' \
  "$D/heldout-errata.tsv" "$D/heldout.tsv" > "$W/heldout-corrected.tsv" 2> "$W/errata.n" \
  || fail "an erratum's original does not match heldout.tsv"
[ "$(cat "$W/errata.n")" = "$ner" ] || fail "errata applied $(cat "$W/errata.n") of $ner"
tail -n +2 "$D/heldout-errata.tsv" | while IFS=$(printf '\t') read -r row orig corr dt why; do
  echo "ERRATUM row=$row original=[$orig] corrected=[$corr] date=$dt reason: $why"
done
set -- $(score "$W/en2goal" "$W/heldout-corrected.tsv" 2>/dev/null)
echo "HELDOUT_CORRECTED exact=$1 refused_differently=$2 wrong=$3 total=$4 (errata=$ner)"
[ "$3" = 0 ] || fail "held-out wrong with errata must be 0"
[ "$rawwr" -le "$ner" ] || fail "raw held-out wrong $rawwr exceeds errata $ner"

# Over-long input (513 bytes) -> 20.
long=$(printf '%0513d' 0)
printf '%s\n' "$long" | "$W/en2goal" > /dev/null; rc=$?
[ "$rc" = 20 ] && echo "REFUSAL long-input rc=20" || fail "long input rc=$rc"
for c in 20 21 22 23; do grep -q "^refuse $c" "$D/cases.tsv" || fail "no dev case for $c"; done

# Stability: goal2en of every accepted G re-parses to the same G.
st=0
for f in "$D/cases.tsv" "$W/heldout-corrected.tsv"; do
  cut -f1 "$f" | grep '^goal' | sort -u > "$W/goals"
  while read -r g; do
    s=$(printf '%s\n' "$g" | "$W/goal2en") || fail "goal2en rc on [$g]"
    g2=$(printf '%s\n' "$s" | "$W/en2goal")
    [ "$g2" = "$g" ] && st=$((st+1)) || fail "unstable [$g] -> [$s] -> [$g2]"
  done < "$W/goals"
done
echo "STABLE count=$st"
out=$(printf 'goal mm ushort 0x04 end\n' | "$W/goal2en"); rc=$?
[ "$rc" = 31 ] && echo "GOAL2EN invalid rc=31 [$out]" || fail "goal2en invalid rc=$rc"
want='Goal for review: one matmul_u32 job multiplying a 4 by 8 matrix by a 8 by 2 matrix with seed 7.'
out=$(printf 'goal mm ushort 0x004 ushort 0x008 ushort 0x002 ulong 0x0007 end\n' | "$W/goal2en")
[ "$out" = "$want" ] || fail "goal2en template [$out]"

# rep2en per class, contradictions and facts rules (interface v1.2 section 5).
TAB=$(printf '\t')
H="job${TAB}attempt${TAB}m${TAB}k${TAB}n${TAB}seed${TAB}exit${TAB}class${TAB}code${TAB}dag_id${TAB}mjr1_sha256${TAB}mjr1_outcome${TAB}verifier"
HX="job${TAB}attempt${TAB}m${TAB}k${TAB}n${TAB}seed${TAB}exit${TAB}class${TAB}code${TAB}dag_id${TAB}mjr1_sha256${TAB}mjr1_outcomx${TAB}verifier"
A=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
B=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
V=02$A
V3=03$B
G1='goal mm ushort 0x004 ushort 0x008 ushort 0x002 ulong 0x0007 end'
G2='goal pair ushort 0x004 ushort 0x008 ushort 0x002 ulong 0x0007 ulong 0x0008 end'
GB='goal mm ushort 0x12c ushort 0x002 ushort 0x002 ulong 0x0001 end'
P1='rep ok mm ushort 0x004 ushort 0x008 ushort 0x002 ulong 0x0007 end'
P2='rep ok pair ushort 0x004 ushort 0x008 ushort 0x002 ulong 0x0007 ulong 0x0008 end'
PFT='rep fail transport end'
# row JOB ATTEMPT SEED EXIT CLASS CODE DAG SHA OUTCOME VERIFIER (dims 4 8 2)
row() { printf '%s\t%s\t4\t8\t2\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "$@"; }
OK1=$(row 1 1 7 0 ok - $A $B 1 $V)
OK1R=$(row 1 2 7 0 ok - $A $B 1 $V)
OK2=$(row 2 1 8 0 ok - $B $A 1 $V3)
TR1=$(row 1 1 7 10 fail transport - - - -)
TR1R=$(row 1 2 7 10 fail transport - - - -)
RP1=$(row 1 1 7 11 refused replay $A - - -)
HDR=$H RB=$W/rep2en MODE=check MISS=0 FIRST=
rep() { # name want_rc G P facts-rows...
  name=$1 want=$2 g=$3 p=$4; shift 4
  { [ "$HDR" = NONE ] || echo "$HDR"; for r in "$@"; do echo "$r"; done; } > "$W/f.$name.tsv"
  out=$(printf '%s\n%s\n' "$g" "$p" | "$RB" "$W/f.$name.tsv"); rc=$?
  LAST=$out
  if [ "$MODE" = check ]; then
    echo "REP2EN $name rc=$rc $out"
    [ "$rc" = "$want" ] || fail "rep2en $name rc=$rc want $want"
  elif [ "$rc" != "$want" ]; then
    MISS=$((MISS+1)); [ -n "$FIRST" ] || FIRST="$name(rc=$rc,want=$want)"
  fi
}
tpl() { # name expected-sentence
  [ "$LAST" = "$2" ] && return
  if [ "$MODE" = check ]; then fail "$1 template [$LAST]"; else MISS=$((MISS+1)); [ -n "$FIRST" ] || FIRST="$1(template)"; fi
}
repall() {
  # English cases (exit 0).
  rep ok-mm 0 "$G1" "$P1" "$OK1"
  tpl ok-mm "Completed and verified: one matmul_u32 job multiplying a 4 by 8 matrix by a 8 by 2 matrix with seed 7; job 1 dag_id $A mjr1_sha256 $B."
  rep ok-pair 0 "$G2" "$P2" "$OK1" "$OK2"
  rep retry-ok 0 "$G1" "$P1" "$TR1" "$OK1R"
  rep retry-fail 0 "$G1" "$PFT" "$TR1" "$TR1R"
  rep bound 0 "$GB" 'rep refused bound end'
  rep refused-replay 0 "$G1" 'rep refused replay end' "$RP1"
  tpl refused-replay "Not completed: job 1 was refused (code replay): the worker refused a replay of an already executed job."
  rep fail-transport 0 "$G1" "$PFT" "$TR1"
  rep pair-job2-refused 0 "$G2" 'rep refused budget end' "$OK1" "$(row 2 1 8 11 refused budget - - - -)"
  tpl pair-job2-refused "Not completed: job 2 was refused (code budget): the job was refused for budget or fuel."
  rep fail-receipt-digest 0 "$G1" 'rep fail verdict end' "$(row 1 1 7 14 fail verdict $A $B - -)"
  # Contradictions (exit 30).
  rep x-ok-vs-fail 30 "$G1" "$P1" "$TR1"
  rep x-operands 30 "$G1" 'rep ok mm ushort 0x004 ushort 0x002 ushort 0x008 ulong 0x0007 end' "$OK1"
  rep x-code 30 "$G1" 'rep fail budget end' "$TR1"
  rep x-class 30 "$G1" 'rep refused transport end' "$TR1"
  rep x-fail-vs-ok 30 "$G1" "$PFT" "$OK1"
  rep x-retry-unreported 30 "$G1" "$PFT" "$TR1" "$OK1R"
  rep x-pair-missing 30 "$G2" "$P2" "$OK1"
  rep x-bound-dispatched 30 "$GB" 'rep refused bound end' "1${TAB}1${TAB}300${TAB}2${TAB}2${TAB}1${TAB}0${TAB}ok${TAB}-${TAB}$A${TAB}$B${TAB}1${TAB}$V"
  tpl x-bound-dispatched "G2E_CONTRADICT bound"
  rep x-facts-operands 30 "$G1" "$P1" "1${TAB}1${TAB}4${TAB}2${TAB}8${TAB}7${TAB}0${TAB}ok${TAB}-${TAB}$A${TAB}$B${TAB}1${TAB}$V"
  rep x-nothing 30 "$G1" "$PFT"
  # Invalid inputs (exit 31).
  rep bad-report 31 "$G1" 'rep ok end'
  rep bad-goal 31 'goal end' "$P1" "$OK1"
  rep bad-facts 31 "$G1" "$P1" "$(row 1 1 7 0 ok - - - - -)"
  # v1.2 header line.
  HDR=NONE; rep i-no-header 31 "$G1" "$P1" "$OK1"
  HDR=$HX; rep i-header-wrong 31 "$G1" "$P1" "$OK1"; HDR=$H
  # v1.2 ok row: mjr1_outcome 1, verifier 66 lowercase hex starting 02/03 (T1).
  rep i-ok-outcome-zzz 31 "$G1" "$P1" "$(row 1 1 7 0 ok - $A $B zzz nonsense)"
  rep i-ok-outcome-2 31 "$G1" "$P1" "$(row 1 1 7 0 ok - $A $B 2 $V)"
  rep i-ok-verifier-64 31 "$G1" "$P1" "$(row 1 1 7 0 ok - $A $B 1 $A)"
  rep i-ok-verifier-04 31 "$G1" "$P1" "$(row 1 1 7 0 ok - $A $B 1 04$A)"
  rep i-ok-verifier-upper 31 "$G1" "$P1" "$(row 1 1 7 0 ok - $A $B 1 02$(printf %s "$A" | tr a A))"
  rep i-nonok-outcome 31 "$G1" "$PFT" "$(row 1 1 7 10 fail transport - - 3 -)"
  rep i-nonok-verifier 31 "$G1" "$PFT" "$(row 1 1 7 10 fail transport - - - $V)"
  rep i-nonok-dag 31 "$G1" "$PFT" "$(row 1 1 7 10 fail transport zz - - -)"
  # v1.2 exit <-> class/code map (T2).
  rep i-exit14-transport 31 "$G1" "$PFT" "$(row 1 1 7 14 fail transport - - - -)"
  rep i-exit3-transport 31 "$G1" "$PFT" "$(row 1 1 7 3 fail transport - - - -)"
  rep i-exit4-replay 31 "$G1" 'rep refused replay end' "$(row 1 1 7 4 refused replay - - - -)"
  rep i-exit10-replay 31 "$G1" 'rep refused replay end' "$(row 1 1 7 10 refused replay - - - -)"
  rep i-exit11-receipt 31 "$G1" 'rep fail receipt end' "$(row 1 1 7 11 fail receipt - - - -)"
  rep i-refused-transport 31 "$G1" 'rep refused transport end' "$(row 1 1 7 10 refused transport - - - -)"
  rep i-fail-replay 31 "$G1" 'rep fail replay end' "$(row 1 1 7 11 fail replay - - - -)"
  # v1.2 history (T4).
  rep i-ok-then-transport 31 "$G1" "$PFT" "$OK1" "$TR1R"
  rep i-ok-then-transport-a1 31 "$G1" "$PFT" "$OK1" "$TR1"
  rep i-refused-then-job2 31 "$G2" "$PFT" "$(row 1 1 7 11 refused worker - - - -)" "$(row 2 1 8 10 fail transport - - - -)"
  rep i-transport-then-job2 31 "$G2" "$P2" "$TR1" "$OK2"
  rep i-job2-first 31 "$G2" "$P2" "$OK2" "$OK1"
  rep i-attempt0 31 "$G1" "$P1" "$(row 1 0 7 0 ok - $A $B 1 $V)"
  rep i-attempt2-first 31 "$G1" "$P1" "$OK1R"
  rep i-attempt-repeat 31 "$G1" "$P1" "$TR1" "$OK1"
  rep i-attempt3 31 "$G1" "$P1" "$TR1" "$TR1R" "$(row 1 3 7 0 ok - $A $B 1 $V)"
  rep i-retry-after-receipt 31 "$G1" "$P1" "$(row 1 1 7 12 fail receipt - - - -)" "$OK1R"
}
repall

# Cost on GB10 Spark Grace CPU, cores 10-14, nice 19, ionice idle.
for t in en2goal goal2en rep2en; do
  case $t in
    en2goal) printf 'run two matmuls of size 4 x 8 x 2 with seeds 7 and 8\n' > "$W/in.$t"; args= ;;
    goal2en) printf '%s\n' "$G2" > "$W/in.$t"; args= ;;
    rep2en) printf '%s\n%s\n' "$G2" "$P2" > "$W/in.$t"; args=$W/f.ok-pair.tsv ;;
  esac
  $RUN /usr/bin/time -v "$W/$t" $args < "$W/in.$t" > /dev/null 2> "$W/time.$t"
  echo "COST $t wall=$(grep 'Elapsed (wall' "$W/time.$t" | sed 's/.*): //') maxrss_kb=$(grep 'Maximum resident' "$W/time.$t" | sed 's/.*: //') chip=GB10-Spark-Grace-CPU cores=10-14 nice=19"
done

# Negative control: the k<->n mutant must fail the dev cases.
set -- $(score "$W/en2goal-mut" "$D/cases.tsv" 2>/dev/null)
echo "MUTANT exact=$1 refused_differently=$2 wrong=$3 total=$4"
[ "$3" -gt 0 ] && echo "MUTANT_CAUGHT" || fail "k<->n mutant not caught"

# Negative controls for rep2en: each facts check is disabled in turn (its
# condition line, which must occur exactly once, becomes `if false`); the
# fixtures above must then give at least one unexpected result.
while IFS='|' read -r mname mline; do
  [ -n "$mname" ] || continue
  n=$(grep -cxF "$mline" "$T/rep2en.om")
  [ "$n" = 1 ] || { fail "mutant $mname line occurs $n times"; continue; }
  awk -v L="$mline" '$0 == L { print "if false"; next } { print }' "$T/rep2en.om" > "$W/rep2en-$mname.om"
  cmp -s "$T/rep2en.om" "$W/rep2en-$mname.om" && { fail "mutant $mname not applied"; continue; }
  build "rep2en-$mname" $REPU "$W/rep2en-$mname.om" "$T/otfrag.om" "$T/main-rep2en.om"
  RB=$W/rep2en-$mname MODE=$mname MISS=0 FIRST=
  repall
  if [ "$MISS" -gt 0 ]; then echo "REPMUTANT $mname missed=$MISS first=$FIRST CAUGHT"; else fail "rep2en mutant $mname not caught"; fi
done <<'EOF'
header|if not othdrok ulong fl
ok-outcome|if not otfone r 11
ok-verifier|if not otfkey r 12
nonok-dashes|if not and otfdash r 11 otfdash r 12
nonok-dag-sha|if not and otfhod r 9 otfhod r 10
code-class|if ne cl wc
code-exit|if ne ex wx
hist-same-job|if not and eq a 2 pt
hist-next-job|if not and eq j add pj 1 eq a 1
hist-prev-ok|if and gt pj 0 not pk
EOF

if [ "$FAIL" = 0 ]; then echo OE45_TRANSLATOR_PASS; exit 0; fi
echo OE45_TRANSLATOR_FAIL; exit 1

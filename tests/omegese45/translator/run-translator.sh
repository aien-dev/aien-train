#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Translator suite for aien-train #45: builds en2goal, goal2en and rep2en
# from .om with the pinned omc, then checks the dev cases, the held-out set
# (counts), every refusal code, boundaries, goal2en/en2goal stability, rep2en
# per outcome class and contradictions, cost, and a k<->n mutant (must fail).
# Loads no model or checkpoint. Shell only builds, runs and compares bytes.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OMC=${OMC:-/tmp/om-main/seed/out/omc}
T=$ROOT/native/omegese45/translate
C=$ROOT/native/omegese45
D=$ROOT/tests/omegese45/translator
W=${WORK:-$(mktemp -d)}
RUN="taskset -c 10-14 nice -n 19"
FAIL=0
fail() { echo "FAIL $*"; FAIL=1; }

build() { # name units...
  n=$1; shift
  cat "$@" > "$W/$n.unit.om" || exit 1
  $RUN "$OMC" < "$W/$n.unit.om" > "$W/$n" 2> "$W/$n.err" || { cat "$W/$n.err"; echo "BUILD_FAIL $n"; exit 1; }
  [ "$(od -An -c -N4 "$W/$n" | tr -d ' ')" = '177ELF' ] || { echo "BUILD_FAIL $n not ELF"; exit 1; }
  chmod +x "$W/$n"
}
echo "omc sha256 $(sha256sum "$OMC" | cut -d' ' -f1)"
build en2goal "$T/otio.om" "$T/en2goal.om" "$T/otfrag.om" "$T/main-en2goal.om"
build goal2en "$T/otio.om" "$C/vocab.om" "$C/msg.om" "$T/otmsg.om" "$T/goal2en.om" "$T/otfrag.om" "$T/main-goal2en.om"
build rep2en "$T/otio.om" "$C/vocab.om" "$C/msg.om" "$T/otmsg.om" "$T/rep2en.om" "$T/otfrag.om" "$T/main-rep2en.om"
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
set -- $(score "$W/en2goal" "$D/heldout.tsv")
echo "HELDOUT exact=$1 refused_differently=$2 wrong=$3 total=$4"
[ "$3" = 0 ] || fail "held-out wrong must be 0"
# Over-long input (513 bytes) -> 20.
long=$(printf '%0513d' 0)
printf '%s\n' "$long" | "$W/en2goal" > /dev/null; rc=$?
[ "$rc" = 20 ] && echo "REFUSAL long-input rc=20" || fail "long input rc=$rc"
for c in 20 21 22 23; do grep -q "^refuse $c" "$D/cases.tsv" || fail "no dev case for $c"; done

# Stability: goal2en of every accepted G re-parses to the same G.
st=0
for f in "$D/cases.tsv" "$D/heldout.tsv"; do
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

# rep2en per class and contradictions.
H='job	attempt	m	k	n	seed	exit	class	code	dag_id	mjr1_sha256	mjr1_outcome	verifier'
A=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
B=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
G1='goal mm ushort 0x004 ushort 0x008 ushort 0x002 ulong 0x0007 end'
G2='goal pair ushort 0x004 ushort 0x008 ushort 0x002 ulong 0x0007 ulong 0x0008 end'
GB='goal mm ushort 0x12c ushort 0x002 ushort 0x002 ulong 0x0001 end'
P1='rep ok mm ushort 0x004 ushort 0x008 ushort 0x002 ulong 0x0007 end'
P2='rep ok pair ushort 0x004 ushort 0x008 ushort 0x002 ulong 0x0007 ulong 0x0008 end'
OK1="1	1	4	8	2	7	0	ok	-	$A	$B	1	$A"
OK2="2	1	4	8	2	8	0	ok	-	$B	$A	1	$A"
TR1="1	1	4	8	2	7	3	fail	transport	-	-	-	-"
RP1="1	1	4	8	2	7	4	refused	replay	-	-	-	-"
rep() { # name want_rc G P facts-rows...
  name=$1 want=$2 g=$3 p=$4; shift 4
  { echo "$H"; for r in "$@"; do echo "$r"; done; } > "$W/f.$name.tsv"
  out=$(printf '%s\n%s\n' "$g" "$p" | "$W/rep2en" "$W/f.$name.tsv"); rc=$?
  echo "REP2EN $name rc=$rc $out"
  [ "$rc" = "$want" ] || fail "rep2en $name rc=$rc want $want"
  LAST=$out
}
rep ok-mm 0 "$G1" "$P1" "$OK1"
[ "$LAST" = "Completed and verified: one matmul_u32 job multiplying a 4 by 8 matrix by a 8 by 2 matrix with seed 7; job 1 dag_id $A mjr1_sha256 $B." ] || fail "ok-mm template"
rep ok-pair 0 "$G2" "$P2" "$OK1" "$OK2"
rep retry-ok 0 "$G1" "$P1" "$TR1" "$OK1"
rep bound 0 "$GB" 'rep refused bound end'
rep refused-replay 0 "$G1" 'rep refused replay end' "$RP1"
[ "$LAST" = "Not completed: job 1 was refused (code replay): the worker refused a replay of an already executed job." ] || fail "refused template"
rep fail-transport 0 "$G1" 'rep fail transport end' "$TR1"
rep x-ok-vs-fail 30 "$G1" "$P1" "$TR1"
rep x-operands 30 "$G1" 'rep ok mm ushort 0x004 ushort 0x002 ushort 0x008 ulong 0x0007 end' "$OK1"
rep x-code 30 "$G1" 'rep fail budget end' "$TR1"
rep x-class 30 "$G1" 'rep refused transport end' "$TR1"
rep x-fail-vs-ok 30 "$G1" 'rep fail transport end' "$OK1"
rep x-pair-missing 30 "$G2" "$P2" "$OK1"
rep x-bound-dispatched 30 "$GB" 'rep refused bound end' "$OK1"
rep x-facts-operands 30 "$G1" "$P1" "1	1	4	2	8	7	0	ok	-	$A	$B	1	$A"
rep x-nothing 30 "$G1" 'rep fail transport end'
rep bad-report 31 "$G1" 'rep ok end'
rep bad-goal 31 'goal end' "$P1" "$OK1"
rep bad-facts 31 "$G1" "$P1" "1	1	4	8	2	7	0	ok	-	-	-	-	-"

# Cost on GB10 Spark Grace CPU, cores 10-14, nice 19.
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

if [ "$FAIL" = 0 ]; then echo OE45_TRANSLATOR_PASS; exit 0; fi
echo OE45_TRANSLATOR_FAIL; exit 1

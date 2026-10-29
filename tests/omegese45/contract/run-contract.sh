#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# omegese45 contract suite (aien-train #45): R45 codec, message DFA, validator,
# binding check, lowering and MJR1 parser, plus a negative control.
#   OMC=<omc at omega-mv bf54b19f> OUT=<scratch dir> sh tests/omegese45/contract/run-contract.sh
# Host glue only (concatenation, sed/awk mutation, byte edits, comparisons);
# every check is answered by the .om driver tests/omegese45/contract/main.om.
# Each check is logged to $OUT/run/<unit>.checks. The good unit must pass every
# check; each one-line mutant (DFA, binding, codec, MJR1) must fail first at its
# named check, or the suite fails.
# Last line OE45_CONTRACT_PASS (exit 0) or OE45_CONTRACT_FAIL <check> (exit 1).
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
: "${OMC:?set OMC to the pinned omc (omega-mv bf54b19f seed/out/omc)}"
: "${OUT:?set OUT to a scratch directory}"
OMC_SHA=62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b
RCPT=$ROOT/tests/omegese45/contract/fixtures/aadc3775d32303b9e697c1c092b5628e9333f012a2ac4c1390f04da113c494cb.mjr1
RCPT_SHA=3306d180de1885a2759b324618d11eca523ef4d7aa22c4144813d71187014855
TSV=$ROOT/docs/omegese45/vocab.tsv
V=$ROOT/native/omegese45
mkdir -p "$OUT/run" "$OUT/work" "$OUT/mutants"
OUT=$(cd "$OUT" && pwd)
W=$OUT/work
NL='
'
fail() { echo "OE45_CONTRACT_FAIL $1"; exit 1; }

have=$(sha256sum "$OMC" | cut -d' ' -f1)
echo "$have  omc" > "$OUT/run/omc.sha256"
[ "$have" = "$OMC_SHA" ] || fail "compiler-sha256 $have"
[ "$(sha256sum "$RCPT" | cut -d' ' -f1)" = "$RCPT_SHA" ] || fail fixture-sha256

build() { # $1 unit.om  $2 out.elf  $3 tag
    "$OMC" < "$1" > "$2" 2> "$OUT/run/$3.compile.stderr"
    echo $? > "$OUT/run/$3.compile.rc"
    [ "$(cat "$OUT/run/$3.compile.rc")" = 0 ] || return 1
    [ "$(od -An -tx1 -N4 "$2" | tr -d ' ')" = 7f454c46 ] || return 1
    chmod 700 "$2"
}

# q MODE PAYLOAD: run the driver $E; sets RC and O (stdout).
q() {
    printf '%s\n%s\n' "$1" "$2" | "$E" > "$W/o" 2> "$W/e"
    RC=$?
    O=$(cat "$W/o")
}
# ex TAG MODE PAYLOAD RC STDOUT: one exact check.
ex() {
    q "$2" "$3"
    echo "$1 rc=$RC out=$O" >> "$LOG"
    [ "$RC" = "$4" ] && [ "$O" = "$5" ] && return 0
    echo "$1 (want rc=$4 out=$5)" > "$W/why"
    return 1
}
bad() { echo "$1" > "$W/why"; return 1; }
# ids LINE: the encoded ids of a valid line (empty on error).
ids() { q E "$1"; [ "$RC" = 0 ] || { echo; return; }; echo "${O#IDS }"; }
nth() { echo "$1" | cut -d' ' -f"$2"; }
cnt() { echo "$1" | wc -w | tr -d ' '; }
bit() { echo "$1" | cut -c"$2"; }

# valid TAG KIND IDCOUNT LINE: VALID, id count, E/D round trip, prefix masks.
valid() {
    ex "$1" V "$4" 0 "VALID $2" || return 1
    vi=$(ids "$4")
    [ "$(cnt "$vi")" = "$3" ] || bad "$1-idcount $(cnt "$vi")" || return 1
    ex "$1-roundtrip" D "$vi" 0 "$4" || return 1
    prefix "$1" "$2" "$vi" "${5:-}"
}

# prefix TAG KIND IDS [full]: after every proper prefix the next id is legal and
# FINAL is 0; after the whole message the mask is empty and FINAL is 1. With
# "full", every id's mask bit must agree with V on prefix+id (bit 1 iff V does
# not report grammar there), so N and V are the same automaton.
prefix() {
    n=$(cnt "$3")
    i=0
    pre=
    while [ "$i" -le "$n" ]; do
        q N "$2$NL$pre"
        [ "$RC" = 0 ] || bad "$1-mask-rc$i" || return 1
        mask=$(echo "$O" | sed -n 's/^MASK //p')
        fin=$(echo "$O" | sed -n 's/^FINAL //p')
        echo "$1-mask$i $mask $fin" >> "$LOG"
        if [ "$i" -lt "$n" ]; then
            nx=$(nth "$3" $((i + 1)))
            [ "$(bit "$mask" $((nx + 1)))" = 1 ] || bad "$1-prefix-mask$i" || return 1
            [ "$fin" = 0 ] || bad "$1-early-final$i" || return 1
            if [ "${4:-}" = full ] && [ "$i" -gt 0 ]; then
                id=0
                while [ "$id" -lt 41 ]; do
                    q D "$pre $id"
                    b=$(bit "$mask" $((id + 1)))
                    if [ "$RC" != 0 ]; then
                        [ "$b" = 0 ] || bad "$1-legal-undecodable$i-$id" || return 1
                    else
                        q V "$O"
                        case $O in
                            *" grammar") g=0 ;;
                            *) g=1 ;;
                        esac
                        [ "$b" = "$g" ] || bad "$1-mask-vs-V$i-$id" || return 1
                    fi
                    id=$((id + 1))
                done
            fi
            if [ -z "$pre" ]; then pre=$nx; else pre="$pre $nx"; fi
        else
            [ "$mask" = 00000000000000000000000000000000000000000 ] || bad "$1-mask-after-end" || return 1
            [ "$fin" = 1 ] || bad "$1-final-at-end" || return 1
        fi
        i=$((i + 1))
    done
}

D() { echo "ushort 0x$1"; }
S() { echo "ulong 0x$1"; }

suite() { # $1 tag; uses $E
    LOG=$OUT/run/$1.checks
    : > "$LOG"
    # Table: T is byte-identical to vocab.tsv.
    printf 'T\n' | "$E" > "$W/T.out" 2> /dev/null || bad T-rc || return 1
    cmp -s "$W/T.out" "$TSV" || bad T-cmp || return 1
    [ "$(wc -l < "$TSV" | tr -d ' ')" = 41 ] || bad tsv-rows || return 1
    # Round trip of every id (digits after 0x).
    id=0
    while [ "$id" -lt 41 ]; do
        w=$(awk -F'\t' -v i="$id" '$1==i{print $2}' "$TSV")
        if [ "$id" -ge 25 ]; then
            ex "rt$id" D "24 $id" 0 "0x$w" || return 1
            ex "rt$id-e" E "0x$w" 0 "IDS 24 $id" || return 1
        else
            ex "rt$id" D "$id" 0 "$w" || return 1
            ex "rt$id-e" E "$w" 0 "IDS $id" || return 1
        fi
        id=$((id + 1))
    done
    ex rt-hexword E "0x0123456789abcdef" 0 "IDS 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40" || return 1
    ex rej-dec-badid D "41" 40 "INVALID 0 badid" || return 1
    ex rej-dec-digit D "2 25" 40 "INVALID 1 digit" || return 1
    ex rej-dec-empty D "" 40 "INVALID 0 empty" || return 1
    # Binding check.
    G1="goal mm $(D 040) $(D 040) $(D 040) $(S 0001) end"
    GP="goal pair $(D 040) $(D 020) $(D 010) $(S 0001) $(S 0002) end"
    A1="act job mm $(D 040) $(D 020) $(D 010) $(S 0001) end"
    A2="act job mm $(D 040) $(D 020) $(D 010) $(S 0002) end"
    ex bind-ok1 B "$GP${NL}1$NL$A1" 0 BIND_OK || return 1
    ex bind-ok2 B "$GP${NL}2$NL$A2" 0 BIND_OK || return 1
    ex bind-seed B "$GP${NL}1$NL$A2" 41 "BIND_MISMATCH seed" || return 1
    ex bind-jobidx0 B "$GP${NL}0$NL$A1" 41 "BIND_MISMATCH jobindex" || return 1
    ex bind-jobidx3 B "$GP${NL}3$NL$A1" 41 "BIND_MISMATCH jobindex" || return 1
    ex bind-mm-job2 B "$G1${NL}2${NL}act job mm $(D 040) $(D 040) $(D 040) $(S 0001) end" 41 "BIND_MISMATCH jobindex" || return 1
    ex bind-dims B "$GP${NL}1${NL}act job mm $(D 040) $(D 021) $(D 010) $(S 0001) end" 41 "BIND_MISMATCH dims" || return 1
    ex bind-notjob B "$GP${NL}1${NL}act refuse bound end" 41 "BIND_MISMATCH notjob" || return 1
    GB="goal mm $(D 101) $(D 001) $(D 100) $(S ffff) end"
    ex bind-bound-101 B "$GB${NL}1${NL}act job mm $(D 101) $(D 001) $(D 100) $(S ffff) end" 42 BIND_BOUND || return 1
    GZ="goal mm $(D 001) $(D 000) $(D 100) $(S 0000) end"
    ex bind-bound-000 B "$GZ${NL}1${NL}act job mm $(D 001) $(D 000) $(D 100) $(S 0000) end" 42 BIND_BOUND || return 1
    GE="goal mm $(D 001) $(D 100) $(D 001) $(S 0000) end"
    ex bind-edges-ok B "$GE${NL}1${NL}act job mm $(D 001) $(D 100) $(D 001) $(S 0000) end" 0 BIND_OK || return 1
    ex bind-mismatch-first B "$GB${NL}1${NL}act job mm $(D 101) $(D 001) $(D 100) $(S fffe) end" 41 "BIND_MISMATCH seed" || return 1
    ex bind-invalid-A B "$GP${NL}1${NL}act job mm $(D 040) end" 40 "INVALID 24 grammar" || return 1
    ex bind-invalid-G B "res ok end${NL}1$NL$A1" 40 "INVALID 0 grammar" || return 1
    # Lowering to decimal m k n seed.
    ex lower-1 L "act job mm $(D 040) $(D 100) $(D 001) $(S ffff) end" 0 "LOWER 64 256 1 65535" || return 1
    ex lower-2 L "act job mm $(D fff) $(D 0a0) $(D 000) $(S 0000) end" 0 "LOWER 4095 160 0 0" || return 1
    ex lower-refuse L "act refuse bound end" 41 "BIND_MISMATCH notjob" || return 1
    ex lower-notA L "$G1" 40 "INVALID 0 grammar" || return 1
    # MJR1: a real receipt (aien-protocols-mv e705fdc9 evidence/aienet0-20261028).
    dag=$(od -An -tx1 -j8 -N32 "$RCPT" | tr -d ' \n')
    job=$(od -An -tx1 -j40 -N16 "$RCPT" | tr -d ' \n')
    dig=$(od -An -tx1 -j144 -N32 "$RCPT" | tr -d ' \n')
    [ "$dag" = aadc3775d32303b9e697c1c092b5628e9333f012a2ac4c1390f04da113c494cb ] || bad mjr1-oracle || return 1
    ex mjr1-real J "$RCPT" 0 "MJR1 completed 1 1 $dag $job $dig" || return 1
    head -c 403 "$RCPT" > "$W/r403"
    ex mjr1-403 J "$W/r403" 43 "MJR1_INVALID 403 length" || return 1
    { cat "$RCPT"; printf x; } > "$W/r405"
    ex mjr1-405 J "$W/r405" 43 "MJR1_INVALID 405 length" || return 1
    : > "$W/r0"
    ex mjr1-empty J "$W/r0" 43 "MJR1_INVALID 0 length" || return 1
    ex mjr1-missing J "$W/does-not-exist" 43 "MJR1_INVALID 0 io" || return 1
    # spec: offset byte reason reported-position (range rules report the range start)
    for spec in "5 0 outcome 5" "5 6 outcome 5" "5 255 outcome 5" "0 0 magic 0" "4 2 version 4" "7 1 reserved 7" "86 1 reserved 84" "338 1 reserved 337" "6 0 attempt 6" "304 4 signer 304"; do
        # shellcheck disable=SC2086
        set -- $spec
        cp "$RCPT" "$W/rm"
        printf "\\$(printf %o "$2")" | dd of="$W/rm" bs=1 seek="$1" conv=notrunc 2> /dev/null
        ex "mjr1-$3-$1-$2" J "$W/rm" 43 "MJR1_INVALID $4 $3" || return 1
    done
    cp "$RCPT" "$W/rm"
    dd if=/dev/zero of="$W/rm" bs=1 seek=144 count=32 conv=notrunc 2> /dev/null
    ex mjr1-completed-zero-output J "$W/rm" 43 "MJR1_INVALID 144 output" || return 1
    cp "$RCPT" "$W/rm"
    printf '\002' | dd of="$W/rm" bs=1 seek=5 conv=notrunc 2> /dev/null
    ex mjr1-refused-with-output J "$W/rm" 43 "MJR1_INVALID 144 output" || return 1
    cp "$RCPT" "$W/rm"
    dd if=/dev/zero of="$W/rm" bs=1 seek=176 count=32 conv=notrunc 2> /dev/null
    ex mjr1-zero-source J "$W/rm" 43 "MJR1_INVALID 176 zerohash" || return 1
    cp "$RCPT" "$W/rm"
    dd if=/dev/zero of="$W/rm" bs=1 seek=240 count=32 conv=notrunc 2> /dev/null
    ex mjr1-zero-policy J "$W/rm" 43 "MJR1_INVALID 240 zerohash" || return 1
    cp "$RCPT" "$W/rm"
    dd if=/dev/zero of="$W/rm" bs=1 seek=272 count=32 conv=notrunc 2> /dev/null
    ex mjr1-zero-verdict J "$W/rm" 43 "MJR1_INVALID 272 verdict" || return 1
    # A zero verdict is required only of COMPLETED: refused, zero output, zero verdict is valid.
    z64=0000000000000000000000000000000000000000000000000000000000000000
    printf '\002' | dd of="$W/rm" bs=1 seek=5 conv=notrunc 2> /dev/null
    dd if=/dev/zero of="$W/rm" bs=1 seek=144 count=32 conv=notrunc 2> /dev/null
    ex mjr1-refused-zero-verdict J "$W/rm" 0 "MJR1 refused 1 1 $dag $job $z64" || return 1
    # attempt 2, generation bytes 01 02 03 04 big-endian (Wolfram FromDigits[{1,2,3,4},256] -> 16909060).
    cp "$RCPT" "$W/rm"
    printf '\002' | dd of="$W/rm" bs=1 seek=6 conv=notrunc 2> /dev/null
    printf '\001\002\003\004' | dd of="$W/rm" bs=1 seek=72 conv=notrunc 2> /dev/null
    ex mjr1-attempt2-gen J "$W/rm" 0 "MJR1 completed 2 16909060 $dag $job $dig" || return 1
    cp "$RCPT" "$W/rm"
    printf '\003' | dd of="$W/rm" bs=1 seek=304 conv=notrunc 2> /dev/null
    ex mjr1-signer-03 J "$W/rm" 0 "MJR1 completed 1 1 $dag $job $dig" || return 1
    # Usage.
    ex usage-mode X "" 2 "" || return 1
    ex usage-ids D "1  2" 2 "" || return 1
    # Generated valid messages of all four kinds, edges included.
    for d in 000 001 040 0ff 100 101 fff; do
        for s in 0000 0001 abcd ffff; do
            ops="$(D "$d") $(D 040) $(D "$d") $(S "$s")"
            valid "gmm-$d-$s" G 24 "goal mm $ops end" || return 1
            valid "gpair-$d-$s" G 30 "goal pair $ops $(S 7e57) end" || return 1
            valid "ajob-$d-$s" A 25 "act job mm $ops end" || return 1
            valid "pokmm-$d-$s" P 25 "rep ok mm $ops end" || return 1
            valid "pokpair-$d-$s" P 31 "rep ok pair $ops $(S 7e57) end" || return 1
        done
    done
    valid aref A 4 "act refuse bound end" full || return 1
    valid rok R 3 "res ok end" || return 1
    for c in replay budget worker transport receipt recomp verdict io; do
        valid "rref-$c" R 4 "res refused $c end" || return 1
        valid "rfail-$c" R 4 "res fail $c end" full || return 1
    done
    for c in bound replay budget worker transport receipt recomp verdict io; do
        valid "pref-$c" P 4 "rep refused $c end" || return 1
        valid "pfail-$c" P 4 "rep fail $c end" || return 1
    done
    valid gpair-full G 30 "goal pair $(D 001) $(D 0a0) $(D 100) $(S 0000) $(S ffff) end" full || return 1
    valid pokpair-full P 31 "rep ok pair $(D 101) $(D fff) $(D 000) $(S 1234) $(S 9abc) end" full || return 1
    # Rejections with typed reasons and byte positions.
    ex rej-unknown V "goal xx" 40 "INVALID 5 unknown" || return 1
    ex rej-upper V "Goal mm" 40 "INVALID 0 upper" || return 1
    ex rej-0X V "goal mm ushort 0X040" 40 "INVALID 16 hexcase" || return 1
    ex rej-hexupper V "goal mm ushort 0x04A" 40 "INVALID 19 upper" || return 1
    ex rej-space2 V "goal  mm" 40 "INVALID 5 space" || return 1
    ex rej-space-lead V " goal" 40 "INVALID 0 space" || return 1
    ex rej-space-trail V "res ok end " 40 "INVALID 10 space" || return 1
    ex rej-tab V "$(printf 'goal\tmm')" 40 "INVALID 4 tab" || return 1
    ex rej-cr V "$(printf 'res ok end\r')" 40 "INVALID 10 cr" || return 1
    ex rej-nonascii V "$(printf 'goal m\303\251')" 40 "INVALID 6 nonascii" || return 1
    ex rej-ctrl V "$(printf 'res\001ok end')" 40 "INVALID 3 ctrl" || return 1
    ex rej-two-lines V "res ok end${NL}res ok end" 40 "INVALID 10 ctrl" || return 1
    ex rej-empty V "" 40 "INVALID 0 empty" || return 1
    ex rej-noend V "act refuse bound" 40 "INVALID 16 noend" || return 1
    ex rej-noend-short V "goal mm ushort 0x04" 40 "INVALID 19 noend" || return 1
    ex rej-trailing V "act refuse bound end end" 40 "INVALID 21 trailing" || return 1
    ex rej-trailing-bytes V "res ok end x" 40 "INVALID 11 unknown" || return 1
    ex rej-overlen-digit V "goal mm ushort 0x0400" 40 "INVALID 20 grammar" || return 1
    ex rej-overlen-seed V "act job mm $(D 040) $(D 040) $(D 040) ulong 0x00010 end" 40 "INVALID 62 grammar" || return 1
    ex rej-pad V "<pad>" 40 "INVALID 0 kind" || return 1
    ex rej-bare-digit V "res 0" 40 "INVALID 4 unknown" || return 1
    ex rej-R-bound V "res refused bound end" 40 "INVALID 12 grammar" || return 1
    ex rej-Rfail-bound V "res fail bound end" 40 "INVALID 9 grammar" || return 1
    ex rej-arefuse-code V "act refuse replay end" 40 "INVALID 11 grammar" || return 1
    ex rej-apair V "act job pair $(D 040) $(D 040) $(D 040) $(S 0001) end" 40 "INVALID 8 grammar" || return 1
    ex rej-gref V "goal refuse bound end" 40 "INVALID 5 grammar" || return 1
    ex rej-pokref V "rep ok refuse end" 40 "INVALID 7 grammar" || return 1
    long=$(printf 'res ok end %.0s' $(seq 1 100))
    ex rej-overlong-bytes V "${long% }" 40 "INVALID 1024 long" || return 1
    ex rej-overlong-ids E "$(printf '0x0000000000 %.0s' $(seq 1 16))0x" 40 "INVALID 208 long" || return 1
    # Dead state: empty mask; model start allows act|rep only.
    ex dead-G N "G${NL}2" 0 "MASK 00000000000000000000000000000000000000000${NL}FINAL 0" || return 1
    ex dead-after-end N "A${NL}2 9 13 5 5" 0 "MASK 00000000000000000000000000000000000000000${NL}FINAL 0" || return 1
    ex start-M N "M$NL" 0 "MASK 00101000000000000000000000000000000000000${NL}FINAL 0" || return 1
    ex start-G N "G$NL" 0 "MASK 01000000000000000000000000000000000000000${NL}FINAL 0" || return 1
    q N "A${NL}2 8"; ma=$O
    ex M-follows-A N "M${NL}2 8" 0 "$ma" || return 1
    q N "P${NL}4 10 7 22 24"; mp=$O
    ex M-follows-P N "M${NL}4 10 7 22 24" 0 "$mp" || return 1
    ex digits-shared N "G${NL}1 6 22 24" 0 "MASK 00000000000000000000000001111111111111111${NL}FINAL 0" || return 1
    ex codes-P N "P${NL}4 11" 0 "MASK 00000000000001111111110000000000000000000${NL}FINAL 0" || return 1
    ex codes-R N "R${NL}3 12" 0 "MASK 00000000000000111111110000000000000000000${NL}FINAL 0" || return 1
    valid M-act-job A 25 "act job mm $(D 001) $(D 002) $(D 003) $(S 0004) end" || return 1
    return 0
}

UNITS="$V/vocab.om $V/msg.om $V/mjr1.om $ROOT/tests/omegese45/contract/main.om"
# shellcheck disable=SC2086
cat $UNITS > "$OUT/unit.om"
build "$OUT/unit.om" "$OUT/oe45c" unit || fail compile
E=$OUT/oe45c
suite unit || fail "$(cat "$W/why")"
echo "unit PASS $(wc -l < "$OUT/run/unit.checks" | tr -d ' ') checks" > "$OUT/run/unit.verdict"

# Negative control: each mutant replaces exactly one line of one unit and must
# fail the suite at the named check (first failing check == expected killer).
# Fields: tag | unit | original line | mutated line | expected killer | what breaks
MUTANTS='m1|msg.om|x = ocdcode x 14|x = ocdcode x 13|rej-R-bound|DFA: R refused admits the C set (bound)
m2|msg.om|for d ulong 15|for d ulong 14|bind-bound-101|DFA: shared digit transition omits f
b1|msg.om|if ne sd get ocgv add 3 job|if ne sd get ocgv 4|bind-ok2|binding 41: seed always compared with job 1
b2|msg.om|const ocadmax ulong 256|const ocadmax ulong 257|bind-bound-101|binding 42: dim 0x101 admitted
b3|msg.om|ret and ge d 1 le d ocadmax|ret and ge d 0 le d ocadmax|bind-bound-000|binding 42: dim 0x000 admitted
b4|msg.om|if ne k get ocgv 2|if ne k k|bind-dims|binding 41: k not compared with G
b5|msg.om|if eq get ocids 1 9|if eq get ocids 1 99|bind-notjob|binding 41: act refuse not reported notjob
b6|msg.om|if or eq job 0 gt job nj|if or eq job 0 gt job 2|bind-mm-job2|binding 41: job 2 admitted for mm
c1|vocab.om|if and ge b 97 le b 102|if and ge b 97 le b 101|rt40-e|codec: hex digit f not encoded
c2|vocab.om|if lt get ocids sub i 1 24|if lt get ocids sub i 1 0|rej-dec-digit|codec: digit id decoded after a word
c3|vocab.om|if and eq b 88 gt i 0|if and eq b 89 gt i 0|rej-0X|codec: 0X reported upper, not hexcase
c4|vocab.om|const ocmaxi ulong 176|const ocmaxi ulong 177|rej-overlong-ids|codec: 177 ids admitted
c5|vocab.om|for i ulong 24|for i ulong 23|rt23-e|codec: word id 23 (ulong) not looked up
j1|mjr1.om|if done|if eq oc 9|mjr1-zero-verdict|MJR1: zero verifier_verdict admitted when completed
j2|mjr1.om|if ocmz 240 272|if ocmz 176 208|mjr1-zero-policy|MJR1: zero policy_hash admitted
j3|mjr1.om|if done|if true|mjr1-refused-zero-verdict|MJR1: zero verdict refused when not completed
j4|mjr1.om|if eq get ocrb 6 0|if eq get ocrb 6 9|mjr1-attempt-6-0|MJR1: attempt 0 admitted
j5|mjr1.om|if or lt oc 1 gt oc 5|if or lt oc 1 gt oc 6|mjr1-outcome-5-6|MJR1: outcome 6 admitted
j6|mjr1.om|if and ne sg 2 ne sg 3|if and ne sg 2 ne sg 2|mjr1-signer-03|MJR1: signer prefix 0x03 refused
j7|mjr1.om|v = add mul v 256 ulong get ocrb add a i|v = add mul v 256 ulong get ocrb add a sub 3 i|mjr1-real|MJR1: generation read little-endian
j8|mjr1.om|put ocmv 1 ulong get ocrb 6|put ocmv 1 ulong 1|mjr1-attempt2-gen|MJR1: attempt not reported'
: > "$OUT/run/mutants.verdict"
MSUMS=
while IFS='|' read -r m u old new want what; do
    awk -v o="$old" -v n="$new" '!d && $0==o {print n; d=1; next} {print} END {exit !d}' "$V/$u" > "$OUT/mutants/$m-$u" || fail "mutant-$m-line-not-found"
    cmp -s "$V/$u" "$OUT/mutants/$m-$u" && fail "mutant-$m-unchanged"
    : > "$OUT/mutants/$m.om"
    for f in vocab.om msg.om mjr1.om; do
        if [ "$f" = "$u" ]; then cat "$OUT/mutants/$m-$u"; else cat "$V/$f"; fi >> "$OUT/mutants/$m.om"
    done
    cat "$ROOT/tests/omegese45/contract/main.om" >> "$OUT/mutants/$m.om"
    build "$OUT/mutants/$m.om" "$OUT/mutants/$m.elf" "$m" || fail "mutant-$m-compile"
    E=$OUT/mutants/$m.elf
    if suite "$m"; then
        fail "mutant-$m-survived"
    fi
    got=$(cut -d' ' -f1 "$W/why")
    [ "$got" = "$want" ] || fail "mutant-$m-killed-by-$got-not-$want"
    echo "$m KILLED $(cat "$W/why") | $u: $what" >> "$OUT/run/mutants.verdict"
    MSUMS="$MSUMS mutants/$m.om mutants/$m.elf"
done <<EOF
$MUTANTS
EOF

# shellcheck disable=SC2086
(cd "$OUT" && sha256sum unit.om oe45c $MSUMS) > "$OUT/artifacts.sha256"
sha256sum "$OMC" $UNITS "$TSV" "$RCPT" >> "$OUT/artifacts.sha256"
cat "$OUT/run/unit.verdict" "$OUT/run/mutants.verdict"
echo "mutants $(wc -l < "$OUT/run/mutants.verdict" | tr -d ' ') KILLED 0 survived"
echo OE45_CONTRACT_PASS

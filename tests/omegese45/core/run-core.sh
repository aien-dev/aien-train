#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# omegese45 pipeline suite (aien-train #45): generator, model, checkpoint and decode CLI.
#   OMC=<omc bf54b19f> OMEGA=<omega-mv checkout with bf54b19f> OUT=<scratch dir> sh tests/omegese45/core/run-core.sh
# Host glue only (build via build.sh, file copies, byte flips, comparisons); every
# computation is in .om except tests/omegese45/core/oracle.py, the stdlib referee.
# Writes each rc/stdout under $OUT/run. Last line OE45_CORE_PASS (exit 0) or
# OE45_CORE_FAIL <check> (exit 1). Heavy steps run under taskset/nice/ionice.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
: "${OMC:?set OMC to the pinned seed/out/omc}"
: "${OMEGA:?set OMEGA to an omega-mv checkout}"
: "${OUT:?set OUT to a scratch directory}"
mkdir -p "$OUT/bin" "$OUT/run" "$OUT/w" "$OUT/mut"
OUT=$(cd "$OUT" && pwd)
P="taskset -c 10-14 nice -n 19 ionice -c3"
ORA="python3 $ROOT/tests/omegese45/core/oracle.py"
VOCAB=$ROOT/docs/omegese45/vocab.tsv
B=$OUT/bin
W=$OUT/w
fail() { echo "OE45_CORE_FAIL $1"; exit 1; }
ok() { echo "ok $1"; }
# run TAG CMD...: run in $W, keep stdout/stderr/rc, print the rc.
run() {
    tag=$1; shift
    (cd "$W" && $P "$@" > "$OUT/run/$tag.stdout" 2> "$OUT/run/$tag.stderr")
    echo $? > "$OUT/run/$tag.rc"
    cat "$OUT/run/$tag.rc"
}
flip() { # flip FILE OFFSET: xor one byte with 1
    b=$(od -An -tu1 -j"$2" -N1 "$1" | tr -d ' ')
    printf "\\$(printf %o $((b ^ 1)))" | dd of="$1" bs=1 seek="$2" conv=notrunc 2> /dev/null
}
field() { # field TAG N: Nth space-separated word of the first stdout line
    sed -n 1p "$OUT/run/$1.stdout" | cut -d' ' -f"$2"
}

# 1. Build (pinned compiler and SHA unit checked by build.sh).
OMC=$OMC OMEGA=$OMEGA OUT=$B sh "$ROOT/tests/omegese45/core/build.sh" > "$OUT/run/build.out" || fail "build $(cat "$OUT/run/build.out")"
ok build

# 2. Generator: determinism, seed sensitivity, contract validity, reference policy.
[ "$(run gen1 "$B/oe45gen" 4501 300 train.oe45 train.ids)" = 0 ] || fail gen
[ "$(run gen2 "$B/oe45gen" 4501 300 again.oe45 again.ids)" = 0 ] || fail gen-rerun
cmp -s "$W/train.oe45" "$W/again.oe45" && cmp -s "$W/train.ids" "$W/again.ids" || fail gen-determinism
cmp -s "$OUT/run/gen1.stdout" "$OUT/run/gen2.stdout" || fail gen-stats-determinism
[ "$(run gen3 "$B/oe45gen" 4502 60 dev.oe45 dev.ids)" = 0 ] || fail gen-dev
cmp -s "$W/train.ids" "$W/dev.ids" && fail gen-seed-insensitive
[ "$(run vgen "$B/oe45test" vgen train.oe45 train.ids)" = 0 ] || fail gen-contract-validity
[ "$(field vgen 2)" = 300 ] || fail gen-episode-count
[ "$(run vgendev "$B/oe45test" vgen dev.oe45 dev.ids)" = 0 ] || fail gen-contract-validity-dev
cp "$W/train.ids" "$W/bad.ids"
flip "$W/bad.ids" 40
[ "$(run vgenbad "$B/oe45test" vgen train.oe45 bad.ids)" = 40 ] || fail vgen-negative-control
$ORA policy "$W/train.oe45" > "$OUT/run/policy.stdout" || fail reference-policy
$ORA policy "$W/dev.oe45" > "$OUT/run/policydev.stdout" || fail reference-policy-dev
ok generator

# 3. Step-0 checkpoint: deterministic init, vocab pin equals vocab.tsv.
mkdir -p "$W/ck0" "$W/ck0b"
[ "$(run init "$B/oe45core" init ck0 train.ids 45 300 1200 7 7)" = 0 ] || fail init
[ "$(run initb "$B/oe45core" init ck0b train.ids 45 300 1200 7 7)" = 0 ] || fail init-rerun
cmp -s "$W/ck0/ck.a" "$W/ck0b/ck.a" || fail init-determinism
[ -e "$W/ck0/ck.b" ] && fail init-one-slot
[ "$(run info0 "$B/oe45core" info ck0)" = 0 ] || fail info
[ "$(field info0 12)" = "$(sha256sum "$VOCAB" | cut -d' ' -f1)" ] || fail vocab-pin
sha256sum "$W/ck0/ck.a" > "$OUT/run/ck0.sha256"
ok step0

# 4. Gradient check, loss decrease, target/mask oracle, causal no-leak.
[ "$(run grad "$B/oe45test" grad ck0 train.ids)" = 0 ] || fail gradient-check
[ "$(run ldec "$B/oe45test" ldec ck0 train.ids 50)" = 0 ] || fail loss-decrease
[ "$(run mask "$B/oe45test" mask train.ids 300)" = 0 ] || fail mask-dump
$ORA mask "$VOCAB" "$W/train.oe45" 300 > "$OUT/run/mask.expect" || fail mask-oracle
cmp -s "$OUT/run/mask.stdout" "$OUT/run/mask.expect" || fail mask-target
[ "$(run leak "$B/oe45test" leak ck0 train.ids)" = 0 ] || fail causal-no-leak
ok model

# 5. Training, kill -9 mid-run, resume: byte-identical to the uninterrupted run.
cp -r "$W/ck0" "$W/cku"
cp -r "$W/ck0" "$W/ckk"
[ "$(run train "$B/oe45core" train cku train.ids 1200 100)" = 0 ] || fail train
krc=$(run kill timeout -s KILL 2 "$B/oe45core" train ckk train.ids 1200 100)
[ "$krc" = 137 ] || fail "kill-not-mid-run rc=$krc"
[ "$(run resume "$B/oe45core" train ckk train.ids 1200 100)" = 0 ] || fail resume
cmp -s "$W/cku/ck.a" "$W/ckk/ck.a" && cmp -s "$W/cku/ck.b" "$W/ckk/ck.b" || fail resume-not-identical
[ "$(run bind "$B/oe45core" train cku dev.ids 1300 100)" = 54 ] || fail data-binding
ok resume

# 6. Round trip: load then save reproduces the payload exactly.
mkdir -p "$W/ckr"
[ "$(run infou "$B/oe45core" info cku)" = 0 ] || fail info-trained
slot=$(field infou 2)
[ "$(run rt "$B/oe45test" rt cku ckr)" = 0 ] || fail round-trip
r=$(ls "$W/ckr")
tail -c +241 "$W/cku/ck.$slot" > "$OUT/run/rt.src"
tail -c +241 "$W/ckr/$r" > "$OUT/run/rt.dst"
cmp -s "$OUT/run/rt.src" "$OUT/run/rt.dst" || fail round-trip-payload
[ "$(run infor "$B/oe45core" info ckr)" = 0 ] || fail info-round-trip
[ "$(field infor 4)" = "$(field infou 4)" ] || fail round-trip-step
ok round-trip

# 7. Corrupt slots: newest corrupt -> older slot; both corrupt -> 51; wrong vocab -> 52.
gen=$(field infou 5)
other=a
[ "$slot" = a ] && other=b
cp -r "$W/cku" "$W/ckc"
flip "$W/ckc/ck.$slot" 5000
[ "$(run infoc "$B/oe45core" info ckc)" = 0 ] || fail fallback
[ "$(field infoc 2)" = "$other" ] && [ "$(field infoc 5)" = $((gen - 1)) ] || fail fallback-slot
cp -r "$W/cku" "$W/ckh"
flip "$W/ckh/ck.$slot" 100
[ "$(run infoh "$B/oe45core" info ckh)" = 0 ] && [ "$(field infoh 2)" = "$other" ] || fail fallback-header
cp -r "$W/cku" "$W/ckt"
head -c 1000 "$W/cku/ck.$slot" > "$W/ckt/ck.$slot"
[ "$(run infot "$B/oe45core" info ckt)" = 0 ] && [ "$(field infot 2)" = "$other" ] || fail fallback-truncated
flip "$W/ckc/ck.$other" 3000000
[ "$(run info51 "$B/oe45core" info ckc)" = 51 ] || fail both-corrupt-51
mkdir -p "$W/ckn"
[ "$(run info51e "$B/oe45core" info ckn)" = 51 ] || fail empty-51
mkdir -p "$W/ckv"
[ "$(run forcev "$B/oe45test" force cku ckv 3)" = 0 ] || fail force-vocab
[ "$(run info52 "$B/oe45core" info ckv)" = 52 ] || fail wrong-vocab-52
ok checkpoint

# 8. Decode: U and G, guided decode never emits an illegal id, budget 53, invalid 50, prefix 40.
$ORA prefixes "$W/dev.oe45" 60 "$W/pre.oe45" "$OUT/run/pre.expect" || fail prefixes
{ echo "oe45 v1"; sed -n 2p "$W/dev.oe45"; } > "$W/one.oe45"
for c in ck0 cku; do
    for m in U G; do
        [ "$(run "b$c$m" "$B/oe45core" batch $c $m pre.oe45 "b$c$m.out")" = 0 ] || fail "batch-$c-$m"
        [ "$(run "v$c$m" "$B/oe45test" vok "b$c$m.out")" = 0 ] || fail "vok-$c-$m"
    done
    [ "$(field "v${c}G" 3)" = 0 ] && [ "$(field "v${c}G" 4)" = 0 ] && [ "$(field "v${c}G" 5)" = 0 ] && [ "$(field "v${c}G" 6)" = 0 ] && [ "$(field "v${c}G" 7)" = 0 ] || fail "guided-illegal-$c"
    [ "$(field "v${c}U" 3)" = 0 ] || fail "unguided-invalid-ok-$c"
done
[ "$(wc -l < "$W/bckuG.out")" = "$(wc -l < "$OUT/run/pre.expect")" ] || fail batch-line-count
[ "$(run decg "$B/oe45core" decode cku G one.oe45)" = 0 ] || fail decode-G
[ "ok $(cat "$OUT/run/decg.stdout")" = "$(sed -n 1p "$W/bckuG.out")" ] || fail decode-batch-agree
urc=$(run decu "$B/oe45core" decode cku U one.oe45)
case $urc in 0|50|53) ;; *) fail "decode-U rc=$urc" ;; esac
mkdir -p "$W/ckm" "$W/cke"
[ "$(run forcem "$B/oe45test" force cku ckm 1)" = 0 ] || fail force-mm
[ "$(run dec53 "$B/oe45core" decode ckm U one.oe45)" = 53 ] || fail budget-53
[ -s "$OUT/run/dec53.stdout" ] && fail budget-53-stdout
[ "$(run dec53g "$B/oe45core" decode ckm G one.oe45)" = 0 ] || fail guided-under-mm-bias
[ "$(run forcee "$B/oe45test" force cku cke 2)" = 0 ] || fail force-end
[ "$(run dec50 "$B/oe45core" decode cke U one.oe45)" = 50 ] || fail invalid-50
{ echo "oe45 v1"; sed -n 2,3p "$W/dev.oe45"; } > "$W/badp.oe45"
[ "$(run dec40 "$B/oe45core" decode cku G badp.oe45)" = 40 ] || fail prefix-40
{ echo "oe45 v2"; sed -n 2p "$W/dev.oe45"; } > "$W/badh.oe45"
[ "$(run dec40h "$B/oe45core" decode cku G badh.oe45)" = 40 ] || fail header-40
[ "$(run dec52 "$B/oe45core" decode ckv G one.oe45)" = 52 ] || fail decode-52
ok decode

# 9. Mutants: each must be KILLED by the named check.
[ "$(run m0 sh -c "printf 0 | cat - '$ROOT/native/omegese45/core/model.om' | '$B/oe45mut' > '$OUT/mut/model0.om'")" = 0 ] || fail mutant0-inject
[ "$(run m1 sh -c "printf 1 | cat - '$ROOT/native/omegese45/core/model.om' | '$B/oe45mut' > '$OUT/mut/model1.om'")" = 0 ] || fail mutant1-inject
OMC=$OMC OMEGA=$OMEGA OUT=$OUT/mut/b0 sh "$ROOT/tests/omegese45/core/build.sh" "$OUT/mut/model0.om" > "$OUT/run/mb0.out" || fail mutant0-build
OMC=$OMC OMEGA=$OMEGA OUT=$OUT/mut/b1 sh "$ROOT/tests/omegese45/core/build.sh" "$OUT/mut/model1.om" > "$OUT/run/mb1.out" || fail mutant1-build
g0=$(run mgrad0 "$OUT/mut/b0/oe45test" grad ck0 train.ids)
[ "$g0" = 86 ] || fail "mutant0-sign-flip-survived rc=$g0"
echo "mutant0 sign-flip KILLED by gradient check (rc 86)"
[ "$(run mmask1 "$OUT/mut/b1/oe45test" mask train.ids 300)" = 0 ] || fail mutant1-mask-run
cmp -s "$OUT/run/mmask1.stdout" "$OUT/run/mask.expect" && fail mutant1-wrong-target-survived
echo "mutant1 wrong-target KILLED by mask/target oracle"
echo OE45_CORE_PASS

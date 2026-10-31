#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# adtrain suite: native tape training example, gates 1-4 and the host arm of 5.
#   OMC=<pinned seed/out/omc> OMEGA=<omega-mv checkout> OUT=<scratch dir> sh tests/adtrain/run-adtrain.sh
# OMEGA must contain the commits pinned in docs/adtrain/dependencies.json; the
# fixture host arm runs `make test-autodiff-fixture` in OMEGA as checked out.
# Host glue only (concatenation, file copies, byte flips, comparisons); every
# computation is in .om. Writes each rc/stdout/stderr under $OUT/run.
# Last line ADTRAIN_SUITE_PASS (exit 0) or ADTRAIN_SUITE_FAIL <gate> (exit 1).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
: "${OMC:?set OMC to the pinned seed/out/omc}"
: "${OMEGA:?set OMEGA to an omega-mv checkout}"
: "${OUT:?set OUT to a scratch directory}"
DEPS=$ROOT/docs/adtrain/dependencies.json
mkdir -p "$OUT/deps" "$OUT/run" "$OUT/work" "$OUT/mutants"
OUT=$(cd "$OUT" && pwd)
fail() { echo "ADTRAIN_SUITE_FAIL $1"; exit 1; }

# 1. Pinned compiler and dependency files from their exact Git objects.
want=$(jq -r .compiler.binary_sha256 "$DEPS")
have=$(sha256sum "$OMC" | cut -d' ' -f1)
echo "$have  omc" > "$OUT/deps/omc.sha256"
[ "$have" = "$want" ] || fail "compiler-sha256 ($have)"
: > "$OUT/deps/files"
n=$(jq '.dependencies | length' "$DEPS")
i=0
while [ "$i" -lt "$n" ]; do
    commit=$(jq -r ".dependencies[$i].commit" "$DEPS")
    path=$(jq -r ".dependencies[$i].path" "$DEPS")
    sha=$(jq -r ".dependencies[$i].sha256" "$DEPS")
    dst=$OUT/deps/$i-$(basename "$path")
    git -C "$OMEGA" show "$commit:$path" > "$dst" 2> "$OUT/deps/$i.stderr" || fail "dependency-missing $commit:$path"
    echo "$sha  $dst" | sha256sum -c --status || fail "dependency-sha256 $path"
    echo "$dst" >> "$OUT/deps/files"
    i=$((i + 1))
done
DEPFILES=$(cat "$OUT/deps/files")

build() { # $1 unit.om  $2 out.elf  $3 tag
    "$OMC" < "$1" > "$2" 2> "$OUT/run/$3.compile.stderr"
    echo $? > "$OUT/run/$3.compile.rc"
    [ "$(cat "$OUT/run/$3.compile.rc")" = 0 ] || return 1
    [ "$(od -An -tx1 -N4 "$2" | tr -d ' ')" = 7f454c46 ] || return 1
    chmod 700 "$2"
}
# run TAG ELF MODE: run one driver mode inside $OUT/work.
run() {
    (cd "$OUT/work" && printf '%s' "$3" | "$2" > "$OUT/run/$1.stdout" 2> "$OUT/run/$1.stderr")
    echo $? > "$OUT/run/$1.rc"
    cat "$OUT/run/$1.rc"
}
flip() { # flip FILE OFFSET: xor one byte with 1
    b=$(od -An -tu1 -j"$2" -N1 "$1" | tr -d ' ')
    printf "\\$(printf %o $((b ^ 1)))" | dd of="$1" bs=1 seek="$2" conv=notrunc 2> /dev/null
}

# shellcheck disable=SC2086
cat $DEPFILES "$ROOT/native/adtrain/mlp.om" "$ROOT/tests/adtrain/main.om" > "$OUT/unit.om"
build "$OUT/unit.om" "$OUT/unit.elf" unit || fail compile
E=$OUT/unit.elf
CK=$OUT/work/adtrain.ck

# 2. Gates 1 and 3: forward, loss, updates that reduce loss (T).
[ "$(run T "$E" T)" = 0 ] || fail train
grep -qx ADTRAIN_T_PASS "$OUT/run/T.stdout" || fail train-output
[ "$(grep -c '^L ' "$OUT/run/T.stdout")" = 201 ] || fail train-L-rows
[ "$(grep -c '^P ' "$OUT/run/T.stdout")" = 58 ] || fail train-P-rows
l0=$(awk '$1=="L"&&$2==0{print $3}' "$OUT/run/T.stdout")
l100=$(awk '$1=="L"&&$2==100{print $3}' "$OUT/run/T.stdout")
l200=$(awk '$1=="L"&&$2==200{print $3}' "$OUT/run/T.stdout")
# loss >= 0, so binary32 bit patterns order like the values
[ "$l200" -lt "$l0" ] && [ "$l200" -lt "$l100" ] || fail loss-gate
echo "L0=$l0 L100=$l100 L200=$l200" > "$OUT/run/loss-gate.txt"

# 3. Gate 2: tape gradients against central differences (G).
[ "$(run G "$E" G)" = 0 ] || fail gradcheck
ok=$(grep -c '^GC [0-9]* [0-9]* 1$' "$OUT/run/G.stdout")
bad=$(grep -c '^GC [0-9]* [0-9]* 0$' "$OUT/run/G.stdout")
[ "$ok" -ge 50 ] && [ "$bad" = 0 ] || fail "gradcheck-rows ok=$ok bad=$bad"

# 4. Fixture for the C oracle (D).
[ "$(run D "$E" D)" = 0 ] || fail fixture
[ "$(wc -c < "$OUT/work/adtrain.fix")" = 748 ] || fail fixture-size
cp "$OUT/work/adtrain.fix" "$OUT/adtrain.fix"

# 5. Gate 4: interrupted at 120, resumed to 200, identical to uninterrupted.
[ "$(run S "$E" S)" = 0 ] || fail save
[ "$(wc -c < "$CK")" = 536 ] || fail checkpoint-size
cp "$CK" "$OUT/adtrain.ck.good"
[ "$(run R "$E" R)" = 0 ] || fail resume
grep -qx ADTRAIN_R_PASS "$OUT/run/R.stdout" || fail resume-output
awk '($1=="L"&&$2>=120)||$1=="P"' "$OUT/run/T.stdout" > "$OUT/run/T.tail"
awk '$1=="L"||$1=="P"' "$OUT/run/R.stdout" > "$OUT/run/R.tail"
[ "$(grep -c '^L ' "$OUT/run/R.tail")" = 81 ] || fail resume-L-rows
cmp -s "$OUT/run/T.tail" "$OUT/run/R.tail" || fail resume-identity

# 6. Checkpoint refusals: other data (66), truncated (65), flipped bytes (65).
[ "$(run Z "$E" Z)" = 0 ] || fail save-other-data
cp "$CK" "$OUT/adtrain.ck.seed29"
[ "$(run R-binding "$E" R)" = 66 ] || fail binding-refusal
head -c 535 "$OUT/adtrain.ck.good" > "$CK"
[ "$(run R-truncated "$E" R)" = 65 ] || fail truncation-refusal
cp "$OUT/adtrain.ck.good" "$OUT/adtrain.ck.flip96"
flip "$OUT/adtrain.ck.flip96" 96
cp "$OUT/adtrain.ck.flip96" "$CK"
[ "$(run R-flip96 "$E" R)" = 65 ] || fail checksum-refusal
cp "$OUT/adtrain.ck.good" "$CK"
flip "$CK" 100
[ "$(run R-flip100 "$E" R)" = 65 ] || fail padding-refusal

# 7. Compiled mutants of mlp.om; each must be caught.
build "$ROOT/tests/adtrain/mutate.om" "$OUT/mutate.elf" mutate || fail mutate-compile
for mode in 0 1 2; do
    { printf '%s' "$mode"; cat "$ROOT/native/adtrain/mlp.om"; } | "$OUT/mutate.elf" > "$OUT/mutants/mlp$mode.om"
    echo $? > "$OUT/mutants/mlp$mode.mutate.rc"
    [ "$(cat "$OUT/mutants/mlp$mode.mutate.rc")" = 0 ] || fail "mutant$mode-inject"
    # shellcheck disable=SC2086
    cat $DEPFILES "$OUT/mutants/mlp$mode.om" "$ROOT/tests/adtrain/main.om" > "$OUT/mutants/unit$mode.om"
    build "$OUT/mutants/unit$mode.om" "$OUT/mutants/unit$mode.elf" "mutant$mode" || fail "mutant$mode-compile"
done
rc=$(run mutant0-T "$OUT/mutants/unit0.elf" T)
[ "$rc" = 86 ] || [ "$rc" = 88 ] || fail "mutant0-survived rc=$rc"
cp "$OUT/adtrain.ck.seed29" "$CK"
[ "$(run mutant1-R-binding "$OUT/mutants/unit1.elf" R)" = 0 ] || fail mutant1-not-load-bearing
cp "$OUT/adtrain.ck.flip96" "$CK"
[ "$(run mutant2-R-flip96 "$OUT/mutants/unit2.elf" R)" = 0 ] || fail mutant2-not-load-bearing

# 8. Gate 5 host arm: the fixture replayed on the C M21 tape (CPU table).
git -C "$OMEGA" rev-parse HEAD > "$OUT/run/fixture-host.omega-head"
make -s -C "$OMEGA" test-autodiff-fixture FIXTURE="$OUT/adtrain.fix" \
    > "$OUT/run/fixture-host.stdout" 2> "$OUT/run/fixture-host.stderr"
echo $? > "$OUT/run/fixture-host.rc"
[ "$(cat "$OUT/run/fixture-host.rc")" = 0 ] || fail fixture-host
grep -Eq '^AD_FIXTURE_HOST: exact=[0-9]+/75 max_ulp=[0-9]+ verdict=(PASS|PASS_TOL)$' \
    "$OUT/run/fixture-host.stdout" || fail fixture-host-verdict
grep -q '^AD_FIXTURE_HOST: .* verdict=FAIL (mutant-sign)$' "$OUT/run/fixture-host.stdout" \
    || fail fixture-host-control

(cd "$OUT" && sha256sum unit.om unit.elf adtrain.fix adtrain.ck.good run/*.stdout) > "$OUT/artifacts.sha256"
echo ADTRAIN_SUITE_PASS

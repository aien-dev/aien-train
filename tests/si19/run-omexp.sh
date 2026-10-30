#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# aien-train#24 ORIG-1 host glue (POSIX sh, disclosed): materialize the pinned sources,
# verify SHA-256, build the omexp test and driver with the pinned seeded compiler, run the
# admission/refusal tests, generate the SYNTHETIC fixture, admit it, and (when RECORDS is
# set) admit the local owner dataset. Every check and verdict is in the .om units.
# Host CPU only. No GPU, network, service or model process is touched.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$ROOT/evidence/SI19-ORIG1/run}
OMC=${OMC:-$HOME/workspace/omega-toc/seed/out/omc}
OMC2=${OMC2:-$HOME/workspace/omega-seedqual-20261025/seed/out/omc}
BRIDGE=${OMEGA_BRIDGE_GIT:-$HOME/workspace/omega-bridge}
OMEGA=${OMEGA_MV_GIT:-$HOME/workspace/omega}
QUIET=${QUIET:-quiet-build}
BUILD=${BUILD:-$(mktemp -d "${TMPDIR:-/tmp}/si19orig1.XXXXXX")}
RECORDS=${RECORDS:-}
OMC_SHA=191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9
mkdir -p "$OUT" "$BUILD/deps"
SUM="$OUT/summary.txt"
: > "$SUM"
note() { echo "$*" | tee -a "$SUM"; }
fail() { note "SI19_ORIG1_RUN_FAIL $*"; exit 1; }

note "build_dir=$BUILD"
note "source_commit=$(git -C "$ROOT" rev-parse HEAD)"
note "host=$(uname -srm)"
git -C "$BRIDGE" show 7e1d0d61755ec1a0bbf20941a18379861b5b9d0d:native/ref32/add32.om > "$BUILD/deps/add32.om" || fail materialize add32
git -C "$BRIDGE" show 7e1d0d61755ec1a0bbf20941a18379861b5b9d0d:native/ref32/mul32.om > "$BUILD/deps/mul32.om" || fail materialize mul32
git -C "$OMEGA" show 3a995ba8dace538c1532c747d829ca6e4a73582d:native/ecosystem663/metadata.om > "$BUILD/deps/metadata.om" || fail materialize metadata
git -C "$ROOT" show c9782bfdcf1885082d1fa6c483473f2a514096ad:native/ecosystem6/affine.om > "$BUILD/deps/affine.om" || fail materialize affine
(cd "$BUILD/deps" && sha256sum -c "$ROOT/docs/si19/deps.sha256") > "$OUT/deps-check.txt" 2>&1 || fail dependency hash
set -- $(sha256sum "$OMC")
[ "$1" = "$OMC_SHA" ] || fail "compiler hash $1"
note "compiler_sha256=$1"

LIB="$BUILD/deps/add32.om $BUILD/deps/mul32.om $BUILD/deps/metadata.om $BUILD/deps/affine.om $ROOT/native/si19/fixture.om $ROOT/native/si19/world.om $ROOT/native/si19/omexp.om $ROOT/native/si19/omexpfix.om"

# build NAME CALLER [LIBRARY...]: compile one concatenated unit; status in OUT.
build() {
  name=$1; caller=$2; shift 2
  [ $# -gt 0 ] || set -- $LIB
  cat "$@" "$caller" > "$BUILD/$name.om"
  "$QUIET" "$OMC" < "$BUILD/$name.om" > "$BUILD/$name.elf" 2> "$OUT/$name.compile.stderr"
  rc=$?
  echo "$rc" > "$OUT/$name.compile.rc"
  [ "$rc" -eq 0 ] || return "$rc"
  head -c 4 "$BUILD/$name.elf" | od -An -c | grep -q 'E   L   F' || return 70
  chmod 700 "$BUILD/$name.elf"
}

ms() { date +%s%N; }

build omexp-test "$ROOT/tests/si19/omexp-test.om" || fail compile omexp-test
build omexp "$ROOT/native/si19/omexpmain.om" || fail compile omexp
if [ -x "$OMC2" ]; then
  "$QUIET" "$OMC2" < "$BUILD/omexp-test.om" > "$BUILD/omexp-test-r6.elf" 2> "$OUT/omexp-test-r6.compile.stderr"
  echo $? > "$OUT/omexp-test-r6.compile.rc"
  chmod 700 "$BUILD/omexp-test-r6.elf"
  note "r6_compiler_sha256=$(sha256sum "$OMC2" | cut -c1-64) compile_rc=$(cat "$OUT/omexp-test-r6.compile.rc")"
fi

t0=$(ms)
"$BUILD/omexp-test.elf" > "$OUT/omexp-test.stdout" 2> "$OUT/omexp-test.stderr"
rc=$?; t1=$(ms)
echo "$rc" > "$OUT/omexp-test.rc"
note "omexp-test rc=$rc wall_ms=$(( (t1 - t0) / 1000000 )) $(cat "$OUT/omexp-test.stdout")"
[ "$rc" -eq 0 ] || fail omexp-test
if [ -x "$BUILD/omexp-test-r6.elf" ]; then
  "$BUILD/omexp-test-r6.elf" > "$OUT/omexp-test-r6.stdout" 2> "$OUT/omexp-test-r6.stderr"
  echo $? > "$OUT/omexp-test-r6.rc"
  note "omexp-test-r6 rc=$(cat "$OUT/omexp-test-r6.rc") $(cat "$OUT/omexp-test-r6.stdout")"
fi

ALLOW=${ALLOW:-$ROOT/docs/si19/orig1-datasets.txt}
allow_block() { { cat "$ALLOW"; head -c 4096 /dev/zero; } | head -c 4096; }

# gen SEED COUNT NAME: the SYNTHETIC fixture to BUILD/NAME.bin
gen() {
  { printf 'G'; le8 "$1"; le8 "$2"; } | "$BUILD/omexp.elf" 3> "$BUILD/$3.bin" > "$OUT/gen-$3.stdout" 2> "$OUT/gen-$3.stderr"
  rc=$?
  echo "$rc" > "$OUT/gen-$3.rc"
  note "gen-$3 rc=$rc $(tr '\n' ' ' < "$OUT/gen-$3.stdout")"
  return "$rc"
}
# le8 N: 8 little-endian bytes of N using only POSIX tools.
le8() {
  v=$1; i=0
  while [ $i -lt 8 ]; do
    printf "\\$(printf '%03o' $((v % 256)))"
    v=$((v / 256)); i=$((i + 1))
  done
}

# inspect NAME FILE: admit a record file and print the baselines.
inspect() {
  t0=$(ms)
  { printf 'I'; allow_block; cat "$2"; } | "$BUILD/omexp.elf" > "$OUT/inspect-$1.stdout" 2> "$OUT/inspect-$1.stderr"
  rc=$?; t1=$(ms)
  echo "$rc" > "$OUT/inspect-$1.rc"
  note "inspect-$1 rc=$rc wall_ms=$(( (t1 - t0) / 1000000 ))"
  return "$rc"
}
# v2 NAME FILE: v2 admission (ORIG1-CONTRACT-v2.md) with counted refusal reasons and splits.
v2() {
  t0=$(ms)
  { printf 'V'; allow_block; cat "$2"; } | "$BUILD/omexp.elf" > "$OUT/v2-$1.stdout" 2> "$OUT/v2-$1.stderr"
  rc=$?; t1=$(ms)
  echo "$rc" > "$OUT/v2-$1.rc"
  note "v2-$1 rc=$rc wall_ms=$(( (t1 - t0) / 1000000 ))"
  return "$rc"
}

gen 7 780 fixture-7-780 || fail gen
inspect fixture "$BUILD/fixture-7-780.bin" || fail inspect fixture
cat "$OUT/inspect-fixture.stdout" | tee -a "$SUM"
v2 fixture "$BUILD/fixture-7-780.bin" || fail v2 fixture
cat "$OUT/v2-fixture.stdout" | tee -a "$SUM"
printf 'S' | "$BUILD/omexp.elf" > "$OUT/splits.txt" 2> "$OUT/splits.stderr" || fail splits
cmp -s "$OUT/splits.txt" "$ROOT/docs/si19/orig1-v2-splits.txt" || fail "split table differs from docs/si19/orig1-v2-splits.txt"
note "v2_split_table=$(sha256sum "$OUT/splits.txt" | cut -c1-64) families=$(wc -l < "$OUT/splits.txt") identical_to_docs=yes"

# Refusals through the process path: a foreign file and a damaged record.
head -c 256 /dev/zero > "$BUILD/zero.bin"
inspect zero "$BUILD/zero.bin"
[ "$?" -eq 66 ] || fail zero file admitted
{ cat "$BUILD/fixture-7-780.bin"; head -c 256 /dev/zero; } > "$BUILD/unlisted.bin"
inspect unlisted "$BUILD/unlisted.bin"
[ "$?" -eq 66 ] || fail unlisted damaged file admitted
cp "$BUILD/fixture-7-780.bin" "$BUILD/forged.bin"
printf '\001' | dd of="$BUILD/forged.bin" bs=1 seek=248 conv=notrunc 2> /dev/null
inspect forged "$BUILD/forged.bin"
[ "$?" -eq 66 ] || fail forged mismatch admitted
printf '\001' | dd of="$BUILD/forged.bin" bs=1 seek=248 conv=notrunc 2> /dev/null
cp "$BUILD/fixture-7-780.bin" "$BUILD/unauth.bin"
printf '\001' | dd of="$BUILD/unauth.bin" bs=1 seek=88 conv=notrunc 2> /dev/null
inspect unauth "$BUILD/unauth.bin"
[ "$?" -eq 67 ] || fail unlisted hash admitted
cat "$OUT/inspect-zero.stdout" "$OUT/inspect-unlisted.stdout" "$OUT/inspect-forged.stdout" "$OUT/inspect-unauth.stdout" | tee -a "$SUM"

if [ -n "$RECORDS" ]; then
  note "owner_records=$RECORDS sha256=$(sha256sum "$RECORDS" | cut -c1-64) bytes=$(wc -c < "$RECORDS")"
  inspect owner "$RECORDS" || fail inspect owner
  cat "$OUT/inspect-owner.stdout" | tee -a "$SUM"
  v2 owner "$RECORDS" || fail v2 owner
  cat "$OUT/v2-owner.stdout" | tee -a "$SUM"
else
  note "owner_records=NOT_RUN (RECORDS unset)"
fi

# Leakage audit (mode A, local analysis): AUDIT_A = the earlier file, AUDIT_B = the later one.
if [ -n "${AUDIT_A:-}" ] && [ -n "${AUDIT_B:-}" ]; then
  na=$(( $(wc -c < "$AUDIT_A") / 256 ))
  { printf 'A'; le8 "$na"; cat "$AUDIT_A" "$AUDIT_B"; } | "$BUILD/omexp.elf" > "$OUT/audit.stdout" 2> "$OUT/audit.stderr"
  rc=$?; echo "$rc" > "$OUT/audit.rc"
  note "audit rc=$rc a_sha256=$(sha256sum "$AUDIT_A" | cut -c1-64) b_sha256=$(sha256sum "$AUDIT_B" | cut -c1-64)"
  cat "$OUT/audit.stdout" | tee -a "$SUM"
  [ "$rc" -eq 0 ] || fail audit
else
  note "audit=NOT_RUN (AUDIT_A/AUDIT_B unset)"
fi

# Fault injection: each altered unit must compile and fail the unchanged test.
i=0
while IFS='|' read -r file orig alt; do
  [ -n "$file" ] || continue
  i=$((i + 1))
  src="$ROOT/native/si19/$file"
  n=$(grep -c -x -F -- "$orig" "$src")
  [ "$n" -eq 1 ] || fail "mutant$i matches $n"
  mdir="$BUILD/mutant$i"; mkdir -p "$mdir"
  awk -v o="$orig" -v a="$alt" '$0 == o { print a; next } { print }' "$src" > "$mdir/$file"
  mlib=""
  for f in $LIB; do
    if [ "$(basename "$f")" = "$file" ]; then mlib="$mlib $mdir/$file"; else mlib="$mlib $f"; fi
  done
  build "mutant$i" "$ROOT/tests/si19/omexp-test.om" $mlib || fail "mutant$i compile"
  "$BUILD/mutant$i.elf" > "$OUT/mutant$i.stdout" 2> "$OUT/mutant$i.stderr"
  rc=$?
  echo "$rc" > "$OUT/mutant$i.rc"
  note "mutant$i $file rc=$rc $(cat "$OUT/mutant$i.stdout") :: $alt"
  [ "$rc" -eq 1 ] || fail "mutant$i survived"
done <<'EOF'
omexp.om|if ne oxword r 31 0|if ne oxword r 31 1
omexp.om|if ne oxword r add 24 a 1|if ne oxword r add 24 a 0
omexp.om|ret -11|ret 0
omexp.om|if le oxseq r oxseq sub r 1|if lt oxseq r oxseq sub r 1
omexp.om|if and eq b 82 eq k 7|if and eq b 82 eq k 8
omexp.om|put XF 3 long shr mul lm ln 8|put XF 3 long shr mul lm lm 8
omexp.om|if ne oxword r 12 31|if eq oxword r 12 0
omexp.om|d = true|d = false
omexp.om|if or gt oxfam r 5 gt oxcore r 1|if or gt oxfam r 5 gt oxcore r 2
omexp.om|ret oxhspl oxfl oxm r oxfl oxn r|ret rem oxfam r 3
omexp.om|if ge oxword r add 16 a 1099511627776|if ge oxword r add 16 a 1099511627777
omexp.om|ret int -11|ret 0
omexp.om|if ge oxseq sub r 1 oxseq r|if gt oxseq sub r 1 oxseq r
omexp.om|ret eq oxpress q p|ret true
omexp.om|ret and eq oxcore q oxcore r eq oxpress q oxpress r|ret eq oxcore q oxcore r
omexp.om|if eq oxbest q oxbest r|if ne oxbest q oxbest r
EOF

(cd "$OUT" && sha256sum -- *.stdout *.rc *.compile.stderr deps-check.txt) > "$OUT/raw.sha256"
(cd "$BUILD" && sha256sum -- *.bin omexp.elf omexp-test.elf omexp.om omexp-test.om) > "$OUT/build.sha256"
note "SI19_ORIG1_RUN_PASS"

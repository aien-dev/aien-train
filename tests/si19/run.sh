#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# aien-train#19 host glue (POSIX sh, disclosed): materialize the pinned sources,
# verify SHA-256, concatenate, compile with the pinned seeded compiler, run, and
# record every status. All training, checks and verdicts are in the .om units.
# Host CPU only. No GPU, network, service or model process is touched.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$ROOT/evidence/SI19/run}
OMC=${OMC:-$HOME/workspace/omega-ecosystem660-bridge/build/bootstrap/omc}
BRIDGE=${OMEGA_BRIDGE_GIT:-$HOME/workspace/omega-bridge}
OMEGA=${OMEGA_MV_GIT:-$HOME/workspace/omega-ecosystem663}
QUIET=${QUIET:-quiet-build}
BUILD=${BUILD:-$(mktemp -d "${TMPDIR:-/tmp}/si19.XXXXXX")}
OMC_SHA=191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9
mkdir -p "$OUT" "$BUILD/deps"
SUM="$OUT/summary.txt"
: > "$SUM"
note() { echo "$*" | tee -a "$SUM"; }
fail() { note "SI19_RUN_FAIL $*"; exit 1; }

note "build_dir=$BUILD"
note "source_commit=$(git -C "$ROOT" rev-parse HEAD)"
note "host=$(uname -srm)"
git -C "$BRIDGE" show 7e1d0d61755ec1a0bbf20941a18379861b5b9d0d:native/ref32/add32.om > "$BUILD/deps/add32.om" || fail materialize add32
git -C "$BRIDGE" show 7e1d0d61755ec1a0bbf20941a18379861b5b9d0d:native/ref32/mul32.om > "$BUILD/deps/mul32.om" || fail materialize mul32
git -C "$OMEGA" show 3a995ba8dace538c1532c747d829ca6e4a73582d:native/ecosystem663/metadata.om > "$BUILD/deps/metadata.om" || fail materialize metadata
git -C "$ROOT" show c7642d96c0b84b6d176dbb2dc8558b536cabe729:native/ecosystem6/affine.om > "$BUILD/deps/affine.om" || fail materialize affine
(cd "$BUILD/deps" && sha256sum -c "$ROOT/docs/si19/deps.sha256") > "$OUT/deps-check.txt" 2>&1 || fail dependency hash
set -- $(sha256sum "$OMC")
[ "$1" = "$OMC_SHA" ] || fail "compiler hash $1"
note "compiler_sha256=$1"

LIB="$BUILD/deps/add32.om $BUILD/deps/mul32.om $BUILD/deps/metadata.om $BUILD/deps/affine.om $ROOT/native/si19/fixture.om $ROOT/native/si19/world.om"

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

build world-test "$ROOT/tests/si19/world-test.om" || fail compile world-test
build world "$ROOT/native/si19/main.om" || fail compile world
readelf -l -d "$BUILD/world.elf" > "$OUT/world.readelf.txt" 2>&1

t0=$(ms)
"$BUILD/world-test.elf" > "$OUT/world-test.stdout" 2> "$OUT/world-test.stderr"
rc=$?; t1=$(ms)
echo "$rc" > "$OUT/world-test.rc"
note "world-test rc=$rc wall_ms=$(( (t1 - t0) / 1000000 )) $(cat "$OUT/world-test.stdout")"
[ "$rc" -eq 0 ] || fail world-test

# run MODE NAME [IMAGE...]: mode byte then images on stdin; fd 3 receives an image.
run() {
  mode=$1; name=$2; shift 2
  t0=$(ms)
  { printf '%s' "$mode"; [ $# -gt 0 ] && cat "$@"; } | "$BUILD/world.elf" 3> "$BUILD/$name.img" > "$OUT/$name.stdout" 2> "$OUT/$name.stderr"
  rc=$?; t1=$(ms)
  echo "$rc" > "$OUT/$name.rc"
  note "$name mode=$mode rc=$rc wall_ms=$(( (t1 - t0) / 1000000 ))"
  return "$rc"
}

run U uninterrupted || fail uninterrupted
run P partial100 || fail partial100
run R resume100 "$BUILD/partial100.img" || fail resume100
run Q partial300 || fail partial300
run R resume300 "$BUILD/partial300.img" || fail resume300
run C compare100 "$BUILD/uninterrupted.img" "$BUILD/resume100.img" || fail compare100
run C compare300 "$BUILD/uninterrupted.img" "$BUILD/resume300.img" || fail compare300
cat "$OUT/compare100.stdout" "$OUT/compare300.stdout" | tee -a "$SUM"
tail -n +3 "$OUT/resume100.stdout" > "$BUILD/r100.txt"
tail -n +2 "$OUT/uninterrupted.stdout" > "$BUILD/u.txt"
cmp -s "$BUILD/u.txt" "$BUILD/r100.txt" || fail resumed report differs
note "resumed_report_identical=yes"

# Negative controls on the process path.
run C compare_partial "$BUILD/uninterrupted.img" "$BUILD/partial100.img"
[ "$?" -eq 1 ] || fail compare of different images did not refuse
cp "$BUILD/partial100.img" "$BUILD/damaged.img"
printf '\001' | dd of="$BUILD/damaged.img" bs=1 seek=20000 conv=notrunc 2> /dev/null
run R resume_damaged "$BUILD/damaged.img"
[ "$?" -eq 65 ] || fail damaged image admitted
head -c 51700 "$BUILD/partial100.img" > "$BUILD/short.img"
run R resume_short "$BUILD/short.img"
[ "$?" -eq 65 ] || fail truncated image admitted
cat "$OUT/resume_damaged.stdout" "$OUT/resume_short.stdout" | tee -a "$SUM"

# Fault injection: each altered unit must compile and fail the unchanged test.
# FILE|ORIGINAL LINE|ALTERED LINE; the original must occur exactly once.
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
  build "mutant$i" "$ROOT/tests/si19/world-test.om" $mlib || fail "mutant$i compile"
  "$BUILD/mutant$i.elf" > "$OUT/mutant$i.stdout" 2> "$OUT/mutant$i.stderr"
  rc=$?
  echo "$rc" > "$OUT/mutant$i.rc"
  note "mutant$i $file rc=$rc $(cat "$OUT/mutant$i.stdout") :: $alt"
  [ "$rc" -eq 1 ] || fail "mutant$i survived"
done <<'EOF'
fixture.om|put FS add b 2 get FT 3|put FS add b 2 get FT 0
world.om|put AP c wmbits get TL add t 4 0|put AP c wmbits get TL add t 5 0
world.om|if not affinite ckget add add base 96 mul i 8|if not affinite ckget add add base 104 mul i 8
world.om|put FS i wsdec ckget add add base 51232 mul i 8|put FS i 0
world.om|wmuse arm c|wmuse arm a
world.om|wmrow 0 a ta|wmrow 0 a a
EOF

(cd "$OUT" && sha256sum -- *.stdout *.rc *.compile.stderr deps-check.txt world.readelf.txt) > "$OUT/raw.sha256"
(cd "$BUILD" && sha256sum -- *.img world.elf world-test.elf world.om world-test.om) > "$OUT/build.sha256"
note "SI19_RUN_PASS"

#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# aien-train#25 ORIG-2 host glue (POSIX sh, disclosed): materialize the pinned sources,
# verify SHA-256, concatenate, compile with the pinned seeded compiler, run the .om
# tests and the fixture training/control/resume/refusal/evaluation runs through the
# process path, inject faults, and record every status. Every judgment about numbers
# is made inside the .om units; this file only sequences and compares bytes/exit codes.
# Host CPU only. No GPU, network, service or model process is touched.
# RECORDS=<omexp-v1 file> ALLOW=<allow-list file>: run the same train/eval steps over a
# given record file (real-record run); the SYNTHETIC fixture and the mutants are skipped.
# V2=1 (ORIG1-CONTRACT-v2.md; benchmarks-mv manifests/prereg/orig3_selector_v2.json): after the
# tests, only the v2 steps on RECORDS (or the fixture): v2 admission (config flag 8), MLP
# candidate and the shuffled/untrained controls at the frozen lr 2^-8, seed 19024, at most
# 64 epochs, patience 16 on VALIDATION; then mode E per image on VALIDATION + HELD-OUT.
V2=${V2:-0}
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$ROOT/evidence/SI19-ORIG2/fixture-run}
OMC=${OMC:-/tmp/omc-191b776e}
OMC2=${OMC2:-$HOME/workspace/omega-seedqual-20261025/seed/out/omc}
BRIDGE=${OMEGA_BRIDGE_GIT:-$HOME/workspace/omega-bridge}
OMEGA=${OMEGA_MV_GIT:-$HOME/workspace/omega}
QUIET=${QUIET:-quiet-build}
BUILD=${BUILD:-$(mktemp -d "${TMPDIR:-/tmp}/si19sel.XXXXXX")}
RECORDS=${RECORDS:-}
ALLOW=${ALLOW:-}
SEED=${SEED:-19024}
FIXSEED=${FIXSEED:-4242}
FIXCOUNT=${FIXCOUNT:-400}
LR_AFF=${LR_AFF:-8}
LR_MLP=${LR_MLP:-8}
OMC_SHA=191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9
OMC2_SHA=734f0a39319fbfa930bcc62cb30dbf06d47d78e126372b74f5baf478e6126b61
mkdir -p "$OUT" "$BUILD/deps"
SUM="$OUT/summary.txt"
: > "$SUM"
note() { echo "$*" | tee -a "$SUM"; }
fail() { note "SI19_SELECT_RUN_FAIL $*"; exit 1; }
command -v "$QUIET" > /dev/null 2>&1 || QUIET=env
TIMEV=""
[ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v"

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
if [ -x "$OMC2" ]; then
  set -- $(sha256sum "$OMC2")
  [ "$1" = "$OMC2_SHA" ] || fail "second compiler hash $1"
  note "second_compiler_sha256=$1"
else
  OMC2=""
  note "second_compiler=NOT_RUN (absent)"
fi

N="$ROOT/native/si19"
LIB="$BUILD/deps/add32.om $BUILD/deps/mul32.om $BUILD/deps/metadata.om $BUILD/deps/affine.om $N/fixture.om $N/world.om $N/omexp.om $N/omexpfix.om $N/select.om"

# build NAME CALLER [LIBRARY...]: compile one concatenated unit with the pinned compiler;
# the second compiler must also accept it (NAME.r6.elf, compile-only check).
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
  if [ -n "$OMC2" ]; then
    "$QUIET" "$OMC2" < "$BUILD/$name.om" > "$BUILD/$name.r6.elf" 2> "$OUT/$name.r6.compile.stderr"
    rc=$?
    echo "$rc" > "$OUT/$name.r6.compile.rc"
    [ "$rc" -eq 0 ] || return "$rc"
    chmod 700 "$BUILD/$name.r6.elf"
  fi
}

ms() { date +%s%N; }

# le64 N: 8 little-endian bytes of N. cfg W...: config words. allow LINE: 4096-byte block.
le64() { n=$1; i=0; while [ "$i" -lt 8 ]; do b=$((n % 256)); n=$((n / 256)); printf "\\$(printf '%03o' "$b")"; i=$((i + 1)); done; }
cfg() { for v in "$@"; do le64 "$v"; done; }
allowblock() { { cat "$1"; head -c 4096 /dev/zero; } | head -c 4096; }

build select-test "$ROOT/tests/si19/select-test.om" || fail compile select-test
build selmain "$N/selmain.om" || fail compile selmain
readelf -l -d "$BUILD/selmain.elf" > "$OUT/selmain.readelf.txt" 2>&1
note "second_compiler_units=$( [ -n "$OMC2" ] && echo 'select-test selmain compiled' || echo NOT_RUN )"

t0=$(ms)
$TIMEV "$BUILD/select-test.elf" > "$OUT/select-test.stdout" 2> "$OUT/select-test.stderr"
rc=$?; t1=$(ms)
echo "$rc" > "$OUT/select-test.rc"
note "select-test rc=$rc wall_ms=$(( (t1 - t0) / 1000000 )) $(cat "$OUT/select-test.stdout") $(grep 'Maximum resident' "$OUT/select-test.stderr" 2>/dev/null | sed 's/^[[:space:]]*//')"
[ "$rc" -eq 0 ] || fail select-test
if [ -n "$OMC2" ]; then
  "$BUILD/select-test.r6.elf" > "$OUT/select-test.r6.stdout" 2> "$OUT/select-test.r6.stderr"
  rc=$?
  echo "$rc" > "$OUT/select-test.r6.rc"
  note "select-test(r6 compiler) rc=$rc $(cat "$OUT/select-test.r6.stdout")"
  [ "$rc" -eq 0 ] || fail select-test r6
fi

# run NAME MODE "CFG WORDS" [IMAGE]: stdin = mode, config, allow block, image, records;
# fd 3 receives an image, fd 4 the per-record lines. Exit code is returned.
run() {
  name=$1; mode=$2; words=$3; img=${4:-}
  t0=$(ms)
  { printf '%s' "$mode"; cfg $words; allowblock "$ALLOWFILE"; [ -n "$img" ] && cat "$img"; cat "$RECFILE"; } \
    | $TIMEV "$BUILD/selmain.elf" 3> "$BUILD/$name.img" 4> "$OUT/$name.records" > "$OUT/$name.stdout" 2> "$OUT/$name.stderr"
  rc=$?; t1=$(ms)
  echo "$rc" > "$OUT/$name.rc"
  note "$name mode=$mode rc=$rc wall_ms=$(( (t1 - t0) / 1000000 )) epochs=$(sed -n 's/^epochs=//p' "$OUT/$name.stdout") updates=$(sed -n 's/^updates=//p' "$OUT/$name.stdout") dev_loss_dec=$(sed -n 's/^dev_loss_dec=//p' "$OUT/$name.stdout") $(grep 'Maximum resident' "$OUT/$name.stderr" 2>/dev/null | sed 's/^[[:space:]]*//')"
  return "$rc"
}
kv() { sed -n "s/^$2=//p" "$OUT/$1.stdout"; }

if [ -n "$RECORDS" ]; then
  [ -f "$RECORDS" ] || fail "RECORDS file $RECORDS"
  [ -f "$ALLOW" ] || fail "ALLOW file $ALLOW"
  RECFILE=$RECORDS; ALLOWFILE=$ALLOW
  note "records_source=$RECORDS (real-record run; development-only)"
else
  { printf 'G'; cfg "$FIXSEED" "$FIXCOUNT"; } | "$BUILD/selmain.elf" 3> "$BUILD/fixture.rb" > "$OUT/fixture.stdout" 2> "$OUT/fixture.stderr"
  rc=$?; echo "$rc" > "$OUT/fixture.rc"
  [ "$rc" -eq 0 ] || fail fixture generation
  FNV=$(sed -n 's/^fnv=//p' "$OUT/fixture.stdout")
  printf '%s SYNTHETIC si19-orig2-fixture-seed%s-n%s\n' "$FNV" "$FIXSEED" "$FIXCOUNT" > "$BUILD/allow.txt"
  RECFILE="$BUILD/fixture.rb"; ALLOWFILE="$BUILD/allow.txt"
  note "records_source=SYNTHETIC fixture seed=$FIXSEED count=$FIXCOUNT fnv=$FNV"
fi
note "records_sha256=$(sha256sum "$RECFILE" | cut -d' ' -f1)"

if [ "$V2" = 1 ]; then
  # Config: kind seed lr maxep patience ctrl stop flags; flags 14 = v2 + VALIDATION + HELD-OUT.
  run v2_train_mlp T "2 $SEED 8 64 16 0 0 14" || fail v2_train_mlp
  run v2_ctrl_shuffled T "2 $SEED 8 64 16 1 0 14" || fail v2_ctrl_shuffled
  run v2_ctrl_untrained T "2 $SEED 8 0 16 0 0 14" || fail v2_ctrl_untrained
  [ "$(kv v2_ctrl_untrained epochs)" = 0 ] || fail untrained control trained
  [ "$(kv v2_ctrl_untrained final_params_fnv)" = "$(kv v2_train_mlp init_fnv)" ] || fail untrained control init differs
  run v2_eval_mlp E "2 $SEED 8 64 16 0 0 14" "$BUILD/v2_train_mlp.img" || fail v2_eval_mlp
  run v2_eval_shuffled E "2 $SEED 8 64 16 1 0 14" "$BUILD/v2_ctrl_shuffled.img" || fail v2_eval_shuffled
  run v2_eval_untrained E "2 $SEED 8 0 16 0 0 14" "$BUILD/v2_ctrl_untrained.img" || fail v2_eval_untrained
  # A v1 config over a v2 image is refused (68): the split rule travels with the image.
  run v2_cross_v1 E "2 $SEED 8 64 16 0 0 6" "$BUILD/v2_train_mlp.img"
  [ "$?" -eq 68 ] || fail v1 config accepted a v2 image
  grep '^v2_admitted=' "$OUT/v2_eval_mlp.stdout" | tee -a "$SUM"
  grep '^eval part=' "$OUT/v2_eval_mlp.stdout" "$OUT/v2_eval_shuffled.stdout" "$OUT/v2_eval_untrained.stdout" | tee -a "$SUM"
  for f in v2_train_mlp v2_ctrl_shuffled v2_ctrl_untrained; do
    note "image $f sha256=$(sha256sum "$BUILD/$f.img" | cut -c1-64) bytes=$(wc -c < "$BUILD/$f.img")"
  done
  (cd "$OUT" && sha256sum -- *.stdout *.rc *.compile.stderr *.records deps-check.txt selmain.readelf.txt) > "$OUT/raw.sha256"
  (cd "$BUILD" && sha256sum -- *.img *.elf *.om 2> /dev/null) > "$OUT/build.sha256"
  note "SI19_SELECT_V2_RUN_PASS"
  exit 0
fi

# Learning-rate sweep (contract s.8: 2^-8, 2^-10, 2^-12 judged on DEV loss only; DEV loss
# sums are recorded here as evidence, the choice LR_AFF/LR_MLP is the operator's input).
for lr in 8 10 12; do
  run "sweep_affine_lr$lr" T "1 $SEED $lr 64 16 0 0 2" || fail "sweep_affine_lr$lr"
  run "sweep_mlp_lr$lr" T "2 $SEED $lr 64 16 0 0 2" || fail "sweep_mlp_lr$lr"
done
note "lr_choice affine=2^-$LR_AFF mlp=2^-$LR_MLP"

# Training, controls, partial + resume, evaluation. Config: kind seed lr maxep patience ctrl stop flags.
run train_affine T "1 $SEED $LR_AFF 64 16 0 0 7" || fail train_affine
run train_mlp T "2 $SEED $LR_MLP 64 16 0 0 7" || fail train_mlp
run ctrl_shuffled T "2 $SEED $LR_MLP 64 16 1 0 2" || fail ctrl_shuffled
run ctrl_noupdate T "2 $SEED 0 64 16 2 0 2" || fail ctrl_noupdate
[ "$(kv ctrl_noupdate final_params_fnv)" = "$(kv ctrl_noupdate init_fnv)" ] || fail no-update control changed params
[ "$(kv ctrl_noupdate updates)" -gt 0 ] || fail no-update control made no updates
note "ctrl_noupdate params_bit_identical=yes updates=$(kv ctrl_noupdate updates)"
run ctrl_untrained T "2 $SEED $LR_MLP 0 16 0 0 7" || fail ctrl_untrained
[ "$(kv ctrl_untrained epochs)" = 0 ] || fail untrained control trained
[ "$(kv ctrl_untrained final_params_fnv)" = "$(kv train_mlp init_fnv)" ] || fail untrained control init differs
for cut in 1 3; do
  run "partial$cut" T "2 $SEED $LR_MLP 64 16 0 $cut 0" || fail "partial$cut"
  run "resume$cut" R "2 $SEED $LR_MLP 64 16 0 0 7" "$BUILD/partial$cut.img" || fail "resume$cut"
  { printf 'C'; cat "$BUILD/train_mlp.img" "$BUILD/resume$cut.img"; } | "$BUILD/selmain.elf" > "$OUT/compare$cut.stdout" 2> "$OUT/compare$cut.stderr"
  rc=$?; echo "$rc" > "$OUT/compare$cut.rc"
  note "compare$cut rc=$rc $(cat "$OUT/compare$cut.stdout")"
  [ "$rc" -eq 0 ] || fail "resume$cut image differs"
  cmp -s "$OUT/train_mlp.stdout" "$OUT/resume$cut.stdout" || fail "resume$cut report differs"
  cmp -s "$OUT/train_mlp.records" "$OUT/resume$cut.records" || fail "resume$cut records differ"
done
note "resumed_report_identical=yes"
run eval_mlp E "2 $SEED $LR_MLP 64 16 0 0 6" "$BUILD/train_mlp.img" || fail eval_mlp
run eval_affine E "1 $SEED $LR_AFF 64 16 0 0 6" "$BUILD/train_affine.img" || fail eval_affine
run eval_shuffled E "2 $SEED $LR_MLP 64 16 1 0 6" "$BUILD/ctrl_shuffled.img" || fail eval_shuffled
run eval_untrained E "2 $SEED $LR_MLP 0 16 0 0 6" "$BUILD/ctrl_untrained.img" || fail eval_untrained
grep '^eval part=' "$OUT/eval_mlp.stdout" "$OUT/eval_affine.stdout" "$OUT/eval_shuffled.stdout" "$OUT/eval_untrained.stdout" | tee -a "$SUM"

# Negative controls on the process path.
{ printf 'C'; cat "$BUILD/train_mlp.img" "$BUILD/partial1.img"; } | "$BUILD/selmain.elf" > "$OUT/compare_partial.stdout" 2>&1
[ "$?" -eq 1 ] || fail compare of different images did not refuse
cp "$BUILD/partial1.img" "$BUILD/damaged.img"
printf '\001' | dd of="$BUILD/damaged.img" bs=1 seek=5000 conv=notrunc 2> /dev/null
run resume_damaged R "2 $SEED $LR_MLP 64 16 0 0 0" "$BUILD/damaged.img"
[ "$?" -eq 65 ] || fail damaged image admitted
# An image cut short would shift the record stream (the image has a fixed length), so the
# truncated image is zero-padded back to the fixed length: the checksum must refuse it.
{ head -c 11000 "$BUILD/partial1.img"; head -c 560 /dev/zero; } > "$BUILD/short.img"
run resume_short R "2 $SEED $LR_MLP 64 16 0 0 0" "$BUILD/short.img"
[ "$?" -eq 65 ] || fail truncated image admitted
cat "$OUT/compare_partial.stdout" "$OUT/resume_damaged.stdout" "$OUT/resume_short.stdout" | tee -a "$SUM"
run bad_config T "3 $SEED 10 64 16 0 0 0"
[ "$?" -eq 68 ] || fail invalid config accepted
GOODALLOW=$ALLOWFILE
sed 's/ SYNTHETIC / REVOKED /; s/ AUTHORIZED / REVOKED /' "$GOODALLOW" > "$BUILD/revoked.txt"
ALLOWFILE="$BUILD/revoked.txt"
run revoked T "2 $SEED $LR_MLP 1 16 0 0 0"
[ "$?" -eq 67 ] || fail revoked dataset admitted
printf '1 SYNTHETIC other\n' > "$BUILD/absent.txt"
ALLOWFILE="$BUILD/absent.txt"
run unlisted T "2 $SEED $LR_MLP 1 16 0 0 0"
[ "$?" -eq 67 ] || fail unlisted dataset admitted
ALLOWFILE=$GOODALLOW
cat "$OUT/revoked.stdout" "$OUT/unlisted.stdout" | tee -a "$SUM"

if [ -z "$RECORDS" ]; then
  # Fault injection: each altered unit must compile and fail the unchanged test.
  # FILE|ORIGINAL LINE|ALTERED LINE; the original must occur exactly once.
  i=0
  while IFS='|' read -r file orig alt; do
    [ -n "$file" ] || continue
    i=$((i + 1))
    src="$N/$file"
    n=$(grep -c -x -F -- "$orig" "$src")
    [ "$n" -eq 1 ] || fail "mutant$i matches $n"
    mdir="$BUILD/mutant$i"; mkdir -p "$mdir"
    awk -v o="$orig" -v a="$alt" '$0 == o { print a; next } { print }' "$src" > "$mdir/$file"
    mlib=""
    for f in $LIB; do
      if [ "$(basename "$f")" = "$file" ]; then mlib="$mlib $mdir/$file"; else mlib="$mlib $f"; fi
    done
    build "mutant$i" "$ROOT/tests/si19/select-test.om" $mlib || fail "mutant$i compile"
    "$BUILD/mutant$i.elf" > "$OUT/mutant$i.stdout" 2> "$OUT/mutant$i.stderr"
    rc=$?
    echo "$rc" > "$OUT/mutant$i.rc"
    note "mutant$i $file rc=$rc $(cat "$OUT/mutant$i.stdout") :: $alt"
    [ "$rc" -eq 1 ] || fail "mutant$i survived"
  done <<'EOF'
select.om|d = add32 get SZ 20 xor y 2147483648|d = add32 get SZ 20 y
select.om|ret and eq and x 2147483648 0 ne x 0|ret eq and x 2147483648 0
select.om|v = add32 get SP i xor mul32 lr get SG i 2147483648|v = add32 get SP i mul32 lr get SG i
select.om|ret eq siget add base 104 get RC 2|ret true
select.om|put SS 5 c|put SS 5 0
select.om|put SX p get SX k|put SX p p
select.om|if ne oxpart r 0|if gt oxpart r 1
select.om|if gt siget add base 176 1|if gt siget add base 176 2
EOF
fi

(cd "$OUT" && sha256sum -- *.stdout *.rc *.compile.stderr *.records deps-check.txt selmain.readelf.txt) > "$OUT/raw.sha256"
(cd "$BUILD" && sha256sum -- *.img *.elf *.om allow.txt fixture.rb 2> /dev/null) > "$OUT/build.sha256"
note "SI19_SELECT_RUN_PASS"

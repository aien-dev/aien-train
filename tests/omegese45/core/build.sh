#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Build the omegese45 pipeline binaries (aien-train #45, pipeline lane).
#   OMC=<omc bf54b19f> OMEGA=<omega-mv checkout with bf54b19f> OUT=<dir> sh tests/omegese45/core/build.sh [MODEL.om]
# Writes $OUT/oe45gen (generator), $OUT/oe45core (CLI), $OUT/oe45test (test driver) and
# $OUT/oe45mut (fault injector). MODEL.om (default native/omegese45/core/model.om) lets the
# suite build mutants. Host glue only: the pinned SHA unit comes from its exact Git object
# (native/omegese45/core/dependencies.json, sha256 checked); its `export fn` lines become
# `fn` (program mode refuses export), and OM-2 requires it first in the unit.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
: "${OMC:?}" "${OMEGA:?}" "${OUT:?}"
mkdir -p "$OUT"
N=$ROOT/native/omegese45
MODEL=${1:-$N/core/model.om}
DEPS=$N/core/dependencies.json
[ "$(sha256sum "$OMC" | cut -d' ' -f1)" = "$(jq -r .compiler.binary_sha256 "$DEPS")" ] || { echo "BUILD_FAIL omc-sha256"; exit 1; }
commit=$(jq -r '.dependencies[0].commit' "$DEPS")
path=$(jq -r '.dependencies[0].path' "$DEPS")
sum=$(jq -r '.dependencies[0].sha256' "$DEPS")
git -C "$OMEGA" show "$commit:$path" > "$OUT/sha256-om2.om" || { echo "BUILD_FAIL sha-unit-missing"; exit 1; }
echo "$sum  $OUT/sha256-om2.om" | sha256sum -c --status || { echo "BUILD_FAIL sha-unit-sha256"; exit 1; }
sed 's/^export fn /fn /' "$OUT/sha256-om2.om" > "$OUT/sha-prog.om"
cat "$N/vocab.om" "$N/msg.om" "$N/data/gen.om" > "$OUT/gen.unit.om"
cat "$OUT/sha-prog.om" "$N/vocab.om" "$N/msg.om" "$MODEL" "$N/core/ckpt.om" \
    "$N/core/run.om" "$N/core/cli.om" > "$OUT/core.unit.om"
cat "$OUT/sha-prog.om" "$N/vocab.om" "$N/msg.om" "$MODEL" "$N/core/ckpt.om" \
    "$N/core/run.om" "$ROOT/tests/omegese45/core/main.om" > "$OUT/test.unit.om"
cp "$ROOT/tests/omegese45/core/mutate.om" "$OUT/mut.unit.om"
for u in gen core test mut; do
    "$OMC" < "$OUT/$u.unit.om" > "$OUT/oe45$u" 2> "$OUT/$u.compile.stderr"
    rc=$?
    [ "$rc" = 0 ] || { echo "BUILD_FAIL $u rc=$rc $(cat "$OUT/$u.compile.stderr")"; exit 1; }
    [ "$(od -An -tx1 -N4 "$OUT/oe45$u" | tr -d ' ')" = 7f454c46 ] || { echo "BUILD_FAIL $u not-elf"; exit 1; }
    chmod 700 "$OUT/oe45$u"
done
echo BUILD_OK

#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Build the omegese45 pipeline binaries (aien-train #45, pipeline lane).
#   OMC=<omc bf54b19f> OMEGA=<omega-mv checkout with bf54b19f> OUT=<dir> sh tests/omegese45/core/build.sh
# Host glue only: the pinned SHA unit comes from its exact Git object (sha256 checked);
# its four `export fn` lines become `fn` (program mode refuses export, error 9), the
# aien-protocols-mv specs/aienet/jobpath/build.sh pattern. OM-2 requires it first.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
: "${OMC:?}" "${OMEGA:?}" "${OUT:?}"
mkdir -p "$OUT"
N=$ROOT/native/omegese45
SHA_COMMIT=bf54b19fa590e8ae925c0e80815516dbf41e7326
SHA_SUM=dbf002dd6aa36585632b9d01d98ac3f0bd2fa0f0475082940de53638847994bb
[ "$(sha256sum "$OMC" | cut -d' ' -f1)" = 62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b ] || { echo "BUILD_FAIL omc-sha256"; exit 1; }
git -C "$OMEGA" show "$SHA_COMMIT:om/sha256/sha256-om2.om" > "$OUT/sha256-om2.om" || { echo "BUILD_FAIL sha-unit-missing"; exit 1; }
echo "$SHA_SUM  $OUT/sha256-om2.om" | sha256sum -c --status || { echo "BUILD_FAIL sha-unit-sha256"; exit 1; }
sed 's/^export fn /fn /' "$OUT/sha256-om2.om" > "$OUT/sha-prog.om"
cat "$N/vocab.om" "$N/msg.om" "$N/data/gen.om" > "$OUT/gen.unit.om"
cat "$OUT/sha-prog.om" "$N/vocab.om" "$N/msg.om" "$N/core/model.om" "$N/core/ckpt.om" \
    "$N/core/run.om" "$N/core/cli.om" > "$OUT/core.unit.om"
for u in gen core; do
    "$OMC" < "$OUT/$u.unit.om" > "$OUT/oe45$u" 2> "$OUT/$u.compile.stderr"
    rc=$?
    [ "$rc" = 0 ] || { echo "BUILD_FAIL $u rc=$rc $(cat "$OUT/$u.compile.stderr")"; exit 1; }
    chmod 700 "$OUT/oe45$u"
done
echo BUILD_OK

#!/bin/sh
# OE45 demo runtime setup: existing Aienet-0 components only (no new executor,
# receipt issuer, scheduler, judge or ledger).
#
# usage: tests/omegese45/demo/setup-runtime.sh RT
#
# Builds into RT (idempotent; a finished RT/pins.env is reused after its pins
# are rechecked):
#   RT/bin/aienet0-run, RT/bin/aienet0-serve   open-humanity 53c39bf2031c, cargo --release --locked
#   RT/bin/matmul_u32_recompute                 aien-protocols e705fdc9 specs/aienet/jobpath/build.sh,
#                                               pinned omc 2c824805...029f, SHA unit dbf002dd... (omega-mv 55845e70)
#   RT/keys RT/pins RT/wkeys RT/wpins           test identities, generated as in aien-protocols
#                                               evidence/aienet0-ledger-auth-20261028/receipt.txt:38-43
#   RT/pins.env                                 every pin and binary sha256 (sourced by run-demo.sh)
# Environment (optional):
#   OMC_PINNED   pinned omc 2c824805 (default /home/drakestapleton/workspace/omega-mcp/build/omc; copied, never modified)
#   OMEGA_MV_GIT an omega-mv git dir holding commit 55845e70 (default /tmp/om-main)
# Exit: 0 ready; 3 NOT_RUN (a pin differs or a source is missing); 60 build failure.
set -eu
RT=${1:?usage: setup-runtime.sh RT}
OH_REPO=https://github.com/aien-dev/open-humanity
OH_PIN=53c39bf2031c
PROTO_REPO=https://github.com/aien-dev/aien-protocols
PROTO_PIN=e705fdc9
OMC_PIN=2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f
SHA_UNIT_COMMIT=55845e707dac0d09bf22b5e16970a73c0a989a8c
RECOMPUTE_PIN=d2cd7382e0dc64e49498ae3b1ceffc73f1ddcbd3ddcf93e6d607498ccd58ea12
OMC_PINNED=${OMC_PINNED:-/home/drakestapleton/workspace/omega-mcp/build/omc}
OMEGA_MV_GIT=${OMEGA_MV_GIT:-/tmp/om-main}
HEAVY="taskset -c 10-14 nice -n 19 ionice -c3"
sha() { sha256sum < "$1" | cut -c1-64; }
mkdir -p "$RT/bin" "$RT/src" "$RT/log"
RT=$(cd "$RT" && pwd)

if [ ! -x "$RT/bin/aienet0-run" ] || [ ! -x "$RT/bin/aienet0-serve" ]; then
  [ -d "$RT/src/open-humanity/.git" ] || git clone -q "$OH_REPO" "$RT/src/open-humanity"
  git -C "$RT/src/open-humanity" checkout -q "$OH_PIN"
  ( cd "$RT/src/open-humanity" &&
    $HEAVY cargo build --release --locked -j 2 -p oh-coordinator -p oh-volunteer \
      --bin aienet0-run --bin aienet0-serve ) > "$RT/log/oh-build.log" 2>&1 || { echo "BUILD_FAIL open-humanity (see $RT/log/oh-build.log)"; exit 60; }
  cp "$RT/src/open-humanity/target/release/aienet0-run" "$RT/src/open-humanity/target/release/aienet0-serve" "$RT/bin/"
fi
OH_HEAD=$(git -C "$RT/src/open-humanity" rev-parse HEAD)
case $OH_HEAD in "$OH_PIN"*) ;; *) echo "NOT_RUN: open-humanity HEAD $OH_HEAD is not $OH_PIN"; exit 3;; esac

if [ ! -x "$RT/bin/matmul_u32_recompute" ]; then
  [ -f "$OMC_PINNED" ] || { echo "NOT_RUN: pinned omc missing at $OMC_PINNED"; exit 3; }
  [ "$(sha "$OMC_PINNED")" = "$OMC_PIN" ] || { echo "NOT_RUN: $OMC_PINNED is not omc $OMC_PIN"; exit 3; }
  cp "$OMC_PINNED" "$RT/bin/omc-2c824805"
  chmod +x "$RT/bin/omc-2c824805"
  [ -d "$RT/src/aien-protocols/.git" ] || git clone -q "$PROTO_REPO" "$RT/src/aien-protocols"
  git -C "$RT/src/aien-protocols" checkout -q "$PROTO_PIN"
  mkdir -p "$RT/src/omega-mv-55845e70/om/sha256"
  git --git-dir="$OMEGA_MV_GIT/.git" show "$SHA_UNIT_COMMIT:om/sha256/sha256-om2.om" \
    > "$RT/src/omega-mv-55845e70/om/sha256/sha256-om2.om" || { echo "NOT_RUN: $OMEGA_MV_GIT lacks omega-mv $SHA_UNIT_COMMIT"; exit 3; }
  OMEGA_OMC="$RT/bin/omc-2c824805" OMEGA_MV="$RT/src/omega-mv-55845e70" \
    $HEAVY sh "$RT/src/aien-protocols/specs/aienet/jobpath/build.sh" "$RT/bin/matmul_u32_recompute" \
    > "$RT/log/recompute-build.log" 2>&1 || { cat "$RT/log/recompute-build.log"; echo "BUILD_FAIL recompute"; exit 60; }
fi
[ "$(sha "$RT/bin/matmul_u32_recompute")" = "$RECOMPUTE_PIN" ] || { echo "NOT_RUN: recompute sha $(sha "$RT/bin/matmul_u32_recompute") is not $RECOMPUTE_PIN"; exit 3; }

if [ ! -f "$RT/pins/executor.p256" ]; then
  umask 077
  mkdir -p "$RT/keys" "$RT/wkeys" "$RT/pins" "$RT/wpins"
  for f in keys/transport.seed keys/submitter.p256 keys/verifier.p256 wkeys/transport.seed wkeys/executor.p256; do
    [ -f "$RT/$f" ] || openssl rand -hex -out "$RT/$f" 32
  done
  "$RT/bin/aienet0-run" keys --keys "$RT/keys" > "$RT/coordinator-public-keys.txt"
  "$RT/bin/aienet0-serve" keys --keys "$RT/wkeys" > "$RT/worker-public-keys.txt"
  awk '$1=="coordinator.ed25519"{print $2 > "'"$RT/wpins/coordinator.ed25519"'"}
       $1=="submitter.p256"{print $2 > "'"$RT/wpins/submitter.p256"'"}' "$RT/coordinator-public-keys.txt"
  awk '$1=="volunteer.ed25519"{print $2 > "'"$RT/pins/volunteer.ed25519"'"}
       $1=="executor.p256"{print $2 > "'"$RT/pins/executor.p256"'"}' "$RT/worker-public-keys.txt"
fi

cat > "$RT/pins.env" <<EOF
AN0_RUN=$RT/bin/aienet0-run
AN0_SERVE=$RT/bin/aienet0-serve
AN0_RUN_SHA256=$(sha "$RT/bin/aienet0-run")
AN0_SERVE_SHA256=$(sha "$RT/bin/aienet0-serve")
OH_COMMIT=$OH_HEAD
PROTO_COMMIT=$(git -C "$RT/src/aien-protocols" rev-parse HEAD 2>/dev/null || echo NOT_BUILT_HERE)
RECOMPUTE=$RT/bin/matmul_u32_recompute
RECOMPUTE_SHA256=$RECOMPUTE_PIN
RECOMPUTE_OMC_SHA256=$OMC_PIN
SHA_UNIT_COMMIT=$SHA_UNIT_COMMIT
KEYS=$RT/keys
PINS=$RT/pins
WKEYS=$RT/wkeys
WPINS=$RT/wpins
RUSTC='$(rustc --version)'
EOF
cat "$RT/pins.env"
echo "RUNTIME_READY $RT"

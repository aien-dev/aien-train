#!/bin/sh
# Build gr_info against the pinned physics checkout (omega physics.lock 6d7cf0d4).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
P=${PHYSICS_DIR:-$HOME/workspace/physics-pf-6d7cf0d4}
NV=$P/third_party/nvidia-open-580.173.02
out=${OUT_DIR:-$here/build}
mkdir -p "$out"
cc -std=gnu11 -Wall -Wextra -Werror -D_GNU_SOURCE -O2 \
  -I"$P/nvrm" -I"$NV/src/common/sdk/nvidia/inc" -I"$NV/kernel-open/common/inc" \
  -I"$NV/kernel-open/nvidia-uvm" -I"$NV/src/nvidia/arch/nvalloc/unix/include" \
  "$here/gr_info.c" "$P/nvrm/nvrm.c" -o "$out/gr_info"
echo "built $out/gr_info (physics $(git -C "$P" rev-parse --short=8 HEAD))"

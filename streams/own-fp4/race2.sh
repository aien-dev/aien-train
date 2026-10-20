#!/bin/bash
# race2.sh (L284 oracle): like race.sh, but waits for an earlier queue to end and retries a run up to 6 times when the
# GPU refuses a context at start-up (cuDevicePrimaryCtxRetain out of memory: other jobs' servers hold the GPU).
# Usage: race2.sh WAIT_PID "recipe:seed[:extra] ..."
D=$HOME/workspace/aien-train-own-fp4-data; H=$(cd "$(dirname "$0")" && pwd); PY=$HOME/.venv/bin/python
mkdir -p $D/runs
while kill -0 "$1" 2>/dev/null; do sleep 60; done
for item in $2; do
  IFS=: read -r rec seed extra <<<"$item"; out=$D/runs/$rec-s$seed
  [ -f $out/result.json ] && { echo "skip $out"; continue; }
  for try in 1 2 3 4 5 6; do
    until timeout 120 $PY -I -c "import torch; torch.ones(1,device='cuda'); assert torch.cuda.mem_get_info()[0] > 6e9" >/dev/null 2>&1; do echo "$(date -u +%H:%MZ) waiting for GPU ($rec)"; sleep 120; done
    echo "$(date -u +%H:%MZ) start $rec seed $seed try $try"
    nice -n 10 $PY -I $H/train.py --recipe $rec --seed $seed --out $out $extra > $out.log 2>&1; rc=$?
    echo "$(date -u +%H:%MZ) end $rec rc=$rc $(tail -1 $out.log | cut -c1-200)"
    [ $rc = 0 ] && break
    grep -q "cuDevicePrimaryCtxRetain\|out of memory" $out.log || break
    sleep 120
  done
done

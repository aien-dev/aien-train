#!/bin/bash
# race4.sh (L284 oracle): like race2.sh, but every run holds /tmp/aien-gb10.lock (shared GPU rule, L182 note 2026-10-20)
# so locked timing measurements by other jobs stay valid; the lock is released between runs. Usage: race4.sh "recipe:seed[:extra] ..."
D=$HOME/workspace/aien-train-own-fp4-data; H=$(cd "$(dirname "$0")" && pwd); PY=$HOME/.venv/bin/python
mkdir -p $D/runs
for item in $1; do
  IFS=: read -r rec seed extra <<<"$item"; out=$D/runs/$rec-s$seed
  [ -f $out/result.json ] && { echo "skip $out"; continue; }
  for try in 1 2 3 4 5 6; do
    echo "$(date -u +%H:%MZ) waiting for lock ($rec)"
    flock -x /tmp/aien-gb10.lock bash -c "
      until timeout 120 $PY -I -c \"import torch; torch.ones(1,device='cuda'); assert torch.cuda.mem_get_info()[0] > 6e9\" >/dev/null 2>&1; do echo \"\$(date -u +%H:%MZ) waiting for GPU ($rec)\"; sleep 60; done
      echo \"\$(date -u +%H:%MZ) start $rec seed $seed try $try\"
      nice -n 10 $PY -I $H/train.py --recipe $rec --seed $seed --out $out $extra > $out.log 2>&1"
    rc=$?
    echo "$(date -u +%H:%MZ) end $rec rc=$rc $(tail -1 $out.log | cut -c1-200)"
    [ $rc = 0 ] && break
    grep -q "cuDevicePrimaryCtxRetain\|out of memory" $out.log || break
    sleep 60
  done
done

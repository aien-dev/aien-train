#!/bin/bash
# race.sh (L284 oracle): run recipes one after another on the shared GPU; each waits until a CUDA context can be made
# (two 27B servers on this box sometimes leave none). Usage: race.sh "recipe:seed[:extra args] ..."
D=$HOME/workspace/aien-train-own-fp4-data; H=$(cd "$(dirname "$0")" && pwd); PY=$HOME/.venv/bin/python
mkdir -p $D/runs
until grep -q "^train " $D/prep.log 2>/dev/null; do sleep 30; done
for item in $1; do
  IFS=: read -r rec seed extra <<<"$item"; out=$D/runs/$rec-s$seed
  [ -f $out/result.json ] && { echo "skip $out"; continue; }
  until timeout 120 $PY -I -c "import torch; torch.ones(1,device='cuda'); assert torch.cuda.mem_get_info()[0] > 6e9" >/dev/null 2>&1; do echo "$(date -u +%H:%MZ) waiting for GPU ($rec)"; sleep 120; done
  echo "$(date -u +%H:%MZ) start $rec seed $seed"
  nice -n 10 $PY -I $H/train.py --recipe $rec --seed $seed --out $out $extra > $out.log 2>&1; rc=$?
  echo "$(date -u +%H:%MZ) end $rec rc=$rc $(tail -1 $out.log | cut -c1-200)"
done

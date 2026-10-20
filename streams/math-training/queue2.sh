#!/bin/sh
# L283 remaining GPU queue (fourier 1e-5 seed 0 already done, receipt in evidence): every GPU run holds /tmp/aien-gb10.lock (one run at a time), so the live 30B server is not starved.
cd "$(dirname "$0")"
S=../../.scratch; EV=../../evidence/math-training; PY=$HOME/.venv/bin/python
L() { log=$1; shift; nice -n 10 ./run_retry.sh "$log" flock /tmp/aien-gb10.lock "$@"; }
for lr in 1e-4 1e-3; do L $S/e2-fourier-$lr.log $PY -u e2.py train fourier $lr 0; done
best() { $PY -I -c "import json,glob; r=[json.load(open(f)) for f in glob.glob('$EV/e2-train-$1-*-0.json')]; print(min(r,key=lambda x:x['val_nll'])['lr'])"; }
for m in lora phm fourier; do lr=$(best $m); L $S/e2-$m-$lr-s1.log $PY -u e2.py train $m $lr 1; done
lr=$(best lora); L $S/e2-lora64.log $PY -u e2.py train lora64 $lr 0
nice -n 5 ./run_retry.sh $S/e2-time.log quiet-measure flock /tmp/aien-gb10.lock $PY -u e2.py time
echo QUEUE-DONE $(date -u +%H:%M:%SZ) > $S/queue.done

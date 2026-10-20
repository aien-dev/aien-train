#!/bin/sh
# Sequential GPU queue for L283 after the E1 timing run: E3 nonlin, E4, E2 sweep + controls, E2 timing (quiet lock).
cd "$(dirname "$0")"
S=../../.scratch; EV=../../evidence/math-training; PY=$HOME/.venv/bin/python
R() { nice -n 10 ./run_retry.sh "$S/$1.log" $PY -u "$@" ; }
until [ -f $EV/e1-time.json ]; do sleep 30; done
nice -n 10 ./run_retry.sh $S/e3-nonlin.log $PY -u e3.py nonlin
nice -n 10 ./run_retry.sh $S/e4.log $PY -u e4.py
for lr in 1e-4 3e-4 1e-3; do nice -n 10 ./run_retry.sh $S/e2-lora-$lr.log $PY -u e2.py train lora $lr 0; done
for lr in 1e-4 3e-4 1e-3; do nice -n 10 ./run_retry.sh $S/e2-phm-$lr.log $PY -u e2.py train phm $lr 0; done
for lr in 1e-5 1e-4 1e-3; do nice -n 10 ./run_retry.sh $S/e2-fourier-$lr.log $PY -u e2.py train fourier $lr 0; done
best() { $PY -c "import json,glob,sys; r=[json.load(open(f)) for f in glob.glob('$EV/e2-train-$1-*-0.json')]; print(min(r,key=lambda x:x['val_nll'])['lr'])"; }
for m in lora phm fourier; do lr=$(best $m); nice -n 10 ./run_retry.sh $S/e2-$m-$lr-s1.log $PY -u e2.py train $m $lr 1; done
lr=$(best lora); nice -n 10 ./run_retry.sh $S/e2-lora64.log $PY -u e2.py train lora64 $lr 0
nice -n 5 ./run_retry.sh $S/e2-time.log quiet-measure flock /tmp/aien-gb10.lock $PY -u e2.py time
echo QUEUE-DONE $(date -u +%H:%M:%SZ) > $S/queue.done

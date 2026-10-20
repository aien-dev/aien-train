#!/bin/sh
# L284 OMMA peak: pause my own training (race.sh child) for the window, run the chip probe under the GB10 lock, resume.
P=$(pgrep -f "train.py --recipe" | head -1)
[ -n "$P" ] && kill -STOP $P
cd /home/drakestapleton/workspace/omega-own-fp4-omma
{ date -u +%FT%TZ; git rev-parse HEAD; git status --short | head; uptime; nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv; } > /home/drakestapleton/workspace/aien-train-own-fp4/evidence/own-fp4/omma-peak/context.txt 2>&1
flock /tmp/aien-gb10.lock build/gpu_omma_peak_test > /home/drakestapleton/workspace/aien-train-own-fp4/evidence/own-fp4/omma-peak/gpu-omma-peak.txt 2>&1; echo "## rc=$?" >> /home/drakestapleton/workspace/aien-train-own-fp4/evidence/own-fp4/omma-peak/gpu-omma-peak.txt
uptime >> /home/drakestapleton/workspace/aien-train-own-fp4/evidence/own-fp4/omma-peak/context.txt
[ -n "$P" ] && kill -CONT $P

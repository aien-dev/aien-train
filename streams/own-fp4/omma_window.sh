#!/bin/bash
# omma_window.sh (L284): one short chip window for the OMMA probe debugging. Waits its turn (spark-window --wait, first come first served),
# runs the proven FP8 control (gpu_qmma_probe_test) then the tiny OMMA probe, saves both outputs, closes the window.
set -u
O=$HOME/workspace/omega-own-fp4-omma; E=$HOME/workspace/aien-train-own-fp4/evidence/own-fp4/omma-peak; mkdir -p $E
export PHYSICS_DIR=$HOME/workspace/physics-bytes-6d7cf0d4
id=$(spark-window open l284-omma --ledger L284 --lease 15 --wait | tail -1) || exit 1
echo "window $id"
cd $O
spark-window run bash -c "build/gpu_qmma_probe_test > $E/ctl-qmma-probe.txt 2>&1; echo ctl rc=\$? >> $E/ctl-qmma-probe.txt; build/gpu_omma_peak_test --tiny > $E/tiny-boot1.txt 2>&1; echo rc=\$? >> $E/tiny-boot1.txt"
spark-window close "$id"

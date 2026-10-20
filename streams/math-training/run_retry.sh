#!/bin/sh
# Rerun a GPU script when CUDA context creation fails on fragmented memory (cuDevicePrimaryCtxRetain OOM).
# Usage: run_retry.sh LOG cmd...   Retries up to 30 times, 60 s apart, only for that specific startup failure.
log=$1; shift
i=0
while [ $i -lt 30 ]; do
  "$@" > "$log" 2>&1 && exit 0
  grep -q "cuDevicePrimaryCtxRetain" "$log" || exit 1
  i=$((i+1)); echo "ctx-retry $i $(date -u +%H:%M:%SZ)" >> "$log.retries"; sleep 60
done
exit 1

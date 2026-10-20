#!/bin/bash
# Fetch FineWeb-Edu sample-10BT parquet shards at a pinned Hub revision, verify each against the
# sha256 the Hub publishes (LFS oid), resume-safe. Usage: fetch.sh <dest-dir> <tree.json>
set -u
REV=87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
DEST=$1; TREE=$2
python3 -I - "$TREE" <<'P' > "$DEST/.list"
import json,sys
for f in json.load(open(sys.argv[1])):
    print(f['path'].split('/')[-1], f['size'], f['lfs']['oid'])
P
while read -r name size oid; do
  f="$DEST/$name"
  have=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$have" != "$size" ]; then
    nice -n 19 ionice -c 3 curl -sSL --retry 8 --retry-delay 5 -C - -o "$f" \
      "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/$REV/sample/10BT/$name" || { echo "FAIL $name"; continue; }
  fi
  got=$(nice -n 19 ionice -c 3 sha256sum "$f" | cut -d' ' -f1)
  if [ "$got" = "$oid" ]; then echo "OK $name $got"; else echo "MISMATCH $name got=$got want=$oid"; fi
done < "$DEST/.list"

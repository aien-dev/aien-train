#!/usr/bin/env python3
"""Data preparation only (not the product path): stream the `text` column of FineWeb-Edu parquet
shards to stdout as length-prefixed documents: u32 little-endian byte length, then UTF-8 bytes.
Usage: extract.py <shard.parquet> [--skip-docs N] [--max-docs N] [--max-bytes N]
Prints nothing else on stdout. Run with the pyarrow venv."""
import sys, struct, argparse
import pyarrow.parquet as pq
ap = argparse.ArgumentParser()
ap.add_argument("shard"); ap.add_argument("--skip-docs", type=int, default=0)
ap.add_argument("--max-docs", type=int, default=1 << 62); ap.add_argument("--max-bytes", type=int, default=1 << 62)
a = ap.parse_args()
out = sys.stdout.buffer
seen = emitted = nbytes = 0
for batch in pq.ParquetFile(a.shard).iter_batches(batch_size=2048, columns=["text"]):
    for t in batch.column(0).to_pylist():
        seen += 1
        if seen <= a.skip_docs: continue
        if emitted >= a.max_docs or nbytes >= a.max_bytes: sys.exit(0)
        b = t.encode("utf-8")
        out.write(struct.pack("<I", len(b))); out.write(b)
        emitted += 1; nbytes += len(b)

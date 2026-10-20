# One-time conversion of the locally cached GSM8K arrow files to JSONL (pyarrow lives only in another venv).
import json, sys, pyarrow as pa
src, dst = sys.argv[1], sys.argv[2]
with pa.memory_map(src) as f:
    t = pa.ipc.open_stream(f).read_all()
with open(dst, "w") as o:
    for q, a in zip(t.column("question").to_pylist(), t.column("answer").to_pylist()):
        o.write(json.dumps({"question": q, "answer": a}) + "\n")
print(dst, t.num_rows)

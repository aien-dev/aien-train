# prep_data.py (L284 oracle): SearchQA web-snippet text already on this Spark -> uint16 token files.
# Arrow files are read as raw bytes (no pyarrow here): every printable UTF-8 run of >= 200 chars is kept as text.
# Tokenizer: TinyLlama-1.1B tokenizer.json (32000 ids). Train = searchqa-train.arrow, held-out val = searchqa-validation.arrow.
import sys, re, hashlib, numpy as np
from tokenizers import Tokenizer
D = sys.argv[1]; OUT = sys.argv[2]; TOK = sys.argv[3]
tok = Tokenizer.from_file(TOK)
run = re.compile(rb'[\x20-\x7e\xc2-\xf4\x80-\xbf\n]{200,}')
for split, cap in (("validation", 4_000_000), ("train", 400_000_000)):
    raw = open(f"{D}/searchqa-{split}.arrow", "rb").read()
    texts = [m.group(0).decode("utf-8", "ignore") for m in run.finditer(raw)]
    ids = []; n = 0
    for i in range(0, len(texts), 2000):
        for enc in tok.encode_batch(texts[i:i + 2000], add_special_tokens=False):
            ids.append(np.asarray([1] + enc.ids, dtype=np.uint16)); n += len(enc.ids) + 1
        if n >= cap: break
    arr = np.concatenate(ids)[:cap]
    arr.tofile(f"{OUT}/{split}.u16")
    print(split, "texts", len(texts), "tokens", arr.size, "sha256", hashlib.sha256(arr.tobytes()).hexdigest()[:16], flush=True)

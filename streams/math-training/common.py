# Shared pieces for the L283 math-training experiments: data, frozen model, feature extraction, receipts.
# PyTorch here is an oracle (feature extraction and baselines), never the product path.
import hashlib, json, os, platform, subprocess, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, ".scratch")
EVID = os.path.join(ROOT, "evidence", "math-training")
MODEL_DIR = os.path.expanduser("~/models/SmolLM2-1.7B-Instruct-31b70e2e869a")
MODEL_SHA = "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8"  # PROVENANCE.md, model.safetensors
SEQ = 256
DEV = "cuda"


def code_sha():
    return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def code_dirty():
    out = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "streams"], capture_output=True, text=True).stdout
    return bool(out.strip())


def receipt(name, payload):
    """Write evidence/math-training/<name>.json with chip, model sha, code sha and command."""
    os.makedirs(EVID, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    body = {
        "chip": "GB10 (DGX Spark)", "torch_device": gpu, "host": platform.node(),
        "model": MODEL_DIR, "model_sha256": MODEL_SHA,
        "code_sha": code_sha(), "code_dirty": code_dirty(),
        "command": " ".join([os.path.relpath(sys.executable, os.path.expanduser("~")) if sys.executable.startswith(os.path.expanduser("~")) else sys.executable] + sys.argv),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__, "numpy": np.__version__,
        **payload,
    }
    path = os.path.join(EVID, name + ".json")
    with open(path, "w") as f:
        json.dump(body, f, indent=1, default=float)
    print("receipt", path)
    return path


def load_split(split):
    with open(os.path.join(SCRATCH, f"gsm8k-{split}.jsonl")) as f:
        return [json.loads(l) for l in f]


def text_of(ex):
    return "Question: " + ex["question"] + "\nAnswer: " + ex["answer"]


def load_model(dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype).to(DEV)
    model.eval()
    return tok, model


def encode(tok, exs, seq=SEQ):
    """Token id lists, one per example, truncated to seq."""
    return [tok(text_of(e), add_special_tokens=False)["input_ids"][:seq] for e in exs]


def batches(ids, bs):
    for i in range(0, len(ids), bs):
        chunk = ids[i:i + bs]
        L = max(len(c) for c in chunk)
        x = torch.full((len(chunk), L), 2, dtype=torch.long)
        m = torch.zeros((len(chunk), L), dtype=torch.bool)
        for j, c in enumerate(chunk):
            x[j, :len(c)] = torch.tensor(c)
            m[j, :len(c)] = True
        yield x, m


@torch.no_grad()
def extract_final(model, ids, cap_tokens, bs=16):
    """Final-norm hidden state h_t (fp32) and next token y_t for positions t with a next token, up to cap_tokens."""
    feats, ys, n = [], [], 0
    for x, m in batches(ids, bs):
        x, m = x.to(DEV), m.to(DEV)
        out = model.model(input_ids=x, attention_mask=m.long())
        h = out.last_hidden_state  # after the final RMSNorm
        valid = m[:, 1:] & m[:, :-1]
        feats.append(h[:, :-1][valid].float().cpu())
        ys.append(x[:, 1:][valid].cpu())
        n += int(valid.sum())
        if n >= cap_tokens:
            break
    X = torch.cat(feats)[:cap_tokens]
    Y = torch.cat(ys)[:cap_tokens]
    return X, Y


def sha_of_tensor(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()

# E2: structured adapter bases vs LoRA on frozen SmolLM2-1.7B, matched trainable-parameter counts.
#   python e2.py check                  -> numeric checks of the three adapter operators (FourierFT custom backward vs autograd, PHM vs dense Kronecker)
#   python e2.py train METHOD LR SEED [STEPS]  -> one fine-tune, writes evidence/math-training/e2-train-METHOD-LR-SEED.json
#   python e2.py time                   -> step time / peak memory per method, 3 repeats (run under quiet-measure)
# Adapters on q,k,v,o,gate,up,down of all 24 layers. Every adapter is written the same way (plain torch ops, no PEFT).
# Loss: next-token cross-entropy on the answer tokens of GSM8K. Fit: train[0:6500]; val: train[7000:7300]; test: test[0:300].
import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
R = 8          # LoRA rank; the other methods are matched to its parameter count
PHM_N = 4      # Kronecker block count n for PHM


class LoRA(nn.Module):
    def __init__(self, base, r=R, alpha=16):
        super().__init__()
        self.base = base
        o, i = base.weight.shape
        self.A = nn.Parameter(torch.randn(r, i) / math.sqrt(i))
        self.B = nn.Parameter(torch.zeros(o, r))
        self.s = alpha / r

    def delta(self, x):
        return (x @ self.A.to(x.dtype).T) @ self.B.to(x.dtype).T * self.s

    def forward(self, x):
        return self.base(x) + self.delta(x)


class PHM(nn.Module):
    """Delta W = sum_m A_m kron (S_m T_m^T): A_m is n x n (shared per layer), S_m is (o/n) x r, T_m is (i/n) x r.
    Applied without building Delta W: split x into n blocks, project each with T_m, mix blocks with A_m, expand with S_m.
    Trainable per matrix: n * r * (o + i) / n = r (o + i), the same as LoRA rank r; plus n^3 shared per layer.
    Rank of Delta W is at most n * (n * r) = n^2 r (128 for n=4, r=8), against r = 8 for LoRA."""
    def __init__(self, base, Ashared, r=R, n=PHM_N, alpha=16):
        super().__init__()
        self.base = base
        o, i = base.weight.shape
        self.n, self.o, self.i = n, o, i
        self.A = Ashared
        self.T = nn.Parameter(torch.randn(n, i // n, r) / math.sqrt(i // n))
        self.S = nn.Parameter(torch.zeros(n, o // n, r))
        self.s = alpha / r

    def delta(self, x):
        xb = x.reshape(*x.shape[:-1], self.n, self.i // self.n)                 # [..., n_j, i/n]
        U = torch.einsum("...jk,mkr->...mjr", xb, self.T.to(x.dtype))            # [..., m, n_j, r]
        V = torch.einsum("mij,...mjr->...mir", self.A.to(x.dtype), U)            # [..., m, n_i, r]
        y = torch.einsum("...mir,mor->...io", V, self.S.to(x.dtype))             # [..., n_i, o/n]
        return y.reshape(*x.shape[:-1], self.o) * self.s

    def dense_delta(self):
        return sum(torch.kron(self.A[m], self.S[m] @ self.T[m].T) for m in range(self.n)) * self.s

    def forward(self, x):
        return self.base(x) + self.delta(x)


class _FourierMatmul(torch.autograd.Function):
    """y = x Delta W^T with Delta W = alpha * Re(sum_k c_k exp(+2 pi i (u_k p / o + v_k q / i))), c real.
    Forward builds Delta W by one inverse 2-D FFT of the scattered coefficients; backward rebuilds it instead of storing it,
    and gets dL/dc_k = alpha * Re(FFT2(dL/dDeltaW))[u_k, v_k] (the adjoint of the scatter + inverse FFT)."""
    @staticmethod
    def forward(ctx, x, c, idx, o, i, alpha):
        dW = _fourier_dense(c, idx, o, i, alpha).to(x.dtype)
        ctx.save_for_backward(x, c, idx)
        ctx.o, ctx.i, ctx.alpha = o, i, alpha
        return x @ dW.T

    @staticmethod
    def backward(ctx, g):
        x, c, idx = ctx.saved_tensors
        dW = _fourier_dense(c, idx, ctx.o, ctx.i, ctx.alpha).to(g.dtype)
        gx = g @ dW
        gW = g.reshape(-1, ctx.o).T.float() @ x.reshape(-1, ctx.i).float()       # [o, i]
        gc = ctx.alpha * torch.fft.fft2(gW, norm="backward").real.reshape(-1)[idx]
        return gx, gc, None, None, None, None


def _fourier_dense(c, idx, o, i, alpha):
    spec = torch.zeros(o * i, dtype=torch.complex64, device=c.device)
    spec[idx] = c.to(torch.complex64)
    return alpha * torch.fft.ifft2(spec.reshape(o, i), norm="forward").real


class FourierFT(nn.Module):
    def __init__(self, base, ncoef, seed):
        super().__init__()
        self.base = base
        o, i = base.weight.shape
        self.o, self.i = o, i
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("idx", torch.randperm(o * i, generator=g)[:ncoef])
        self.c = nn.Parameter(torch.zeros(ncoef))
        self.alpha = 1.0 / math.sqrt(ncoef)

    def delta(self, x):
        return _FourierMatmul.apply(x, self.c, self.idx.to(x.device), self.o, self.i, self.alpha)

    def forward(self, x):
        return self.base(x) + self.delta(x)


def attach(model, method, seed):
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    n_train = 0
    for li, layer in enumerate(model.model.layers):
        Ash = nn.Parameter(torch.randn(PHM_N, PHM_N, PHM_N, device=DEV) / PHM_N) if method == "phm" else None
        if Ash is not None:
            layer.register_parameter("phm_A", Ash)
        for parent in (layer.self_attn, layer.mlp):
            for name in TARGETS:
                if not hasattr(parent, name):
                    continue
                base = getattr(parent, name)
                o, i = base.weight.shape
                if method.startswith("lora"):
                    r = int(method[4:] or R)
                    m = LoRA(base, r=r)
                elif method == "phm":
                    m = PHM(base, Ash)
                elif method == "fourier":
                    m = FourierFT(base, R * (o + i), seed * 1000 + li * 10 + TARGETS.index(name))
                else:
                    raise ValueError(method)
                setattr(parent, name, m.to(DEV))
    params = [p for p in model.parameters() if p.requires_grad]
    return params, sum(p.numel() for p in params)


def answer_batches(tok, exs, bs, seq=SEQ):
    """(ids, mask over loss positions): loss only on tokens after 'Answer:'."""
    rows = []
    for e in exs:
        q = tok("Question: " + e["question"] + "\nAnswer:", add_special_tokens=False)["input_ids"]
        a = tok(" " + e["answer"], add_special_tokens=False)["input_ids"]
        ids = (q + a)[:seq]
        rows.append((ids, len(q)))
    for k in range(0, len(rows), bs):
        chunk = rows[k:k + bs]
        L = max(len(r[0]) for r in chunk)
        x = torch.full((len(chunk), L), 2, dtype=torch.long)
        lm = torch.zeros((len(chunk), L), dtype=torch.bool)
        am = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (ids, qlen) in enumerate(chunk):
            x[j, :len(ids)] = torch.tensor(ids)
            am[j, :len(ids)] = 1
            lm[j, qlen:len(ids)] = True
        yield x.to(DEV), am.to(DEV), lm.to(DEV)


def lm_loss(model, x, am, lm):
    logits = model(input_ids=x, attention_mask=am).logits[:, :-1].float()
    tgt = x[:, 1:]
    m = lm[:, 1:]
    return F.cross_entropy(logits[m], tgt[m], reduction="sum"), int(m.sum())


@torch.no_grad()
def eval_nll(model, tok, exs):
    tot, n = 0.0, 0
    for x, am, lm in answer_batches(tok, exs, 8):
        l, k = lm_loss(model, x, am, lm)
        tot += float(l)
        n += k
    return tot / n


def splits():
    tr, te = load_split("train"), load_split("test")
    return tr[:6500], tr[7000:7300], te[:300]


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        torch.manual_seed(0)
        out = {}
        # FourierFT: custom backward vs plain autograd through torch.fft, small matrix, fp64-accurate in fp32
        o, i, nc = 48, 32, 100
        base = nn.Linear(i, o, bias=False).to(DEV)
        f = FourierFT(base, nc, 1).to(DEV)
        with torch.no_grad():
            f.c.normal_()
        x = torch.randn(5, i, device=DEV, requires_grad=True)
        y = f.delta(x)
        g = torch.randn_like(y)
        gx, gc = torch.autograd.grad(y, (x, f.c), g)
        c2 = f.c.detach().clone().requires_grad_(True)
        x2 = x.detach().clone().requires_grad_(True)
        y2 = x2 @ _fourier_dense(c2, f.idx, o, i, f.alpha).T
        gx2, gc2 = torch.autograd.grad(y2, (x2, c2), g)
        out["fourier_forward_rel_err"] = float((y - y2).norm() / y2.norm())
        out["fourier_grad_x_rel_err"] = float((gx - gx2).norm() / gx2.norm())
        out["fourier_grad_c_rel_err"] = float((gc - gc2).norm() / gc2.norm())
        # PHM: structured apply vs the dense Kronecker sum
        base = nn.Linear(64, 96, bias=False).to(DEV)
        A = nn.Parameter(torch.randn(PHM_N, PHM_N, PHM_N, device=DEV))
        p = PHM(base, A, r=3).to(DEV)
        with torch.no_grad():
            p.S.normal_()
        x = torch.randn(7, 64, device=DEV)
        out["phm_structured_vs_dense_rel_err"] = float((p.delta(x) - x @ p.dense_delta().T).norm() / (x @ p.dense_delta().T).norm())
        out["phm_dense_rank"] = int(torch.linalg.matrix_rank(p.dense_delta()))
        out["phm_rank_bound_n_squared_times_r"] = PHM_N * PHM_N * 3  # n terms, each rank(A_m) * rank(S_m T_m^T) = n * r
        print(out)
        receipt("e2-check", out)

    elif mode == "train":
        method, lr, seed = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        steps = int(sys.argv[5]) if len(sys.argv) > 5 else 300
        bs, accum = 4, 2
        tok, model = load_model()
        fit, val, test = splits()
        base_val = eval_nll(model, tok, val)
        base_test = eval_nll(model, tok, test)
        params, ntrain = attach(model, method, seed)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20))
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(fit), generator=g).tolist()
        curve, k = [], 0
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.train()
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            tl, tn = 0.0, 0
            for _ in range(accum):
                exs = [fit[order[(k + j) % len(fit)]] for j in range(bs)]
                k += bs
                x, am, lm = next(answer_batches(tok, exs, bs))
                l, n = lm_loss(model, x, am, lm)
                (l / n / accum).backward()
                tl += float(l)
                tn += n
            opt.step()
            sched.step()
            if step % 50 == 0 or step == steps:
                model.eval()
                v = eval_nll(model, tok, val)
                model.train()
                curve.append({"step": step, "train_nll": tl / tn, "val_nll": v, "t": time.time() - t0})
                print(method, lr, seed, curve[-1], flush=True)
        model.eval()
        res = {"method": method, "lr": lr, "seed": seed, "steps": steps, "batch": bs * accum, "trainable_params": ntrain,
               "val_nll_frozen": base_val, "curve": curve, "val_nll": curve[-1]["val_nll"],
               "test_nll": eval_nll(model, tok, test), "test_nll_frozen": base_test,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "seconds_unlocked": time.time() - t0}
        print(res["val_nll"], res["test_nll"], ntrain)
        receipt(f"e2-train-{method}-{lr:g}-{seed}", res)

    elif mode == "time":
        reps = {}
        fit, _, _ = splits()
        for method in ("none", "lora", "phm", "fourier"):
            reps[method] = []
            for r in range(3):
                tok, model = load_model()
                if method == "none":
                    for p in model.parameters():
                        p.requires_grad_(False)
                    model.model.embed_tokens.weight.requires_grad_(False)
                    params, ntrain = [], 0
                else:
                    params, ntrain = attach(model, method, 0)
                opt = torch.optim.AdamW(params, lr=1e-4) if params else None
                exs = fit[:4]
                x, am, lm = next(answer_batches(tok, exs, 4))
                times = []
                torch.cuda.reset_peak_memory_stats()
                for s in range(12):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    l, n = lm_loss(model, x, am, lm)
                    if params:
                        (l / n).backward()
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    times.append(time.time() - t0)
                reps[method].append({"step_s_median_after_warmup": sorted(times[2:])[len(times[2:]) // 2],
                                     "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "trainable": ntrain,
                                     "tokens_per_step": int(am.sum())})
                print(method, r, reps[method][-1], flush=True)
                del model, opt
                torch.cuda.empty_cache()
        receipt("e2-time", {"note": "batch 4 x up to 256 tokens, one optimizer step per timed step; 'none' = forward only, frozen", "repeats": reps})

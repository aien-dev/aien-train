# E1: direct head solve vs gradient training on frozen SmolLM2-1.7B features.
#   python e1.py sweep   -> picks every hyperparameter on the val split, scores test once, writes e1-sweep.json
#   python e1.py time    -> re-runs the chosen configs 3x (run under quiet-measure), writes e1-time.json
# Task A: next token over the K most frequent fit next-tokens + OTHER (K+1 classes).
# Task B: binary, "the next token is a digit".
# Cost unit: one "pass" = every row of X read once (a fused X W and X^T R over a chunk counts as one pass).
import json, os, sys, time, math
import torch
from common import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
K = 1024
CH = 16384
MODE = sys.argv[1] if len(sys.argv) > 1 else "sweep"


def sync():
    torch.cuda.synchronize()


def load(name):
    d = torch.load(os.path.join(SCRATCH, f"e1-{name}.pt"))
    X = torch.cat([d["X"], torch.ones(d["X"].shape[0], 1)], 1)  # append 1 so every head has a bias row
    return X.to(DEV), d["Y"].to(DEV)


Xf, Yf = load("fit")
Xv, Yv = load("val")
Xt, Yt = load("test")
N, D = Xf.shape
E = torch.load(os.path.join(SCRATCH, "e1-embed.pt")).to(DEV)  # [V, d], the frozen tied LM head

# ---- class map for task A
counts = torch.bincount(Yf, minlength=E.shape[0])
top = torch.argsort(counts, descending=True)[:K]
cmap = torch.full((E.shape[0],), K, dtype=torch.long, device=DEV)
cmap[top] = torch.arange(K, device=DEV)
cf, cv, ct = cmap[Yf], cmap[Yv], cmap[Yt]
C = K + 1
coverage = {s: float((c < K).float().mean()) for s, c in (("fit", cf), ("val", cv), ("test", ct))}

# ---- labels for task B (digit tokens, from the tokenizer)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
digit = torch.zeros(E.shape[0], dtype=torch.bool)
for v in range(E.shape[0]):
    s = tok.decode([v]).strip()
    digit[v] = len(s) > 0 and s.isdigit()
digit = digit.to(DEV)
bf, bv, bt = digit[Yf].float(), digit[Yv].float(), digit[Yt].float()


# ---- metrics
def eval_logits_A(logit_fn, X, c):
    """top-1 and NLL in class space; logit_fn(chunk) -> [n, C] log-probabilities or scores."""
    hit, nll, n = 0, 0.0, X.shape[0]
    for i in range(0, n, CH):
        lp = logit_fn(X[i:i + CH])
        hit += int((lp.argmax(1) == c[i:i + CH]).sum())
        nll += float(-torch.log_softmax(lp, 1).gather(1, c[i:i + CH, None]).sum())
    return {"top1": hit / n, "nll": nll / n}


def ceiling_A(X):
    """Frozen LM head, restricted to the same K+1 classes: p(OTHER) = sum of the rest of the vocabulary."""
    def f(x):
        lp = torch.log_softmax(x[:, :-1] @ E.T, 1)
        other = lp.clone()
        other[:, top] = -float("inf")
        return torch.cat([lp[:, top], torch.logsumexp(other, 1, keepdim=True)], 1)
    return f


def auc(score, y):
    o = torch.argsort(score)
    r = torch.empty_like(score)
    r[o] = torch.arange(1, len(score) + 1, device=score.device, dtype=score.dtype)
    npos = y.sum()
    nneg = len(y) - npos
    return float((r[y > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def eval_B(logit, y):
    p = torch.sigmoid(logit)
    pred = (p > 0.5).float()
    tpr = float((pred * y).sum() / y.sum())
    tnr = float(((1 - pred) * (1 - y)).sum() / (1 - y).sum())
    nll = float(torch.nn.functional.binary_cross_entropy_with_logits(logit, y))
    return {"auc": auc(logit, y), "bal_acc": (tpr + tnr) / 2, "nll": nll}


def ceiling_B(X):
    out = []
    for i in range(0, X.shape[0], CH):
        lp = torch.log_softmax(X[i:i + CH, :-1] @ E.T, 1)
        a = torch.logsumexp(lp[:, digit], 1)
        b = torch.logsumexp(lp[:, ~digit], 1)
        out.append(a - b)  # logit of p(digit)
    return torch.cat(out)


# ---- the direct solve: one pass builds G = X^T X and H = X^T Y, then a Cholesky solve
def gram_pass(X, labels, ncls):
    """G (fp64) and H (fp64). Each chunk matmul is fp32, accumulated in fp64. H for one-hot targets is a scatter-add."""
    G = torch.zeros(D, D, dtype=torch.float64, device=DEV)
    H = torch.zeros(D, ncls, dtype=torch.float64, device=DEV)
    for i in range(0, X.shape[0], CH):
        x = X[i:i + CH]
        G += (x.T @ x).double()
        if labels.dtype == torch.long:
            H.index_add_(1, labels[i:i + CH], x.T.double())
        else:
            H[:, 0] += (x.T @ labels[i:i + CH]).double()
    return G, H


def ridge_solve(G, H, lam, n):
    A = G / n + lam * torch.eye(D, dtype=torch.float64, device=DEV)
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(H / n, L).float()


def fit_scale(scores_fn, X, c):
    """One scalar beta so softmax(beta * scores) is a probability model; fit on val by golden-section on NLL."""
    def nll(b):
        return eval_logits_A(lambda x: b * scores_fn(x), X, c)["nll"]
    lo, hi = 1.0, 200.0
    g = (math.sqrt(5) - 1) / 2
    for _ in range(30):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


# ---- gradient baselines on the cached features
def ce_grad_pass(W, X, c, lam):
    """Mean CE + lam/2 |W|^2: loss and gradient in one pass over X."""
    g = torch.zeros_like(W)
    loss = 0.0
    for i in range(0, X.shape[0], CH):
        x = X[i:i + CH]
        p = torch.softmax(x @ W, 1)
        loss += float(-torch.log(p.gather(1, c[i:i + CH, None]).clamp_min(1e-30)).sum())
        p[torch.arange(x.shape[0], device=DEV), c[i:i + CH]] -= 1
        g += x.T @ p
    n = X.shape[0]
    return loss / n + lam / 2 * float((W * W).sum()), g / n + lam * W


def adamw_run(X, c, Xv, cv, loss_kind, lr, wd, target, max_epochs=30, bs=4096, seed=0):
    torch.manual_seed(seed)
    W = torch.zeros(D, C, device=DEV, requires_grad=True)
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=wd)
    hist, reach = [], {}
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        perm = torch.randperm(X.shape[0], device=DEV)
        for i in range(0, X.shape[0], bs):
            idx = perm[i:i + bs]
            s = X[idx] @ W
            if loss_kind == "ce":
                loss = torch.nn.functional.cross_entropy(s, c[idx])
            else:
                loss = ((s - torch.nn.functional.one_hot(c[idx], C).float()) ** 2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = eval_logits_A(lambda x: x @ W, Xv, cv)
        sync()
        hist.append({"epoch": ep, "val_top1": m["top1"], "val_nll": m["nll"], "t": time.time() - t0})
        for k, tv in target.items():
            if k not in reach and m["top1"] >= tv:
                reach[k] = {"epoch": ep, "passes": ep, "t": time.time() - t0}
        if len(reach) == len(target):
            break
    return W.detach(), hist, reach


def lbfgs_run(X, c, Xv, cv, lam, target, max_evals=300):
    W = torch.zeros(D, C, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W], lr=1, max_iter=10, history_size=20, line_search_fn="strong_wolfe")
    evals = [0]

    def closure():
        evals[0] += 1
        with torch.no_grad():
            l, g = ce_grad_pass(W, X, c, lam)
        W.grad = g
        return torch.tensor(l)

    hist, reach = [], {}
    t0 = time.time()
    while evals[0] < max_evals:
        opt.step(closure)
        with torch.no_grad():
            m = eval_logits_A(lambda x: x @ W, Xv, cv)
        sync()
        hist.append({"passes": evals[0], "val_top1": m["top1"], "val_nll": m["nll"], "t": time.time() - t0})
        for k, tv in target.items():
            if k not in reach and m["top1"] >= tv:
                reach[k] = {"passes": evals[0], "t": time.time() - t0}
        if len(reach) == len(target):
            break
    return W.detach(), hist, reach


def bohning_run(X, c, Xv, cv, lam, target, G, W0=None, max_passes=200):
    """Fixed-Hessian MM (Bohning 1992): Hessian of mean CE <= 1/2 (I - 11^T/C) kron G/n.
    Factor (G/(2n) + lam I) once; each step is one gradient pass plus two triangular solves.
    The column-mean part of the step solves lam * dW_m = grad_m (softmax ignores it)."""
    n = X.shape[0]
    L = torch.linalg.cholesky(G / (2 * n) + lam * torch.eye(D, dtype=torch.float64, device=DEV))
    W = torch.zeros(D, C, device=DEV) if W0 is None else W0.clone()
    hist, reach = [], {}
    t0 = time.time()
    for k in range(1, max_passes + 1):
        _, g = ce_grad_pass(W, X, c, lam)
        gm = g.mean(1, keepdim=True)
        W -= torch.cholesky_solve((g - gm).double(), L).float() + gm / lam
        m = eval_logits_A(lambda x: x @ W, Xv, cv)
        sync()
        hist.append({"passes": k, "val_top1": m["top1"], "val_nll": m["nll"], "t": time.time() - t0})
        for kk, tv in target.items():
            if kk not in reach and m["top1"] >= tv:
                reach[kk] = {"passes": k, "t": time.time() - t0}
        if len(reach) == len(target):
            break
    return W, hist, reach


# ---- task B: exact Newton (IRLS) for logistic regression, one weighted Gram pass per step
def irls_run(X, y, lam, steps=12, W0=None):
    n = X.shape[0]
    w = torch.zeros(D, device=DEV) if W0 is None else W0.clone()
    hist = []
    for k in range(1, steps + 1):
        Hs = torch.zeros(D, D, dtype=torch.float64, device=DEV)
        g = torch.zeros(D, dtype=torch.float64, device=DEV)
        for i in range(0, n, CH):
            x = X[i:i + CH]
            p = torch.sigmoid(x @ w)
            g += (x.T @ (p - y[i:i + CH])).double()
            Hs += ((x * (p * (1 - p))[:, None]).T @ x).double()
        g = g / n + lam * w.double()
        Hs = Hs / n + lam * torch.eye(D, dtype=torch.float64, device=DEV)
        step = torch.linalg.solve(Hs, g)
        w = w - step.float()
        hist.append({"step": k, "grad_norm": float(g.norm()), "step_norm": float(step.norm())})
        if float(step.norm()) < 1e-6 * max(1.0, float(w.norm())):
            break
    return w, hist


def adamw_B(X, y, lr, wd, epochs, bs=4096, seed=0):
    torch.manual_seed(seed)
    w = torch.zeros(D, device=DEV, requires_grad=True)
    opt = torch.optim.AdamW([w], lr=lr, weight_decay=wd)
    for ep in range(epochs):
        perm = torch.randperm(X.shape[0], device=DEV)
        for i in range(0, X.shape[0], bs):
            idx = perm[i:i + bs]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X[idx] @ w, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return w.detach()


def timed(f):
    sync()
    t0 = time.time()
    r = f()
    sync()
    return r, time.time() - t0


LAMS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
X_BYTES = N * D * 4

if MODE == "sweep":
    out = {"N_fit": N, "N_val": Xv.shape[0], "N_test": Xt.shape[0], "D_with_bias": D, "K": K,
           "class_coverage_in_topK": coverage, "digit_rate": {"fit": float(bf.mean()), "val": float(bv.mean()), "test": float(bt.mean())}}
    # ceilings
    out["ceiling_A"] = {"val": eval_logits_A(ceiling_A(Xv), Xv, cv), "test": eval_logits_A(ceiling_A(Xt), Xt, ct)}
    out["ceiling_B"] = {"val": eval_B(ceiling_B(Xv), bv), "test": eval_B(ceiling_B(Xt), bt)}
    cv_top = out["ceiling_A"]["val"]["top1"]
    target = {"ceil-1.0": cv_top - 0.010, "ceil-0.5": cv_top - 0.005}
    out["target_val_top1"] = target
    print("ceilings", out["ceiling_A"], out["ceiling_B"])

    # ridge, task A
    (G, H), tg = timed(lambda: gram_pass(Xf, cf, C))
    rs = {}
    for lam in LAMS:
        W = ridge_solve(G, H, lam, N)
        rs[lam] = eval_logits_A(lambda x: x @ W, Xv, cv)
    lam_r = max(rs, key=lambda l: rs[l]["top1"])
    Wr = ridge_solve(G, H, lam_r, N)
    beta = fit_scale(lambda x: x @ Wr, Xv, cv)
    out["ridge_A"] = {"val_by_lambda": {str(k): v for k, v in rs.items()}, "lambda": lam_r, "beta": beta,
                      "gram_pass_seconds_unlocked": tg,
                      "val": eval_logits_A(lambda x: beta * (x @ Wr), Xv, cv),
                      "test": eval_logits_A(lambda x: beta * (x @ Wr), Xt, ct)}
    # exactness: gradient of the ridge objective at the solution, and chunk-order invariance of G
    Wd = torch.cholesky_solve(H / N, torch.linalg.cholesky(G / N + lam_r * torch.eye(D, dtype=torch.float64, device=DEV)))
    grad = (G / N) @ Wd - H / N + lam_r * Wd
    G2, _ = gram_pass(Xf.flip(0), cf.flip(0), C)
    out["ridge_A"]["checks"] = {"rel_grad_norm_at_solution_fp64": float(grad.norm() / (H / N).norm()),
                                "G_reverse_chunk_order_rel_diff": float((G - G2).norm() / G.norm()),
                                "cond_G_plus_lam": float(torch.linalg.cond(G / N + lam_r * torch.eye(D, dtype=torch.float64, device=DEV)))}
    print("ridge", out["ridge_A"]["lambda"], out["ridge_A"]["val"], out["ridge_A"]["test"], out["ridge_A"]["checks"])

    # AdamW CE and MSE, sweep on val
    for kind in ("ce", "mse"):
        runs = []
        for lr in (1e-3, 3e-3, 1e-2):
            for wd in (0.0, 0.01):
                W, hist, reach = adamw_run(Xf, cf, Xv, cv, kind, lr, wd, target)
                runs.append({"lr": lr, "wd": wd, "best_val_top1": max(h["val_top1"] for h in hist), "reach": reach, "hist": hist})
                print(kind, lr, wd, runs[-1]["best_val_top1"], reach)
        best = max(runs, key=lambda r: (len(r["reach"]), -min([v["passes"] for v in r["reach"].values()] or [99]), r["best_val_top1"]))
        W, hist, reach = adamw_run(Xf, cf, Xv, cv, kind, best["lr"], best["wd"], target)
        out[f"adamw_{kind}_A"] = {"runs": runs, "chosen": {"lr": best["lr"], "wd": best["wd"]},
                                  "val": eval_logits_A(lambda x: x @ W, Xv, cv), "test": eval_logits_A(lambda x: x @ W, Xt, ct), "reach": reach}

    # L-BFGS full batch CE
    lb = {}
    for lam in (1e-5, 1e-4):
        W, hist, reach = lbfgs_run(Xf, cf, Xv, cv, lam, target)
        lb[lam] = {"hist": hist, "reach": reach, "W": W}
        print("lbfgs", lam, hist[-1], reach)
    lam_l = max(lb, key=lambda l: (len(lb[l]["reach"]), max(h["val_top1"] for h in lb[l]["hist"])))
    W = lb[lam_l]["W"]
    out["lbfgs_ce_A"] = {"lambda": lam_l, "runs": {str(k): {"hist": v["hist"], "reach": v["reach"]} for k, v in lb.items()},
                         "val": eval_logits_A(lambda x: x @ W, Xv, cv), "test": eval_logits_A(lambda x: x @ W, Xt, ct), "reach": lb[lam_l]["reach"]}

    # Bohning MM: cold start, and warm start from the scaled ridge head
    for name, W0 in (("bohning_cold_A", None), ("bohning_warm_A", beta * Wr)):
        bm = {}
        for lam in (1e-5, 1e-4):
            W, hist, reach = bohning_run(Xf, cf, Xv, cv, lam, target, G, W0=W0, max_passes=100)
            bm[lam] = {"hist": hist, "reach": reach, "W": W}
            print(name, lam, hist[-1], reach)
        lam_b = max(bm, key=lambda l: (len(bm[l]["reach"]), max(h["val_top1"] for h in bm[l]["hist"])))
        W = bm[lam_b]["W"]
        out[name] = {"lambda": lam_b, "runs": {str(k): {"hist": v["hist"], "reach": v["reach"]} for k, v in bm.items()},
                     "val": eval_logits_A(lambda x: x @ W, Xv, cv), "test": eval_logits_A(lambda x: x @ W, Xt, ct), "reach": bm[lam_b]["reach"]}

    # task B
    Gb, Hb = gram_pass(Xf, bf, 1)
    rb = {}
    for lam in LAMS:
        w = ridge_solve(Gb, Hb, lam, N)[:, 0]
        rb[lam] = eval_B(Xv @ w - 0.5, bv)  # ridge on 0/1 labels: threshold 0.5 -> logit sign
    lam_b = max(rb, key=lambda l: rb[l]["auc"])
    w = ridge_solve(Gb, Hb, lam_b, N)[:, 0]
    out["ridge_B"] = {"lambda": lam_b, "val": rb[lam_b], "test": eval_B(Xt @ w - 0.5, bt),
                      "note": "AUC and balanced accuracy only; ridge outputs are not log-odds, NLL is not meaningful"}
    ib = {}
    for lam in (1e-5, 1e-4, 1e-3):
        w, hist = irls_run(Xf, bf, lam)
        ib[lam] = {"w": w, "hist": hist, "val": eval_B(Xv @ w, bv)}
    lam_i = min(ib, key=lambda l: ib[l]["val"]["nll"])
    w = ib[lam_i]["w"]
    out["irls_newton_B"] = {"lambda": lam_i, "steps": len(ib[lam_i]["hist"]), "hist": ib[lam_i]["hist"],
                            "val": ib[lam_i]["val"], "test": eval_B(Xt @ w, bt)}
    ab = {}
    for lr in (1e-3, 3e-3, 1e-2):
        w = adamw_B(Xf, bf, lr, 0.0, 10)
        ab[lr] = {"w": w, "val": eval_B(Xv @ w, bv)}
    lr_b = min(ab, key=lambda l: ab[l]["val"]["nll"])
    out["adamw_bce_B"] = {"lr": lr_b, "epochs": 10, "val": ab[lr_b]["val"], "test": eval_B(Xt @ ab[lr_b]["w"], bt)}
    print("B", out["ridge_B"]["test"], out["irls_newton_B"]["test"], out["adamw_bce_B"]["test"])
    out["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    out["X_fit_bytes_fp32"] = X_BYTES
    receipt("e1-sweep", out)

elif MODE == "time":
    sw = json.load(open(os.path.join(EVID, "e1-sweep.json")))
    # The sweep's ceiling targets (frozen LM head minus 0.5 / 1.0 points on val) were reached by no method, so timing
    # uses two reachable ones: the ridge head's own val quality, and the best AdamW-CE val quality minus 0.5 points.
    target = {"ridge_quality": sw["ridge_A"]["val"]["top1"],
              "best_ce_minus_0.5": max(r["best_val_top1"] for r in sw["adamw_ce_A"]["runs"]) - 0.005}
    reps = []
    for r in range(3):
        row = {}
        torch.cuda.reset_peak_memory_stats()
        (G, H), row["ridge_gram_pass"] = timed(lambda: gram_pass(Xf, cf, C))
        Wr, row["ridge_solve"] = timed(lambda: ridge_solve(G, H, sw["ridge_A"]["lambda"], N))
        row["ridge_total"] = row["ridge_gram_pass"] + row["ridge_solve"]
        row["ridge_peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        for kind in ("ce", "mse"):
            ch = sw[f"adamw_{kind}_A"]["chosen"]
            torch.cuda.reset_peak_memory_stats()
            (_, _, reach), t = timed(lambda: adamw_run(Xf, cf, Xv, cv, kind, ch["lr"], ch["wd"], target))
            row[f"adamw_{kind}"] = {"reach": reach, "total": t, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        torch.cuda.reset_peak_memory_stats()
        (_, _, reach), t = timed(lambda: lbfgs_run(Xf, cf, Xv, cv, sw["lbfgs_ce_A"]["lambda"], target))
        row["lbfgs_ce"] = {"reach": reach, "total": t, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        torch.cuda.reset_peak_memory_stats()
        beta = sw["ridge_A"]["beta"]
        (_, _, reach), t = timed(lambda: bohning_run(Xf, cf, Xv, cv, sw["bohning_warm_A"]["lambda"], target, G, W0=beta * Wr, max_passes=100))
        row["bohning_warm"] = {"reach": reach, "total": t, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        reps.append(row)
        print(r, json.dumps(row)[:400])
    receipt("e1-time", {"target_val_top1": target, "repeats": reps, "note": "times include per-epoch/per-step val evaluation for the iterative methods; ridge has none",
                        "X_fit_bytes_fp32": X_BYTES})

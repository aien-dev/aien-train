# fp4sim.py (L284 own-fp4, ORACLE ONLY, never shipped): simulated 4-bit numerics for training a small transformer.
# Every quantizer returns the dequantized tensor split as (q * block_scale) in bf16-exact form plus a float32 per-tensor
# scale, so the GEMM is run the way the GB10's OMMA.SF does it: block-scaled 4-bit products accumulated in FP32, then the
# per-tensor scale applied once to the output.
import math, torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_grid = {}
def grid(dev):
    if dev not in _grid: _grid[dev] = E2M1.to(dev)
    return _grid[dev]

def round_e2m1(y, stochastic=False, gen=None):
    """Same results as round_e2m1_ref, by arithmetic: the e2m1 grid has spacing 0.5 below 2, 1 in [2, 4), 2 in [4, 6].
    Nearest: torch.round is half-to-even, and an even multiple of the spacing is an even code in every region.
    Stochastic: floor(a / step + u) * step lands on the two neighbouring grid points with the right odds."""
    a = y.abs().clamp(max=6.0)
    step = torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0))
    if stochastic:
        u = torch.rand(a.shape, device=a.device, generator=gen)
        r = a / step
        q = torch.minimum(torch.floor(r + u), torch.ceil(r)) * step  # never past the upper neighbour (fp32 rounding of r + u)
    else:
        q = torch.round(a / step) * step
    return torch.copysign(q.clamp(max=6.0), y)

def round_e2m1_ref(y, stochastic=False, gen=None):
    """y already divided by its scale; returns values on the e2m1 grid (saturating at +-6).
    RNE: ties go to the code with an even last bit (codes 0..7 = grid index). SR: up with probability (y-lo)/(hi-lo)."""
    g = grid(y.device); s = torch.sign(y); a = y.abs().clamp(max=6.0)
    i = torch.bucketize(a, g, right=True).clamp(1, 7) - 1          # lo index
    lo = g[i]; hi = g[(i + 1).clamp(max=7)]
    if stochastic:
        p = torch.where(hi > lo, (a - lo) / (hi - lo), torch.zeros_like(a))
        r = torch.rand(a.shape, device=a.device, generator=gen)
        q = torch.where(r < p, hi, lo)
    else:
        mid = (lo + hi) / 2
        up = (a > mid) | ((a == mid) & (((i + 1) % 2) == 0))
        q = torch.where(up, hi, lo)
    return s * q

def e4m3(x):  # round to FP8 E4M3 (RNE, finite), as the hardware's UE4M3 scale
    return x.clamp(max=448.0).to(torch.float8_e4m3fn).to(torch.float32)

def _blocks(x, dim, b):
    """move dim last and split it into blocks of b: returns view (..., n/b, b) and an undo function"""
    xt = x.transpose(dim, -1)
    sh = xt.shape
    assert sh[-1] % b == 0, (sh, b)
    return xt.reshape(*sh[:-1], sh[-1] // b, b), (lambda y: y.reshape(sh).transpose(dim, -1))

def quant_nv(x, dim, stochastic=False, gen=None, block=16, scale_mode="amax"):
    """NVFP4: e2m1 values, one UE4M3 scale per `block` elements along `dim`, one FP32 tensor scale.
    Returns (qs, g): qs = q*blockscale (bf16-exact), g = FP32 tensor scale; x ~= qs * g."""
    x = x.float()
    amax_t = x.abs().max().clamp(min=1e-30)
    g = amax_t / (6.0 * 448.0)
    xb, undo = _blocks(x, dim, block)
    amax_b = xb.abs().amax(-1, keepdim=True)
    if scale_mode == "amax":
        sb = e4m3(amax_b / 6.0 / g)
        sb = torch.where(sb == 0, torch.ones_like(sb), sb)
        q = round_e2m1(xb / (sb * g), stochastic, gen)
    else:  # "mse": best of 4 clip ratios per block (our candidate d); RNE only
        best = None
        for c in (6.0, 5.0, 4.5, 4.0):
            s = e4m3(amax_b / c / g); s = torch.where(s == 0, torch.ones_like(s), s)
            qq = round_e2m1(xb / (s * g))
            err = ((qq * s * g - xb) ** 2).sum(-1, keepdim=True)
            if best is None: best = (err, qq, s)
            else:
                m = err < best[0]
                best = (torch.where(m, err, best[0]), torch.where(m, qq, best[1]), torch.where(m, s, best[2]))
        q, sb = best[1], best[2]
    return undo(q * sb), g

def quant_nv2d(w, stochastic=False, gen=None):
    """16x16 2D blocks for weights (NVFP4 paper): one UE4M3 scale per 16x16 tile, replicated to 1x16 rows for the
    tensor core; the same quantized W serves Fprop (contract over in) and Dgrad (contract over out)."""
    w = w.float(); O, I = w.shape
    amax_t = w.abs().max().clamp(min=1e-30); g = amax_t / (6.0 * 448.0)
    t = w.reshape(O // 16, 16, I // 16, 16)
    amax_b = t.abs().amax(dim=(1, 3), keepdim=True)
    sb = e4m3(amax_b / 6.0 / g); sb = torch.where(sb == 0, torch.ones_like(sb), sb)
    q = round_e2m1(t / (sb * g), stochastic, gen)
    return (q * sb).reshape(O, I), g

def quant_mx(x, dim, stochastic=False, gen=None):
    """MXFP4 (OCP): e2m1, one E8M0 power-of-two scale per 32 along dim, X = 2^(floor(log2 amax) - 2), no tensor scale."""
    x = x.float(); xb, undo = _blocks(x, dim, 32)
    amax_b = xb.abs().amax(-1, keepdim=True).clamp(min=2.0 ** -126)
    s = torch.exp2(torch.floor(torch.log2(amax_b)) - 2)
    q = round_e2m1(xb / s, stochastic, gen)
    return undo(q * s), torch.ones((), device=x.device)

def ternary(w, per="tensor"):
    """BitNet b1.58 absmean ternary: W ~= alpha * round(clip(W/alpha, -1, 1)), values {-1,0,1} are e2m1 codes."""
    w = w.float()
    if per == "tensor":
        a = w.abs().mean().clamp(min=1e-8)
        return (w / a).round().clamp(-1, 1) * a, torch.ones((), device=w.device)
    O, I = w.shape; t = w.reshape(O // 16, 16, I // 16, 16)
    a = e4m3(t.abs().mean(dim=(1, 3), keepdim=True).clamp(min=1e-8))
    return ((t / a).round().clamp(-1, 1) * a).reshape(O, I), torch.ones((), device=w.device)

def hadamard(n, dev):
    h = torch.ones(1, 1)
    while h.shape[0] < n: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(dev)

def rotate_tokens(x, R):
    """apply orthogonal R (b x b) to blocks of b consecutive rows (the token / contraction dim of Wgrad)"""
    T, C = x.shape; b = R.shape[0]
    return (R @ x.float().reshape(T // b, b, C)).reshape(T, C)

def rel_err(a, b):
    return ((a - b).float().norm() / b.float().norm().clamp(min=1e-30)).item()

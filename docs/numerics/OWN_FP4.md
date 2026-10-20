# Our own 4-bit training numerics for the GB10 (L284, own-fp4)

Status: DRAFT, parts fill in as receipts land. Every number is tagged **spec** (maker's page), **assembler** (NVIDIA's own
assembler run here, the strongest evidence short of the chip for "what the chip accepts"), **measured** (run on this
Spark's GB10 through Omega, with the receipt path) or **oracle** (a PyTorch simulation used only to choose a design,
never shipped).

Words used below, once each:
- **Tensor core**: the part of the GPU that does small matrix multiplies in one instruction. It is where the GB10's speed is.
- **E2M1 / FP4**: a 4-bit number: 1 sign bit, 2 exponent bits, 1 fraction bit. It can only be 0, 0.5, 1, 1.5, 2, 3, 4, 6
  (and their negatives).
- **Block scale**: one shared multiplier for a small group (block) of 4-bit numbers, so the group can sit at any size.
- **E8M0 / UE4M3**: the two formats the scale can be stored in. E8M0 is a pure power of two (1, 2, 4, 1/2 ...). UE4M3 is
  an 8-bit float with 3 fraction bits, so it can also be 1.25, 1.5 and so on.
- **MXFP4**: the open industry format: 32 numbers share one E8M0 scale. **NVFP4**: NVIDIA's format: 16 numbers share one
  UE4M3 scale, plus one ordinary 32-bit scale for the whole tensor.
- **Rounding**: choosing which of the allowed values a real number becomes. **Nearest** picks the closest (ties to the
  "even" code). **Stochastic** picks up or down at random in proportion to the distance, so on average nothing is lost.
- **GEMM**: a big matrix multiply. Training one layer needs three: forward (Fprop), gradient for the input (Dgrad),
  gradient for the weights (Wgrad).
- **Master copy**: the full-precision copy of the weights the optimizer updates; the 4-bit copy is made from it.
- **Hadamard transform**: a fixed "mixing" matrix of +1 and -1 entries that spreads one big value across a block so
  no single value dominates the block's scale. "Random" Hadamard also flips signs by a fixed random pattern.

## Part 1. What the silicon fixes and what we own

### 1.1 What the GB10 tensor core accepts (assembler, then chip)

Method: write each candidate instruction in NVIDIA's portable assembly (PTX), hand it to NVIDIA's assembler for this
chip (`ptxas -arch=sm_121a`, CUDA 13.0 build 36424714), and read back the machine instruction it produced
(`nvdisasm`). The assembler is used here as a research oracle only; Omega never links it. Receipt:
`evidence/own-fp4/asm/MATRIX.txt` with every source and disassembly beside it.

| What we asked for (PTX) | GB10 answer (assembler) | Meaning |
|---|---|---|
| `mma.m16n8k32 kind::f8f6f4` e2m1 x e2m1 | `QMMA.16832.F32.E2M1.E2M1` | 4-bit values carried in 8-bit slots, **same instruction and same work per instruction as FP8**. Saves memory, not time. |
| same, e2m1 x e4m3 (mixed) | `QMMA.16832.F32.E2M1.E4M3` | 4-bit weights times 8-bit activations is allowed in one instruction. |
| `kind::mxf8f6f4 block_scale scale_vec::1X ue8m0` | `QMMA.SF.16832 ... E8` | 8-bit-slot op with a hardware block scale, 32 values per scale. |
| `m16n8k64 kind::mxf4 scale_vec::2X ue8m0` | `OMMA.SF.16864.F32.E2M1.E2M1.E8` | **The fast 4-bit op**: twice the depth (k 64) per instruction, MX scales, 32 values per scale. |
| `kind::mxf4nvf4 scale_vec::4X ue4m3` | `OMMA.SF.16864 ... UE4M3.4X` | Same fast op with NVFP4 scales, 16 values per scale. |
| `kind::mxf4nvf4 scale_vec::2X ue8m0` | `OMMA.SF.16864 ... E8` | MX-style scales through the NV path. |
| `scale_vec::4X ue8m0` (16 per power-of-two scale) | rejected: needs PTX 9.1, this assembler is 9.0 | The manual says it exists from PTX 9.1 [2]; not testable with the toolkit on this box. |
| `scale_vec::2X ue4m3`, `kind::mxf4 scale_vec::1X` | rejected | No 32-per-UE4M3 and no 64-per-scale option. |
| sparse `mma.sp m16n8k128 mxf4nvf4` | `OMMA.SF.SP.168128` | 2:4 sparse fast op exists (this is where NVIDIA's "1 petaFLOP" comes from [1]); noted, not chased here. |
| `cvt.rn.satfinite.e2m1x2.f32` | `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` | Hardware convert to FP4 with nearest rounding: yes. |
| `cvt.rs.satfinite.e2m1x4.f32` (stochastic rounding) | **rejected on sm_121a and sm_120a**; accepted on sm_100a and sm_103a (`F2FP ... .RS`) | **The GB10 has no hardware stochastic rounding.** NVIDIA's paper says "Blackwell GPUs have native support for ... stochastic rounding for FP4 conversion instructions" [3]; that is true of the data-centre chips, not of this one. On GB10 stochastic rounding is software (random bits + integer ops on the general cores). |
| `cvt.rp/rz.satfinite.ue8m0x2.f32` | `F2FP ... E8 ... .RP/.RZ` | Hardware convert to a power-of-two scale, rounding up or toward zero. |
| `tcgen05.*` (the data-centre Blackwell tensor-memory MMA) | rejected | GB10 uses warp-level `mma.sync` only. |
| `mma ... s4.s4` (4-bit integers) | emulated with `IMMA.16832.S8.S8` plus unpacking | No native 4-bit integer path. |

What the manual says about precision [2]: "Element-wise multiplication of matrix A and B is performed with specified
precision. Accumulation of the intermediate values is performed with at least single precision." The block scales are
applied by the hardware inside the instruction: "D = (A * scale_A) * (B * scale_B) + C" [2]. Scales are per row of A and
per column of B along the multiply's inner dimension (shapes M x SFA_N and SFB_M x N) [2]. So the silicon applies **only
1-D scales along the inner dimension**. NVIDIA's "16 x 16 two-dimensional" weight blocks are a software choice made
before the multiply: "2D block scales are replicated for each of the 1×16 blocks when being passed into Tensor Cores" [3].

### 1.2 Measured rates (chip, through Omega)

| Instruction | Dense rate (measured) | Ratio to FP8 | Receipt |
|---|---|---|---|
| HMMA.16816 BF16 | 118.81 TFLOPS | 0.50 | omega `doc/machine-ceiling@3c8ce239` `evidence/MACHINE-CEILING-20261020/raw/gpu-tc-peak.txt` (L151) |
| QMMA.16832 FP8 E4M3 | 237.52 TFLOPS | 1.00 | same, `raw/gpu-qmma.txt` (L151) |
| QMMA.16832 E2M1 (no block scale) | PENDING | | `evidence/own-fp4/omma-peak/` |
| QMMA.SF.16832 E2M1 E8 | PENDING | | |
| OMMA.SF.16864 MXFP4 (E8, 32/scale) | PENDING | | |
| OMMA.SF.16864 NVFP4 (UE4M3, 16/scale) | PENDING | | |
| Maker's figure | "up to 1 petaFLOP of AI computing performance" with FP4 [1]; L151 quotes NVIDIA's hardware page as "with sparsity" | | spec |

How the 4-bit probe works: Omega had no FP4 tensor-core code. The new probe (omega branch `numerics/own-fp4-omma`,
`src/omega_blackwell_codegen_omma.c`, `tests/gpu_omma_peak_test.c`) emits each 4-bit instruction by taking the exact
bits the assembler produced (above) and patching them into Omega's own encoder, the same method Omega already used for
FP8 (QMMA). It then runs long chains of the instruction with constant inputs and a non-trivial scale (2.0) and checks
every output equals the exact expected value, so a scale the chip ignored or misread shows up as a wrong number, not as
a fast time.

### 1.3 The knobs: fixed by silicon or ours

| Knob | Fixed by silicon or ours | What NVFP4 chose [3] | What else is possible on GB10 |
|---|---|---|---|
| Element format of the fast path | **silicon**: E2M1 only (OMMA.SF) | E2M1 | E2M1 x E4M3 mixed runs only on the FP8-rate op. Ternary {-1, 0, +1} is a subset of E2M1, so ternary weights ride the fast op for free. |
| Block length along the inner dimension | **silicon**: 32 (E8M0) or 16 (UE4M3, or E8M0 with PTX 9.1) | 16 | 32 |
| Scale encoding | **silicon**: E8M0 or UE4M3 | UE4M3 + one FP32 per tensor | E8M0 (exact powers of two, cheaper to compute, coarser) |
| Scale shape beyond 1-D | **ours** (software, before the multiply) | 16x16 tiles for weights, 1x16 for activations and gradients | any tiling that is constant along 16 or 32 inner elements; extra per-row/per-column factors applied after the multiply |
| Accumulator | **silicon**: at least FP32 [2] | FP32 | exact accumulation order and rounding are unspecified by the manual (UNKNOWN; the next chip probe measures it) |
| How values get to the grid (rounding) | **ours**: nearest is a hardware convert; stochastic is software on GB10 | nearest for weights and activations, stochastic for gradients | stochastic everywhere, nearest everywhere, error feedback |
| Transform before quantizing | **ours** | 16x16 random Hadamard on Wgrad inputs only | none, fixed Hadamard, per-token power-of-two balancing (exact and nearly free), rotate all three GEMMs |
| How scales are chosen | **ours** | block max / 6 | best-of-several clip per block (MSE), percentile, learned |
| Which tensors are 4-bit | **ours** | all linear layers except the last ~15% of blocks; attention, embeddings, head, norms in BF16 | everything, or fewer |
| Master copy and optimizer state | **ours** | FP32 master, FP32 Adam states | 4-bit weight + 8-bit residual (error feedback); 8-bit optimizer states |
| Not multiplying at all (ternary) | **ours** | not used | BitNet b1.58 ternary weights [6]; on GB10 run on the 4-bit tensor core, not as adds |

## Part 2. Candidate designs of our own

Notation for one linear layer: input X (tokens x in), weight W (out x in), output Y = X W^T, incoming gradient dY.
The three GEMMs: Fprop Y = X W^T (sums over "in"), Dgrad dX = dY W (sums over "out"), Wgrad dW = dY^T X (sums over
tokens). q(.) is "quantize to E2M1 with block scales along the summed dimension"; qn uses nearest rounding, qs
stochastic. g is the FP32 per-tensor scale. The tensor core computes sum_k (a_k s_a)(b_k s_b) in FP32 for each 16- or
32-long block; g_a g_b is applied once to the result.

**Yardstick, NVFP4 as published [3].** Fprop: Y = qn(X) qn2D(W)^T. Dgrad: dX = qs(dY) qn2D(W). Wgrad:
dW = qs(R dY)^T qn(R X), with R a 16x16 Hadamard times one fixed random sign vector, applied in blocks of 16 tokens
(R^T R = I, so in exact arithmetic dW is unchanged). FP32 master W and FP32 Adam states. Last ~15% of blocks, attention,
embeddings, head and norms in BF16. Bytes per parameter per optimizer step: read+write master 8, read gradient 4,
read+write two Adam moments 16, write the 4-bit copy 0.5 + 1/256 scale = **28.504 B** (calc:538798c3).

**(a) Error feedback instead of an FP32 master.** Stored per weight: c = 4-bit code (with its 16x16 UE4M3 scale) and
r = an 8-bit residual (E4M3 with one power-of-two scale per 16x16 tile). The weight the model sees is w = deq(c); the
optimizer's "master" is m = deq(c) + deq(r). Each step, with Adam update u:
m' = deq(c) + deq(r) + u;  c' = qn2D(m');  r' = q8(m' - deq(c')).
So the rounding error of the 4-bit weight is carried forward in r instead of being thrown away; only the error of the
8-bit residual (about 1/16 of the residual, itself at most half a 4-bit step) is lost. Variant (a-SR) rounds r
stochastically, so even updates smaller than one residual step survive on average. Negative control (a0): no residual
at all (weights live only in 4 bits; any update smaller than half a 4-bit step is lost). Bytes per step:
1.5 + 1.5 + 4/256 (codes, residual, two scales) + 4 (gradient) + 16 (Adam) = **23.016 B** (calc:f959c88e), 0.8075 of
NVFP4's (calc:6b1f86f5). The advisor's point stands: the master is only a fraction of the budget; with 8-bit Adam
moments the pair becomes **11.016 B** vs **16.504 B** (calc:b0eef921, calc:c9f6b25a). Kernel sketch for Omega: one
fused "update" kernel per weight tensor, one CTA per 16x16 tile: load c (128 B), r (256 B), two scale bytes, the FP32
gradient tile and Adam moments; FFMA the Adam math; find the tile max (warp shuffle); F2FP.E2M1 (hardware nearest
convert, present on GB10) for c'; subtract; F2FP.E4M3 for r' (or software stochastic rounding: add random bits from a
counter-based generator, then truncate); store. Rate: memory-bound, no tensor core. Known failure mode: if r's own
rounding loses updates systematically (RNE), small learning rates late in training stall; the race checks this.

**(b) Our own outlier handling for Wgrad instead of the random Hadamard.** Candidates, all exact in real arithmetic
(dW unchanged):
- signs only (diagonal of +-1): **cannot help, by algebra**: block-max scaling and rounding of |x| do not depend on the
  sign, so the quantization error is identical. Kept in the study as a control that must show no change.
- fixed Hadamard without the random signs (16x16).
- per-token power-of-two balancing: X_t -> X_t s_t, dY_t -> dY_t / s_t with s_t = 2^round(log2 sqrt(|dY_t|/|X_t|)).
  Because s_t is a power of two this is exact on the chip (an exponent add), costs 1 operation per element instead of
  16 multiply-adds, and needs no tensor-core pass. It evens out tokens whose gradient is huge against tokens whose
  gradient is tiny, which is the outlier pattern along the token dimension.
- balancing followed by the 16x16 RHT; 128x128 RHT (the paper's larger option).
Measured on real Wgrad operands from the BF16 baseline (4 layers x 3 points in training): see Part 3, table B.
Kernel sketch: the balancing is folded into the quantize kernel that already reads X and dY (one extra FMUL by a power
of two per element, one row-norm reduction per token); the RHT is a 16x16 GEMM per tile (on GB10 it can run as an
HMMA, 32 flops per element).

**(c) Ternary weights.** W ~ a * T with T in {-1, 0, +1} and a = mean|W| (BitNet b1.58 absmean [6]) per tensor (c) or
per 16x16 tile (c16, which maps onto the NVFP4 hardware block scale: the tile scale becomes the UE4M3 scale). Forward:
Y = qn(X) (aT)^T; Dgrad uses the same aT; Wgrad as NVFP4 (it has no weight in it, so ternary does not change it).
Latent FP32 weights and Adam states as in BitNet ("straight-through": the gradient passes through the rounding as if it
were not there). On GB10, -1, 0, +1 are exact E2M1 codes, so the ternary GEMM runs on the **fast 4-bit tensor core**,
not as adds on the general cores (the adds path would run at the general-core rate, which L151 measured at 17.69 TFLOPS
FFMA, 0.074 of the FP8 tensor rate: an adds-only kernel cannot beat the tensor core here). Storage: 1.58 bits
of information, 2 bits packed, 4 bits as a tensor-core operand. Bytes per optimizer step: unchanged from NVFP4 (FP32
latent weights): the advisor is right that ternary is a win for inference memory and forward/Dgrad operand bytes, not
for training optimizer traffic. Omega already has a ternary W2A8 GEMV on the general cores
(`src/omega_blackwell_codegen_tern.c`, omega main), the inference-side half. Known failure mode: at equal parameter
count ternary loses quality; BitNet reports parity only at larger sizes [6].

**(d) Rotate everything + best-clip scales (from Quartet [4] and the MXFP4 recipe [5]).** Quartet keeps all linear
layers in FP4 with a Hadamard rotation in the forward pass and stochastic rounding in the backward [4]; Tseng et al.
use RHT to bound the variance of stochastic rounding [5]. Ours: R applied along the summed dimension of all three GEMMs,
and for forward tensors each 16-block's scale picked as the best of four clip ratios (amax/6, /5, /4.5, /4) by squared
error. Cost: one extra 16x16 rotation per GEMM input (HMMA) and four trial quantizations in the quantize kernel
(general cores). Failure mode: clipping hurts when the clipped values carry the signal.

Microsoft's FP4 framework [7] (a differentiable gradient estimator and outlier clamping with compensation) needs an extra
high-precision "compensation" matrix multiply; it is noted, not raced here (budget), and listed as a next experiment.

Tensor-core rate each design can use: all of (a)-(d) feed the same OMMA.SF.16864 instruction in all three GEMMs
(rate in 1.2). The differences are in the side kernels (quantize, transform, update), which run on the general cores
or as small HMMA tiles.

## Part 3. The race (oracle)

PENDING

## Part 4. Verdict

PENDING

## Sources

[1] https://www.nvidia.com/en-us/products/workstations/dgx-spark/ , "NVIDIA DGX Spark", accessed 2026-10-20. "Fifth-generation Tensor Cores with support for FP4 deliver up to 1 petaFLOP of AI computing performance, combined with 128 GB of system mem[ory]"
[2] https://docs.nvidia.com/cuda/parallel-thread-execution/index.html , "PTX ISA" (section 9.7.16.3 Block Scaling for mma.sync, and mma), accessed 2026-10-20. "D = (A * scale_A) * (B * scale_B) + C." / "Element-wise multiplication of matrix A and B is performed with specified precision. Accumulation of the intermediate values is performed with at least single precision." / "Support for .scale_vec::4X on .ue8m0 as .stype with .kind::mxf4nvf4 is introduced in PTX ISA version 9.1."
[3] https://arxiv.org/abs/2509.25149 (HTML https://arxiv.org/html/2509.25149), NVIDIA, "Pretraining Large Language Models with NVFP4", accessed 2026-10-20. "Our method integrates Random Hadamard transforms (RHT) to bound block-level outliers, employs a two-dimensional quantization scheme for consistent representations across both the forward and backward passes, utilizes stochastic rounding for unbiased gradient estimation, and incorporates selective high-precision layers." / "Further, Blackwell GPUs have native support for several rounding modes including round-to-nearest-even and stochastic rounding for FP4 conversion instructions." / "2D block scales are replicated for each of the 1×16 blocks when being passed into Tensor Cores, and continue to leverage an FP32 per-tensor scale."
[4] https://arxiv.org/abs/2505.14669 , Castro et al., "Quartet: Native FP4 Training Can Be Optimal for Large Language Models", accessed 2026-10-20. "In this paper, we investigate hardware-supported FP4 training and introduce a new approach for accurate, end-to-end FP4 training with all the major computations (i.e., linear layers) in low precision."
[5] https://arxiv.org/abs/2502.20586 , Tseng et al., "Training LLMs with MXFP4", accessed 2026-10-20. "Our key insight is to compute unbiased gradient estimates with stochastic rounding (SR), resulting in more accurate model updates. However, directly applying SR to MXFP4 can result in high variance from block-level outliers, harming convergence. To overcome this, we use the random Hadamard tranform to theoretically bound the variance of SR."
[6] https://arxiv.org/abs/2402.17764 , Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", accessed 2026-10-20. "In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single parameter (or weight) of the LLM is ternary {-1, 0, 1}. It matches the full-precision (i.e., FP16 or BF16) Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance"
[7] https://arxiv.org/abs/2501.17116 , Wang et al., "Optimizing Large Language Model Training Using FP4 Quantization", accessed 2026-10-20. "This work introduces the first FP4 training framework for LLMs, addressing these challenges with two key innovations: a differentiable quantization estimator for precise weight updates and an outlier clamping and compensation strategy to prevent activation collapse."

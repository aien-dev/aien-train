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

(equations and kernel sketches: see below; race numbers in Part 3)

## Part 3. The race (oracle)

PENDING

## Part 4. Verdict

PENDING

## Sources

[1] https://www.nvidia.com/en-us/products/workstations/dgx-spark/ , "NVIDIA DGX Spark", accessed 2026-10-20. "Fifth-generation Tensor Cores with support for FP4 deliver up to 1 petaFLOP of AI computing performance, combined with 128 GB of system mem[ory]"
[2] https://docs.nvidia.com/cuda/parallel-thread-execution/index.html , "PTX ISA" (section 9.7.16.3 Block Scaling for mma.sync, and mma), accessed 2026-10-20. "D = (A * scale_A) * (B * scale_B) + C." / "Element-wise multiplication of matrix A and B is performed with specified precision. Accumulation of the intermediate values is performed with at least single precision." / "Support for .scale_vec::4X on .ue8m0 as .stype with .kind::mxf4nvf4 is introduced in PTX ISA version 9.1."
[3] https://arxiv.org/abs/2509.25149 (HTML https://arxiv.org/html/2509.25149), NVIDIA, "Pretraining Large Language Models with NVFP4", accessed 2026-10-20. "Our method integrates Random Hadamard transforms (RHT) to bound block-level outliers, employs a two-dimensional quantization scheme for consistent representations across both the forward and backward passes, utilizes stochastic rounding for unbiased gradient estimation, and incorporates selective high-precision layers." / "Further, Blackwell GPUs have native support for several rounding modes including round-to-nearest-even and stochastic rounding for FP4 conversion instructions." / "2D block scales are replicated for each of the 1×16 blocks when being passed into Tensor Cores, and continue to leverage an FP32 per-tensor scale."

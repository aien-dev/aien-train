# Training on one GB10: what fits on the math cores, and how fast it can go (L279)

Drake's question (2026-10-20): can this Spark train our own language model if we cut the software out of the way,
and can the training stay on the "math cores" (the GPU's compute units and the fast storage inside them) instead of
streaming everything through the 128 GB main memory every step?

Short answer, details below:

- **What fits inside the chip is small.** All the fast storage inside the GPU adds up to about 42 MiB (registers
  12 MiB, per-core shared memory and L1 6 MiB, L2 cache 24 MiB). A model whose weights, gradients and optimizer
  state all live there has between about **1.2 and 11 million parameters**, depending on how many bytes each
  parameter costs. That is a toy model: useful for experiments and very small specialist tasks, far from a
  general chat model.
- **For training, the memory pipe is not the main wall.** Unlike answering questions one word at a time (where
  the weights are read once per word and memory speed decides everything), training processes thousands of words
  per pass over the weights. Our measured numbers say a normal training step stops being limited by memory speed
  once it processes about **3,300 words per step**, and normal training uses far more than that. The wall for
  training is how busy we keep the tensor cores.
- **How big, how long, on this one box:** at a realistic 50% of the measured 16-bit tensor rate, a 125-million
  parameter model trained the compute-optimal way takes about **9 hours**, 350 million about **2.9 days**,
  1 billion about **23 days**, 3 billion about **210 days**. At the speed of our best tensor kernel today (30.6
  trillion operations per second, about a quarter of the chip's 16-bit rate) each figure is roughly twice as long.

Every number below says whether it is **spec** (the maker's or a reviewer's figure, cited), **measured** (our own
code on this GB10, with the receipt path), or **computed** (arithmetic on measured numbers, with an aien-calc
receipt id `calc:xxxxxxxx`).

Words used here:
- **SM (streaming multiprocessor):** one of the GPU's compute units. Each has its own tensor cores (matrix-multiply
  units), registers and shared memory.
- **Registers:** the fastest storage, inside each SM, private to each thread.
- **Shared memory / L1:** a 128 KB scratchpad and cache inside each SM, shared by the threads running there.
- **L2 cache:** a 24 MB cache shared by all SMs, still on the chip.
- **Main memory (DRAM):** the 128 GB LPDDR5x the CPU and GPU share. Off the GPU's compute dies, much slower.
- **Parameter:** one number (weight) in the model. A "125M model" has 125 million of them.
- **Gradient:** for each parameter, the direction and size of the change the last batch asks for.
- **Optimizer state:** extra numbers per parameter the update rule keeps between steps (Adam keeps two).
- **Token:** a word or piece of a word; models read and write tokens.

## 1. What is on the chip, and what fits there (Q1)

### 1.1 Inventory

| Item | Value | Kind and source |
|---|---|---|
| CUDA cores (FP32 lanes) | 6,144 | measured through our path: RM `NV2080_CTRL_CMD_GR_GET_INFO` index `GPU_CORE_COUNT` = 6144, physics `evidence/m19r_gate4_forge_hwid_evidence.json` (physics@e95e3ed, code `forge/forge_descriptor.c`); spec agrees: "CUDA Cores: 6,144" [1] |
| Max resident warps per SM | 48 (1,536 threads) | measured through our path: same receipt, `max_warps_per_sm` 48; spec agrees: "Maximum number of resident warps per SM: 48" [2] |
| SMs per TPC | 2 | measured through our path: same receipt, `sm_per_tpc` 2 |
| SM count | 48 | spec: "The iGPU features 48 Streaming Multiprocessors running at up to 2.55 GHz" [3]. Consistent with 6,144 cores at 128 per SM. Direct SM count through our path: tool `streams/physics-gb10/gr_info.c` (read-only RM query: TPC count, SMs per TPC, register banks); receipt `evidence/physics-gb10/gr_info.txt` when the build queue releases it (see 1.4) |
| Registers per SM | 64 K x 32-bit = 256 KiB | spec: "Number of 32-bit registers per SM: 64 K" for compute capability 12.x [2] |
| Shared memory per SM | up to 100 KB (99 KB per block) | spec [2]. **Our launcher today caps a block at 16 KiB** (omega `src/omega_blackwell_qmd.h:41`, issue #560, noted by L272), so Omega kernels cannot yet use the full 100 KB |
| L1 + shared memory per SM | 128 KB | spec: "Unified Data Cache Size (KB): 128" [2]; "has 128 KB of L1/Shared Memory" [3] |
| L2 cache | 24 MB | spec: "GB10's 24 MB L2" [3]; also reported by ServeTheHome, see omega `docs/MACHINE_CEILING.md` (doc/machine-ceiling@3c8ce239) |
| Main memory | 128 GB LPDDR5x, 273 GB/s | spec: "128 GB LPDDR5x unified system memory, 256-bit interface, 4266 MHz, 273 GB/s bandwidth" [1] |
| Main memory read, GPU | **245.1 GB/s** | measured (L151): omega doc/machine-ceiling@3c8ce239, `evidence/MACHINE-CEILING-20261020/raw/gpu-tc-peak.txt`, window 20261020T030919Z-9d9d5a |
| Main memory write, GPU | **175.1 GB/s** | measured (L151): same window, `raw/gpu-ceiling.txt`. Writes matter for training (gradients, optimizer state) |
| L2 read, all SMs | **5,110.8 GB/s** | measured (L151): `raw/gpu-pf-mem.txt` |
| L2 write | "just under 1 TB/s" (half rate) | spec/outside measurement [3]; not measured by us: UNKNOWN row U1 |
| Shared memory read, all SMs | **14.38 TB/s** (16 KB blocks) | measured (L151): `raw/gpu-ceiling.txt` |
| BF16 tensor rate (HMMA.16816.F32.BF16) | **118.81 TFLOPS** | measured (L151): `raw/gpu-tc-peak.txt`. NVIDIA publishes no dense BF16 figure |
| FP8 tensor rate (QMMA.16832.F32.E4M3) | **237.52 TFLOPS** | measured (L151): `raw/gpu-qmma.txt`. NVIDIA publishes "up to 1 PFLOP ... at FP4 precision with sparsity" [1], no dense FP8 figure |
| Best Omega tensor GEMM on a real shape | **30.6 TFLOPS** (FP8, 4096 x 5120 x 6144, loaded box) | measured: handoff `2026-10-19-cascade-prefill.md` line 60, binary qgemm-1f37f9f6. That is 0.129 of the FP8 rate (calc:fb7f5693, from MACHINE_CEILING) |

These ceilings were measured by L151 (cc-machine-ceiling) in one quiet window with our own kernels. L272
(cc-speed-onchip) is re-measuring the on-chip walls for decode in `docs/speed/GB10_CEILINGS.md` on omega
feat/decode-onchip-megakernel; it had not pushed when this was written (checked 2026-10-20 ~18Z). This stream ran
no bandwidth or tensor-rate benchmark of its own, by rule.

### 1.2 The on-chip total

| Storage | Per SM | x 48 SMs | Kind |
|---|---|---|---|
| Registers | 256 KiB | 12,582,912 B = 12 MiB (calc:aac73c4f) | spec per SM [2], x 48 [3] |
| L1 + shared memory | 128 KiB | 6,291,456 B = 6 MiB (calc:9e88360f) | spec per SM [2][3] |
| Inside the SMs | | 18,874,368 B = 18 MiB (calc:b90d90eb) | computed |
| L2 (shared by all SMs) | | 25,165,824 B = 24 MiB | spec [3] (taken as MiB; if the 24 MB is decimal the total drops by 1.2 MB) |
| **Everything on the GPU die** | | **44,040,192 B = 42 MiB** (calc:8bc2b7c3) | computed |

Main memory holds 128 GB. The chip holds about 1/3000 of that.

This total is an upper bound. A real kernel needs registers for its own working values, and L1 shares its space
with shared memory, so the usable amount for stored model numbers is lower. How much lower is a property of the
kernel; the Q3 probe measures one case.

### 1.3 The largest model that fits

Bytes per parameter for each training scheme:

| Scheme | What is kept per parameter | Bytes |
|---|---|---|
| A. Standard mixed precision with Adam | BF16 weight 2 + BF16 gradient 2 + FP32 master weight 4 + FP32 Adam first moment 4 + FP32 second moment 4 | 16 |
| B. BF16 with 8-bit optimizer state | BF16 weight 2 + BF16 gradient 2 + two 8-bit Adam moments 2 (no FP32 master) | 6 |
| C. Muon-style (momentum only, no second moment) | BF16 weight 2 + BF16 gradient 2 + BF16 momentum 2 | 6 |
| D. FP8 throughout | FP8 weight 1 + FP8 gradient 1 + two 8-bit moments 2 | 4 |
| Weights only, BF16 | gradients and state streamed from main memory | 2 |
| Weights only, FP8 | same | 1 |

| Scheme | Fits in all 42 MiB (registers + L1/shared + L2) | Fits inside the SMs only (18 MiB) |
|---|---|---|
| A (16 B) | **2,752,512 parameters** (calc:c630f624) | 1,179,648 (calc:c856b8ed) |
| B or C (6 B) | 7,340,032 (calc:f4ec138e) | 3,145,728 (calc:a62edc13) |
| D (4 B) | 11,010,048 (calc:9108596c) | 4,718,592 (calc:93fb98bb) |
| Weights only, BF16 | 22,020,096 (calc:45c96809) | 9,437,184 (calc:45b867e6) |
| Weights only, FP8 | 44,040,192 | 18,874,368 |

And none of these leave room for the activations (the intermediate values each layer computes in the forward pass
and the backward pass needs again). Those also have to live somewhere.

**The honest size of a "math cores only" model on this chip is about 1 to 11 million parameters** for full
training state on chip, and up to about 22 million (BF16) or 44 million (FP8) if only the weights stay on chip and
the gradients and optimizer state stream through main memory.

What a model that small is good for: narrow tasks with a small vocabulary of behaviour (a classifier, a router that
picks which bigger model to call, a draft model that guesses the next few words for a big model to check, a
controller for one device), research on training methods where many quick runs matter more than quality, and
teaching. It will not hold general knowledge or carry a conversation. The L278 efficiency stream covers the
literature on tiny models; it had not pushed when this was written (origin/study/efficiency was at the skeleton
commit ed8b6ea, checked 2026-10-20 ~18Z).

### 1.4 Open items for Q1

- U0: direct SM count and register-bank geometry through our own RM query. Tool written
  (`streams/physics-gb10/gr_info.c`, read-only: one RM client, one GR_GET_INFO call, no channel, no allocation, no
  launch); its build was queued behind the box's shared build lock at the time of writing. Receipt goes to
  `evidence/physics-gb10/gr_info.txt`.
- U1: L2 write bandwidth on our own path (only an outside figure exists [3]). Owned by L272's ceilings work, not
  this stream.

## 2. The training roofline on this chip (Q2)

### 2.1 The ridge point

The **ridge point** is the number of arithmetic operations a kernel must do per byte it moves to or from main memory
before the math, not the memory, sets its speed. Below it the tensor cores wait for data; above it they are the
limit.

| | Rate | Ridge (FLOPs per byte read) |
|---|---|---|
| BF16 tensor | 118.81 TFLOPS measured | **484.7** (calc:ac4723cb) |
| FP8 tensor | 237.52 TFLOPS measured | **969.1** (calc:05ef5299) |

Both against the measured 245.1 GB/s read rate. Writes are slower (175.1 GB/s measured), which the step model below
accounts for separately.

### 2.2 Work and bytes in one training step

Work: the standard count is **6 x N operations per token** for a model with N parameters: 2N for the forward pass
(one multiply and one add per weight) and about twice that for the backward pass. Kaplan et al.: "Accounting for
the backwards pass (approximately twice the compute as the forwards pass), we then define the estimated
non-embedding compute as C≈6N floating point operators per training token." [4] So a step that processes B tokens
does 6NB operations. This leaves out attention's own work, which grows with context length; at short contexts it is
small next to 6N.

Bytes, scheme A (16 B per parameter), three designs:

**(a) The normal way.** Each step: forward reads the BF16 weights (2N), backward reads them again (2N) and writes
BF16 gradients (2N), the optimizer reads the FP32 master weights, both moments and the gradient (14N) and writes the
master weights, both moments and the new BF16 weights (14N). That is 18N bytes read and 16N bytes written per step,
whatever B is. Time on memory: N x (18 / 245.1 GB/s + 16 / 175.1 GB/s). Time on math: 6NB / 118.81 TFLOPS.
The weight-and-state traffic stops being the limit when

  **B ≥ 3,264 tokens per step** (calc:1bcfc00a; N cancels out).

Large training runs carry far more than that per step (Chinchilla 70B used "1.5M → 3M" tokens per batch [5]), so
in design (a) the weight and optimizer traffic is already a small share of the step. The bytes that grow with B are
the **activations**: Korthikanti et al. count the stored activations of one transformer layer as "sbh(34 + 5as/h)"
bytes (s sequence length, b batch, h hidden width, a attention heads) [6]. With attention done the fused way
(scores never written to memory) the per-token, per-layer figure is 34h bytes, written in the forward pass and read
back in the backward pass. With N ≈ 12 L h² for L layers, work per token is 72 L h²; memory time per token is
34 L h (1/245.1 GB/s + 1/175.1 GB/s). Activations stop being the limit when

  **hidden width h ≥ 549** (calc:743f74c8).

For scale, a 12-layer model of exactly that width has 12 x 12 x 549² = 43,401,744 parameters (calc:8becebee); the
wider models in the table in 2.4 sit above the threshold. So, at the counted byte level, **a dense transformer
training step on the GB10 is compute-bound, not memory-bound**, for every model size in the table in 2.4. The
caveat is that 34h bytes counts only what is stored for the backward pass; a chain of separate kernels (matrix
multiply, then normalise, then activation function, then add) also writes and rereads every intermediate in
between. How many times over that multiplies the 34h figure in our own kernels is not measured: UNKNOWN row U2.
Every factor k of extra traffic raises the width threshold to 549k.

**(b) Layer-resident.** A persistent kernel (one that stays running on the GPU instead of being launched again per
operation) holds one layer's weights on chip for the whole micro-batch, so weights are read once per step per
layer and the gradient for that layer is accumulated on chip; only activations stream. Bytes per step become the
optimizer traffic once (the same 34N as (a) if the optimizer runs every step) plus the activation traffic. The
crossover batch is the same as (a), because (a) already reads weights only once per step; residency helps when
a step is split into micro-batches that would otherwise reread weights and re-add gradients each time. The limit is
the layer size: a whole layer (12h² parameters) fits in 42 MiB as BF16 weights only up to **h = 1,354**
(calc:2b49881b), and with its full scheme A state up to **h = 478** (calc:13ecaa54). The real gain of (b) is fusion:
keeping the intermediates between the matrix multiplies on chip removes the unmeasured factor k above.

**(c) Fully on chip.** Weights, gradients and optimizer state never leave the chip; only the token batch streams in
(4 bytes per token id). Work per byte is 6N / 4 = 1.5N, which passes the BF16 ridge for any model above
**323 parameters** (calc:000271d5). It is always compute-bound. Its limit is size: 2,752,512 parameters at scheme A
(section 1.3). At the measured BF16 rate that model would process up to **7.19 million tokens per second**
(calc:5513f69c) at 100% tensor use.

### 2.3 Time per token and tokens per second

Because designs (a) and (b) are compute-bound for real model widths, their speed is set by the tensor rate and how
much of it the kernels use (called **utilization**: delivered operations divided by the chip's peak). Time per token
= 6N / (utilization x rate).

| Model | Design | Tokens per second at 100% BF16 | at 50% BF16 |
|---|---|---|---|
| 2.75M, scheme A | (c) fully on chip | 7,194,000 (calc:5513f69c) | about 3.6 million |
| 125M | (a) or (b) | 158,410 (calc:234a6d6e) | 79,207 (calc:4856abbc) |

Utilization assumptions, stated: **today** = our best measured tensor GEMM, 30.6 TFLOPS (FP8, real shape, loaded
box; section 1.1); **target BF16** = 50% of the measured 118.81 TFLOPS, about the level large training runs report
(PaLM: "very high efficiency of 46.2% in model FLOPs utilization (observed throughput relative to theoretical max
throughput)" [7]); **target FP8** = 50% of the measured 237.52 TFLOPS, which assumes FP8 training works at our
quality bar (not shown here).

### 2.4 Days to train compute-optimal models on one GB10

**Compute-optimal** (the "Chinchilla" rule): for a given compute budget, model size and training tokens should grow
together, "for compute-optimal training, the model size and the number of training tokens should be scaled
equally" [5]. Their Table 3 gives 20.2 billion tokens for a 1 billion parameter model [5], about 20 tokens per
parameter. With D = 20N tokens, total work = 6N x 20N = 120N² operations.

| Model | Tokens (20N) | Total work | Days, today (30.6 TFLOPS) | Days, BF16 at 50% (59.4 TFLOPS) | Days, FP8 at 50% (118.8 TFLOPS) |
|---|---|---|---|---|---|
| 50M | 1 billion | 3.0e17 | 0.11 (calc:e8e694cf) | 0.058 (calc:b21dcb17) | 0.029 (calc:246c83a8) |
| 125M | 2.5 billion | 1.9e18 | 0.71 (calc:6a08b864) | 0.37 (calc:fa2eb778) | 0.18 (calc:5ed28589) |
| 350M | 7 billion | 1.5e19 | 5.6 (calc:32211650) | 2.9 (calc:a6986476) | 1.4 (calc:837a2552) |
| 1B | 20 billion | 1.2e20 | 45 (calc:96fd4990) | 23 (calc:d76d8642) | 12 (calc:53791ae6) |
| 3B | 60 billion | 1.1e21 | 409 (calc:31e78732) | 210 (calc:d9255171) | 105 (calc:b9679e62) |

Read it as: **this one Spark can train a compute-optimal model of a few hundred million parameters in days, a
1 billion parameter model in two to six weeks, and a 3 billion parameter model only in months.** These are
compute-bound numbers; they assume the data pipeline and memory keep up (2.2 says the memory does at the counted
level), and they leave out attention's context-length work, evaluation and restarts.

Memory capacity is not the limit at these sizes: scheme A at 3B is 48 GB of weights and state, within the 128 GB,
but not alongside today's 27B model server, which holds most of the unified memory. A training run needs the box,
or most of it.

### 2.5 What Q2 means for Drake's idea

For decoding (answering one word at a time), every word reads all the weights from memory, so memory speed is the
wall and keeping things on chip is the whole game; L272 owns that. For training, the same weights serve thousands
of tokens per step, so the counted memory traffic is already below the tensor wall on this chip. The speed of
training on the Spark is decided by **how much of the tensor rate our kernels deliver** (today 0.13 of FP8 peak on
a real shape) and by the **extra traffic between unfused kernels** (U2). That is where "get the bloat out of the
way" pays for training: fused, persistent kernels that keep each layer's intermediates on chip and keep the tensor
cores fed.

## 3. The on-chip training experiment (Q3)

(In progress.)

## 4. Verdict (Q4)

(In progress.)

## UNKNOWN rows

| Id | What is not known | Experiment that resolves it | Owner |
|---|---|---|---|
| U0 | SM count and register banks read through our own RM path | run `streams/physics-gb10/gr_info` (read-only) | L279 |
| U1 | L2 write bandwidth on our path | L2 write stream probe | L272 ceilings |
| U2 | Extra main-memory traffic per token from unfused kernels in a real Omega training step (factor k over the 34h stored-activation count) | count bytes per kernel in one Omega forward+backward layer, then measure | first training-side kernel (section 4) |

## Sources

[1] https://docs.nvidia.com/dgx/dgx-spark/hardware.html , NVIDIA DGX Spark User Guide, Hardware Overview, accessed
2026-10-20. "128 GB LPDDR5x unified system memory, 256-bit interface, 4266 MHz, 273 GB/s bandwidth"; "CUDA Cores:
6,144"; "up to 1 PFLOP (petaFLOP) at FP4 precision with sparsity".

[2] https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html , CUDA Programming
Guide, Compute Capabilities, accessed 2026-10-20. Compute capability 12.x column: "Number of 32-bit registers per
SM: 64 K"; "Maximum amount of shared memory per SM: 100 KB"; "Maximum amount of shared memory per thread block:
99 KB"; "Unified Data Cache Size (KB): 128"; "Maximum number of resident warps per SM: 48".

[3] https://chipsandcheese.com/p/analyzing-nvidia-gb10s-gpu , Chips and Cheese, "Analyzing Nvidia GB10's GPU",
accessed 2026-10-20. "The iGPU features 48 Streaming Multiprocessors running at up to 2.55 GHz"; "GB10 behaves much
like Nvidia's consumer Blackwell architecture and has 128 KB of L1/Shared Memory"; "GB10's 24 MB L2 is likely
almost as effective"; "GB10's L2 handles writes at half rate, and provides just under 1 TB/s of write bandwidth."

[4] https://arxiv.org/html/2001.08361 , Kaplan et al., "Scaling Laws for Neural Language Models", 2020, accessed
2026-10-20. "Accounting for the backwards pass (approximately twice the compute as the forwards pass), we then
define the estimated non-embedding compute as C≈6N floating point operators per training token."

[5] https://arxiv.org/pdf/2203.15556v1 , Hoffmann et al., "Training Compute-Optimal Large Language Models", 2022,
accessed 2026-10-20. "we find that for compute-optimal training, the model size and the number of training tokens
should be scaled equally"; Table 3: "1 Billion ... 1.21e+20 ... 20.2 Billion"; Table 4, Chinchilla 70B batch size: "1.5M → 3M" with caption "the training batch size (# tokens)"; "We verify this by training a more
compute-optimal 70B model, called Chinchilla, on 1.4 trillion tokens."

[6] https://arxiv.org/pdf/2205.05198 , Korthikanti et al., "Reducing Activation Recomputation in Large Transformer
Models", 2022, accessed 2026-10-20. "Summing the memory required for attention, MLP, and the layer-norms, the memory
required to store the activations for a single layer of a transformer network is: Activations memory per layer =
sbh(34 + 5as/h)".

[7] https://arxiv.org/pdf/2204.02311 , Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways", 2022,
accessed 2026-10-20. "we describe how we were able to scale pipeline-free training of PaLM 540B to 6144 chips across
two TPU v4 Pods while achieving very high efficiency of 46.2% in model FLOPs utilization (observed throughput
relative to theoretical max throughput) and 57.8% in hardware FLOPs utilization."

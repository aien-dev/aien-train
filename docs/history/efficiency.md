# Training well with little compute: what is already proven

Stream L278, branch `study/efficiency`, written 2026-10-20 for Drake. Sibling chapters cover the labs (OpenAI, everyone else) and the measured physics of the GB10 chip. This chapter covers one question: what has been shown, by whom, on what hardware, about getting a good language model out of very little compute, and what it would take on one GB10 (the chip inside the DGX Spark).

**In plain words.**
- The best public proof that engineering beats brute force is the GPT-2 speedrun: the same model and the same quality target went from 45 minutes to 0.665 minutes on 8 H100 chips, 67.7x faster (calc:2981eefe) [13]. About 30x of that came from learning more from each token of text, and 2.23x from faster machinery (section 2).
- Better data is worth a factor of several in text needed (section 3). Lower-precision numbers (FP8) are proven at scale; 4-bit is proven once (section 4).
- Your idea of keeping everything on the math cores: the GB10 holds about 42 MiB on-chip, enough for the training state of a 2.8 to 6.3 million parameter model. Anything larger must stream through, and the published ways to make that streaming cheap are in section 5.
- The one real training log on our chip: a 561M-parameter model took 9.8 days on a single DGX Spark and reached about 45% of the chip's measured BF16 speed [64]. The literature-side answer to "can we train on the Spark" is yes for 125M to 350M parameters (hours to days), no for 3B at the Chinchilla ratio (about 150 days at 40% utilization) (section 8).
- Correction to the brief: 207.7 TFLOPS FP8 is a reviewer's measurement, not an NVIDIA spec (section 0).
- Section 9 holds 28 lessons, each with the kernel we would build.


How to read it. Every outside claim has a number in square brackets that points to the Sources list at the end, with the exact sentence copied from the page. Every number names the hardware it was measured on. Arithmetic is shown, and each result carries a receipt id (`calc:xxxxxxxx`) from our exact-arithmetic tool, so it can be checked. `[unverified]` marks a claim I could not source. "Tokens" are the word-pieces a model reads and predicts; "FLOPs" are individual multiply or add operations; "loss" is the model's average surprise at the next token (lower is better).

## 0. The machine, and one correction to the brief

What the vendor states about the DGX Spark: 128 GB of memory shared by the CPU and GPU, 273 GB/s of memory bandwidth, and "up to 1 PFLOP (petaFLOP) at FP4 precision with sparsity" [1][2]. FP4 is a 4-bit number format; "with sparsity" means the figure assumes the matrix has a particular pattern of zeros, so dense work gets about half. NVIDIA's own pages give no FP8 figure [1][2].

**Correction.** The brief calls 207.7 TFLOPS "dense FP8, NVIDIA-stated". It is not. It is a measurement by a reviewer (StorageReview, with the open "mamf-finder" tool, which searches for the largest matrix-multiply speed the chip actually reaches): "At BF16 precision, we observe a MAMF of 99.8 TFLOPs, while FP8 (E4M3) demonstrates a MAMF of 207.7 TFLOPs." [6] I use 207.7 TFLOPS below as a measured best case for one big matrix multiply, not as a vendor spec. Our own handoffs already label it "third-party measured". A real training step also does non-matrix work, so the usable fraction is lower (section 8).

Other chip facts used in this chapter:

| Fact | Value | Source |
|---|---|---|
| CUDA cores | 6,144 | NVIDIA hardware guide [2] |
| Streaming multiprocessors (SMs, the chip's independent compute blocks) | 48 (= 6,144 / 128 cores per SM, matching a retailer listing "48SM/6144SP") | [2][5], secondary; our Omega work also treats it as 48 |
| Tensor cores (the matrix-multiply units) | 192, 5th generation | [5], secondary |
| Registers (the fastest storage, private to each SM) | 64 K 32-bit registers per SM = 256 KB per SM | CUDA guide [4] |
| Shared memory / L1 (fast on-SM scratchpad) | 128 KB per SM of unified data cache, of which at most 100 KB can be shared memory (compute capability 12.x); our chip reports capability 12.1 | [4]; local `nvidia-smi` reports `NVIDIA GB10, 12.1` |
| L2 cache (on-chip, shared by all SMs) | 24 MB | Hot Chips 2025 slides [3] |
| Main memory | 128 GB LPDDR5X, 256-bit, 273 GB/s | [1][2] |
| Raw interface rate | about 301 GB/s ("Up-to 9400 Mbs, ~301GB/s raw bandwidth") | [3] |

The 48-SM count is inferred from NVIDIA's 6,144 CUDA cores at 128 per SM plus a retailer listing; NVIDIA's pages I fetched do not print an SM count. Treat it as high confidence, not as a vendor quote.

## 1. How much data and compute a model needs

**Kaplan (OpenAI, 2020).** Loss falls as a smooth power law in model size, data and compute; the paper's conclusion is that "optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence" [7]. Its fitted exponents are about 0.076 for model size and 0.095 for data, and it estimates training compute as C ≈ 6 N × (batch × steps), i.e. about 6 FLOPs per parameter per training token [7]. This advice, big model and few tokens, turned out to be wrong in the large-model regime (next entry).

**Chinchilla (DeepMind, 2022).** Training 400+ models from 70 million to over 16 billion parameters on 5 to 500 billion tokens, the authors found "the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled" [8]. Their 70B-parameter Chinchilla, trained on 4x more data than the 280B Gopher with the same compute, beat Gopher, GPT-3 and others [8]. The paper's own table of compute-optimal runs shows 1 billion parameters paired with 20.2 billion tokens [8], which is the source of the "about 20 tokens per parameter" rule I use below. Kaplan and Chinchilla disagreed because Kaplan under-trained its models; Chinchilla's data is the better-supported of the two, and I weight it.

**Llama practice: train far past Chinchilla.** Chinchilla minimises cost to reach a loss; it ignores that a deployed model is run billions of times. Meta states: "the Chinchilla-optimal amount of training compute for an 8B parameter model corresponds to ~200B tokens ... model performance continues to improve even after the model is trained on two orders of magnitude more data" [9]. Llama 3 8B saw over 15T tokens [9], 75x the Chinchilla amount (15T / 200B = 75, calc:c8ee9718). LLaMA-1 started the practice: its 13B model "outperforms GPT-3 (175B) on most benchmarks" [10]. The Llama 3 paper says its flagship was near compute-optimal while "we also train our smaller models for much longer than is compute-optimal" [12].

**What it means for us.** A model we will run ourselves, many times, on one box, belongs in the over-trained regime: pick a small model and feed it far more than 20 tokens per parameter. The cost is more training FLOPs per unit of quality. The saving is a smaller, cheaper-to-run model.

**Repeating data (Muennighoff et al., 2023).** With a fixed compute budget, "training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero" [11]. Their fit puts the half-life of repeated tokens at about 15 repetitions (16 epochs): past that, returns fall "extremely fast" [11]. Tested up to 9B parameters and 900B tokens [11]. This matters on one machine because a small clean corpus can be reused up to about 4 times at essentially no loss, and up to roughly 16 times with diminishing value.

**Smallest model worth training on one machine.** The literature gives no single threshold; it gives this ladder, each rung measured: (a) a 1M to 35M-parameter model trained on a single V100 in at most 30 hours writes coherent stories if the data is simple enough (TinyStories, section 3) [17]; (b) a 124M-parameter GPT-2-class model reaches 3.28 FineWeb loss in under 40 seconds on 8 H100s with the speedrun recipe (section 2) [13]; (c) a 350M-class model reaches 2.92 in 17.35 minutes on 8 H100s [13]; (d) phi-1 shows 1.3B parameters is enough for 50.6% on a code benchmark with curated data (section 3) [18]. For a first real run on one GB10, 124M to 350M parameters is the range where published recipes already exist end to end; section 8 does the arithmetic.

## 2. Training speedruns: the engineered record

The modded-nanogpt repository fixes a goal and lets anyone compete: "the fastest algorithm to use 8 NVIDIA H100 GPUs to train a language model that attains 3.28 cross-entropy loss on the FineWeb validation set" [13]. The target loss comes from Andrej Karpathy's llm.c GPT-2 (124M) replication [13]. FineWeb is a public web text dataset (section 3). The rules forbid changing the data stream (batch size, sequence length and architecture may change), require 3.28 or better with statistical significance (p < 0.01), and ban extra compiler flags [13]. This makes it the cleanest public record of pure engineering gain against a fixed target.

**Baseline, and a disagreement between sources.** The repository's table lists the llm.c baseline as "45 minutes", and says that run needed 10B tokens [13]. Karpathy's own post says the 124M model takes "~90 minutes" on "one 8X A100 80GB SXM node", cost "about $20", 10 billion FineWeb tokens, ending near 3.29 loss [14]. The two are consistent only if the 45-minute figure is an H100 time, which the speedrun repository implies (its runs are timed on 8 H100s) but Karpathy's post gives only as a community comment. I use 45 minutes on 8 H100 for the speedrun ratios and 90 minutes on 8 A100 for the original.

**Headline.** The current record (#92, 2026-08-30) is 0.665 minutes (39.9 seconds, calc:b9eec80a) against 45 minutes: a 67.7x speedup (calc:2981eefe) on the same hardware class and the same loss target [13]. The README says the current algorithm needs "under 330M tokens" against 10B for llm.c [13], a 30.3x cut in data (calc:4494ba9c). So about 30x of the 67.7x is "learn more from each token" and the remaining 2.23x (calc:84566df3) is more throughput per token. In GPU-time, 0.665 minutes on 8 GPUs is 0.0887 H100-hours (calc:249723b3). One caveat: record #92 sits far below the trend (1.126 to 0.665 minutes in one step) and bundles many changes, including a hashed n-gram table of 84.6M rows [13]; I list it as the repository reports it and have not reproduced it.

**Phases** (times from the table below):
- Records 1 to 11 (May to Nov 2024): 45 to 7.2 minutes, a 6.25x gain (calc:2109ddb0). Mostly optimizer and architecture changes.
- Records 12 to 20 (Nov 2024 to Jan 2025): 7.2 to 2.992 minutes, 2.41x (calc:2f5f326b). Attention and embedding changes.
- Records 21 to 92 (Jan 2025 to Aug 2026): 2.933 to 0.665 minutes, 4.41x (calc:43dd99e7). Mostly kernels, precision and many small architecture changes.

**What the big steps were, in order of effect**
- *Muon optimizer* (#3, 31.4 to 24.9 min, 1.26x, calc:80839589). Muon "takes the updates from SGD-momentum and applies a Newton-Schulz iteration that approximately orthogonalizes each one before the update is applied" [15], i.e. it replaces each weight-matrix update with the nearest "evenly spread" matrix, using five rounds of small matrix multiplies. Jordan reports a 1.35x gain in speedrun training speed and "a FLOP overhead below 1%" [15]; he also reports a 1.5B-parameter run reaching GPT-2-XL-level HellaSwag in 10 8xH100-hours versus 13.3 with AdamW [15]. Muon is applied only to the 2D hidden-layer weight matrices; AdamW still handles embeddings, output layer and scalars [15].
- *Modern architecture bundle* (#5, 22.3 to 15.2 min, 1.47x, calc:707656dd): padded embeddings, ReLU² activation, zero-initialised projections, QK-norm (normalising the attention query and key vectors) [13].
- *FlexAttention with long context windows* (#12, 7.2 to 5.03 min, 1.43x, calc:b9e92c47): replaced 1024-token dense causal attention with 64K-token FlexAttention (PyTorch's programmable attention kernel) [13].
- *Value embeddings* (#14, #15, #17): extra learned per-token vectors mixed into attention values; U-net style skip connections (#11, #15): early layers feed late layers directly [13].
- *Precision*: bfloat16 activations (#10), FP8 output head (#19), later FP8 for MLP and QKV forward and backward (#84, #89, #90, #92) [13]. FP8 is an 8-bit float format; section 4.
- *Fused kernels*: Triton (a GPU kernel language) kernels for symmetric matmul (#27), fused linear + ReLU² (#59), fused softcapped cross-entropy (#60), more cross-entropy and transpose kernels (#69, #75, #76, #79) [13]. Kernel fusion is the memory-traffic technique of section 5.
- *Schedule and batch tricks*: attention-window warmup (#13), learning rate decay to 0.1 not 0 (#19), longer cooldown fraction 0.45 (#26), batch-size schedule (#46), reduced batch size (#21, #39) [13].
- *Optimizer refinements*: Polar Express replacing Newton-Schulz (#38), NorMuon (#41, #42), cautious weight decay (#43, #50), Adam-update-every-other-step (#39), unified optimizers (#61), ANVIL optimizer stack (#92) [13].
- *Systems*: gradient all-reduce overlap, reduce-scatter, async data loading, CUDA-graph capture of the whole training step (#22 to #24, #33, #92) [13]. Eight GPUs have to exchange gradients; one GB10 does not, so the gradient-communication records do not transfer to us.

**Full record table, track 1 (GPT-2 124M class, 3.28 FineWeb val loss, 8 H100).** Copied from the repository README [13]; a saved copy is in `evidence/efficiency/`. Record 21 appears three times because the repository re-timed it twice (rule change, newer PyTorch).

| # | Record time | What changed (as listed) | Date | By |
|---|---|---|---|---|
| 1 | 45 minutes | llm.c baseline | 05/28/24 | @karpathy, llm.c contributors |
| 2 | 31.4 minutes | Tuned learning rate & rotary embeddings | 06/06/24 | @kellerjordan0 |
| 3 | 24.9 minutes | Introduced the Muon optimizer | 10/04/24 | @kellerjordan0, @jxbz |
| 4 | 22.3 minutes | Muon improvements | 10/11/24 | @kellerjordan0, @bozavlado |
| 5 | 15.2 minutes | Pad embeddings, ReLU², zero-init projections, QK-norm | 10/14/24 | @Grad62304977, @kellerjordan0 |
| 6 | 13.1 minutes | Distributed the overhead of Muon | 10/18/24 | @kellerjordan0 |
| 7 | 12.0 minutes | Upgraded PyTorch 2.5.0 | 10/18/24 | @kellerjordan0 |
| 8 | 10.8 minutes | Untied embedding and head | 11/03/24 | @Grad62304977, @kellerjordan0 |
| 9 | 8.2 minutes | Value and embedding skip connections, momentum warmup, logit softcap | 11/06/24 | @Grad62304977, @kellerjordan0 |
| 10 | 7.8 minutes | Bfloat16 activations | 11/08/24 | @kellerjordan0 |
| 11 | 7.2 minutes | U-net pattern skip connections & double lr | 11/10/24 | @brendanh0gan |
| 12 | 5.03 minutes | 1024-ctx dense causal attention → 64K-ctx FlexAttention | 11/19/24 | @KoszarskyB |
| 13 | 4.66 minutes | Attention window warmup | 11/24/24 | @fernbear.bsky.social |
| 14 | 4.41 minutes | Value Embeddings | 12/04/24 | @KoszarskyB |
| 15 | 3.95 minutes | U-net pattern value embeddings, assorted code optimizations | 12/08/24 | @leloykun, @YouJiacheng |
| 16 | 3.80 minutes | Split value embeddings, block sliding window, separate block mask | 12/10/24 | @YouJiacheng |
| 17 | 3.57 minutes | Sparsify value embeddings, improve rotary embeddings, drop an attn layer | 12/17/24 | @YouJiacheng |
| 18 | 3.4 minutes | Lower logit softcap from 30 to 15 | 01/04/25 | @KoszarskyB |
| 19 | 3.142 minutes | FP8 head, offset logits, lr decay to 0.1 instead of 0.0 | 01/13/25 | @YouJiacheng |
| 20 | 2.992 minutes | Merged QKV weights, long-short attention, attention scale, lower Adam epsilon, batched Muon | 01/16/25 | @leloykun, @fernbear.bsky.social, @YouJiacheng, @brendanh0gan, @scottjmaddox, @Grad62304977 |
| 21 | 2.933 minutes | Reduced batch size | 01/26/25 | @leloykun |
| 21 | 2.997 minutes | 21st record with new timing | 02/01/25 | not a new record, just re-timing #21 with the updated rules |
| 21 | 3.014 minutes | 21st record with latest torch | 05/24/25 | not a new record, just re-timing #21 with latest torch |
| 22 | 2.990 minutes | Faster gradient all-reduce | 05/24/25 | @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad; The Enigma project |
| 23 | 2.979 minutes | Overlap computation and gradient communication | 05/25/25 | @ryanyang0 |
| 24 | 2.966 minutes | Replace gradient all_reduce with reduce_scatter | 05/30/25 | @vagrawal |
| 25 | 2.896 minutes | Upgrade PyTorch to 2.9.0.dev20250713+cu126 | 07/13/25 | @kellerjordan0 |
| 26 | 2.863 minutes | Align training batch starts with EoS, increase cooldown frac to .45 | 07/13/25 | @classiclarryd |
| 27 | 2.817 minutes | Transpose one of the MLP matrices + add Triton kernel for symmetric matmul | 07/18/25 | @byronxu99 |
| 28 | 2.812 minutes | Sparse attention gate | 08/23/25 | @classiclarryd |
| 29 | 2.731 minutes | Flash Attention 3, 2048 max_doc_len, update ws schedule | 09/03/25 | @varunneal |
| 30 | 2.717 minutes | Drop first MLP layer | 09/05/25 | @EmelyanenkoK |
| 31 | 2.656 minutes | Dynamically incorporate YaRN during training and validation | 09/10/25 | @classiclarryd |
| 32 | 2.625 minutes | Optimize distributed training, improve skip connection gating, and enhance bfloat16 usage | 09/11/25 | @bernard24 & AI system hiverge.ai |
| 33 | 2.565 minutes | Asynchronously fetch and index data batches, extend final layer attention window for validation | 09/15/25 | @classiclarryd |
| 34 | 2.547 minutes | Smear token embeddings 1 position forward | 09/18/25 | @classiclarryd |
| 35 | 2.527 minutes | Drop first attn layer, extend all long windows for validation, update schedule | 09/21/25 | @classiclarryd |
| 36 | 2.495 minutes | MuonCustomSizing, perform mlp and attn reduce scatter in shared call | 09/23/25 | @classiclarryd |
| 37 | 2.483 minutes | Compute cross entropy in BF16 during training | 09/27/25 | @Gusarich |
| 38 | 2.476 minutes | Polar Express, replacement for Newton-Schulz | 09/29/25 | @varunneal |
| 39 | 2.447 minutes | Only update Adam params every other step, reduce batch size | 09/30/25 | @classiclarryd |
| 40 | 2.358 minutes | Backout, misc hyperparameter tuning, optimize lambda padding | 10/04/25 | @classiclarryd |
| 41 | 2.345 minutes | NorMuon | 10/24/25 | @li_zichong |
| 42 | 2.313 minutes | Update NorMuon LR, Step Logic | 10/27/25 | @varunneal |
| 43 | 2.284 minutes | Cautious Weight Decay w/ schedule | 11/10/25 | @varunneal |
| 44 | 2.269 minutes | Backward hooks on Adam, Profiling 101 | 11/16/25 | @akash5474 |
| 45 | 2.248 minutes | Refine skip arch, update exponential decay init | 11/18/25 | @classiclarryd |
| 46 | 2.203 minutes | Batch size schedule | 11/29/25 | @varunneal |
| 47 | 2.193 minutes | Multiply attn lambda with weight instead of data, fix warmup | 12/10/25 | @roeeshenberg |
| 48 | 2.170 minutes | Speed up Muon, additional pre-multiply lambda, reshape matrices, update lr, update NorMuon axis | 12/11/25 | @ChrisJMcCormick |
| 49 | 2.146 minutes | Partial Key Offset | 12/14/25 | @classiclarryd |
| 50 | 2.128 minutes | Extend Cautious Weight Decay to Adam parameters | 12/18/25 | @roeeshenberg |
| 51 | 2.075 minutes | Retie Embed to lm_head, retune fp8 scales | 12/19/25 | @varunneal |
| 52 | 2.037 minutes | Smooth scalars via beta increase, decrease smear gate lr, freeze scalars during transitions, adam all reduce | 12/21/25 | @ChrisJMcCormick |
| 53 | 1.988 minutes | Multi-token prediction, untie embed/lm_head at 2/3 training, lr update, tweak CWD | 12/22/25 | @varunneal, feat. @classiclarryd |
| 54 | 1.940 minutes | Asymmetric Logit Rescale | 12/26/25 | @classiclarryd |
| 55 | 1.918 minutes | Gates on value embeds and skip connection | 12/29/25 | @classiclarryd |
| 56 | 1.894 minutes | Optimize and compile Adam, increase Adam buffer precision, move gates from Muon to Adam parameter banks | 12/31/25 | @ChrisJMcCormick |
| 57 | 1.878 minutes | Bfloat16 attn/mlp weights, mixed precision Muon, interweave Adam/Muon, finer-grain Adam beta | 01/04/26 | @classiclarryd, feat. @YouJiacheng, @ChrisJMcCormick |
| 58 | 1.820 minutes | Paired Head Attention | 01/07/26 | @classiclarryd |
| 59 | 1.781 minutes | Fused triton kernel for linear relu square MLP step | 01/10/26 | @andrewbriand8, @Joshrav21 |
| 60 | 1.765 minutes | Fused triton kernel for softcapped multi-token prediction cross entropy step | 01/16/26 | @soren_dunn_ & AI System Locus |
| 61 | 1.748 minutes | Unified Optimizers and Transposed LM Head | 01/18/26 | @ChrisJMcCormick |
| 62 | 1.655 minutes | Bigram Hash Embedding | 01/19/26 | @classiclarryd |
| 63 | 1.650 minutes | Untie Value Embeds | 01/26/26 | @photon_mz |
| 64 | 1.630 minutes | Tuned nonzero Attn V and O init | 01/30/26 | @srashedll |
| 65 | 1.613 minutes | Group Value Embeds into single parameter | 01/30/26 | @varunneal |
| 66 | 1.595 minutes | Torch 2.10 | 01/31/26 | - |
| 67 | 1.540 minutes | Tune fused softcap kernels and fuse fp8 quantization in LM head | 01/31/26 | @andrewbriand8 |
| 68 | 1.535 minutes | Move bigram hash to GPU | 01/31/26 | @dhrvji |
| 69 | 1.528 minutes | Kernel Optimizations | 02/02/26 | @EmmettBicker & AI System Aster |
| 70 | 1.521 minutes | Tune value embed layout and ve_gates | 02/03/26 | @photon_mz |
| 71 | 1.516 minutes | Sparse bigram gradient comms and optimized loading on CPU | 02/06/26 | @roeeshenberg |
| 72 | 1.496 minutes | Increase minimum lr and add max_seq_len schedule | 02/10/26 | @dualverse-ai & AI System Station |
| 73 | 1.485 minutes | Partitioned Hyperconnections | 02/12/26 | @sisovicm |
| 74 | 1.468 minutes | Flattened GPT forward, removed post attention lambdas, added transpose kernels | 02/16/26 | @ChrisJMcCormick |
| 75 | 1.453 minutes | Cross Entropy Kernel Optimizations | 02/23/26 | @moof2x |
| 76 | 1.446 minutes | Reuse and tune backward transpose kernel | 02/28/26 | @samacqua |
| 77 | 1.435 minutes | Replace partitioned hyperconnections with single saved activation | 03/06/26 | @classiclarryd |
| 78 | 1.426 minutes | Tighten bounds on fa3 max_num_docs to match fineweb distribution | 03/22/26 | @ChrisJMcCormick |
| 79 | 1.411 minutes | Fuse Cross Entropy Fwd/Bwk Kernel, to avoid recalc on softcap sigmoid | 04/04/26 | @andrewbriand8 |
| 80 | 1.406 minutes | In Muon orthogonize Q and K matrices in pairs of heads, instead of across the full 6 head matrix | 04/08/26 | @samacqua |
| 81 | 1.363 minutes | MUDD Skip Connections | 04/22/26 | @Lisennlp |
| 82 | 1.353 minutes | Learnable XSA | 04/29/26 | @_djdumpling |
| 83 | 1.328 minutes | Sign Trick on Bigram Embed | 05/20/26 | @TrianX |
| 84 | 1.320 minutes | FP8 on MLP up-projection forward pass | 05/21/26 | @sisovicm |
| 85 | 1.271 minutes | MUDD gates and Lightweight Dynamically Composable MHA | 05/27/26 | @Lisennlp |
| 86 | 1.266 minutes | Algebraic rewrite of XSA, same math faster execution | 05/27/26 | @aryavohra |
| 87 | 1.256 minutes | Faster Implementation of Relu^2 Kernel | 06/11/26 | @cong_ml and AI System Recursive |
| 88 | 1.243 minutes | Prefix token prediction auxiliary loss | 07/13/26 | @jvarho |
| 89 | 1.23 minutes | MLP down projection in FP8 with efficient delayed scaling metric | 07/17/26 | @Mister-dev-oss, @CerovazS, @MarioPaerle, @GabrieleCirillo, @crisostomi |
| 90 | 1.13 minutes | 128 -> 96 dim QK, Fuse QK Norm, RoPE, and KeyOffset into Triton, Move MLP bwk to FP8, Move QKV fwd and bwk to FP8. | 08/03/26 | @theonlyglitch_ |
| 91 | 1.126 minutes | Mask logits for infeasible token continuations during validation. | 08/06/26 | @jvarho |
| 92 | 0.665 minutes | Sampled softmax over the early stages, full FP8 MLP fwd+bwd, 84.6M-row hashed n-gram table, ANVIL optimizer stack, depth reduction + mixed-width QK attention, full CUDA-graph capture of the training step | 08/30/26 | @DevenPzak |

**Track 2: GPT-2 Medium class (350M parameters), target loss 2.92, 8 H100.** The first record is the llm.c baseline at 5.8 hours, then 29.3 minutes for the speedrun recipe scaled up, down to 17.35 minutes at record 18 (2025-12-31): a 20.06x gain (calc:3f309a02) [13]. Table [13]:

| # | Record time | What changed (as listed) | Date | By |
|---|---|---|---|---|
| 1 | 5.8 hours | llm.c baseline (350M parameters) | 05/28/24 | @karpathy, llm.c contributors |
| 2 | 29.3 minutes | Initial record based on scaling up the GPT-2 small track speedrun | 01/18/25 | @kellerjordan0 |
| 3 | 28.1 minutes | Added standard weight decay | 02/08/25 | @kellerjordan0 |
| 4 | 27.7 minutes | Tuned Muon Newton-Schulz coefficients | 02/14/25 | @leloykun |
| 5 | 27.2 minutes | Increased learning rate cooldown phase duration | 03/06/25 | @YouJiacheng |
| 6 | 25.95 minutes* | 2x MLP wd, qkv norm, all_reduce/opt.step() overlap, optimized skip pattern | 03/25/25 | @YouJiacheng |
| 7 | 25.29 minutes | Remove FP8 head; ISRU logits softcap; New sharded mixed precision Muon; merge weights | 04/16/25 | @YouJiacheng |
| 8 | 24.50 minutes | Cubic sliding window size schedule, 2× max window size (24.84 minutes) 24.5min repro | 04/22/25 | @jadenj3o |
| 9 | 24.12 minutes | Add two value embeddings | 08/28/25 | @snimu |
| 10 | 24.07 minutes | Second input embedding | 09/11/25 | @snimu |
| 11 | 23.45 minutes | Upgrade from torch 2.7 to torch==2.10.0.dev20251210+cu126 | - | - |
| 12 | 23.28 minutes | Snoo Optimizer (Outer optimizer around Adam and Muon) | 09/16/25 | @dominikkallusky |
| 13 | 23.14 minutes | EMA Wrapper on Muon | 09/17/25 | @acutkosky |
| 14 | 23.08 minutes | Combine both records 12 & 13 | 09/30/25 | @acutkosky |
| 15 | 23.03 minutes | Backout (Skip from 2/3 point to pre-lm_head) | 10/04/25 | @snimu |
| 16 | 22.99 minutes | Smear-MTP | 11/02/25 | @snimu |
| 17 | 22.98 minutes | Remove Redundant Mask Op | 11/12/25 | @manikbhandari |
| 18 | 17.35 minutes | Bulk transfer short track features | 12/31/25 | - |

**Fewer GPUs.** I found no official speedrun table for one GPU. The rules time on 8 H100s only [13], and the forks I found keep that target (one fork's README: "4.7 minutes on 8xH100 instead of 45") [19]. The code can be pointed at fewer GPUs, but I have not seen a published single-GPU time to cite. Single-GPU records that do exist are in section 8.

**Cramming (Geiping and Goldstein, Dec 2022): BERT in one day on one GPU.** They asked "How far can we get with a single GPU in just one day?" [16]. BERT is an older text-understanding model trained by filling in blanked-out words. Their "crammed BERT" reached a GLUE score (a standard set of 9 language-understanding tasks) of 78.3 after one day on a single RTX 2080 Ti (a 2018 consumer card) and 78.1 on an RTX A4000, against 80.9 for the fully trained BERT-base [16]. In total compute that is 5 exaFLOP (2080 Ti) to 13 exaFLOP (RTX A6000), against 13,824 exaFLOP for RoBERTa-base and 7,299,072 exaFLOP for PaLM in the paper's own comparison table [16]. Their findings, which still hold for GPT-style training: performance "closely follows scaling laws observed in large-compute settings" [16]; with a single GPU the useful micro-batch is smaller than the best batch, so they accumulate gradients and use a batch-size ramp; they disable linear-layer biases, use a one-cycle learning rate schedule with gradient clipping at 0.5, and filter and sort the data [16]. Scaling down was hard: the paper sorts recent improvements by whether they help at this budget, and reports some of them do not [16]. The speedrun above is the closest successor for decoder (GPT-style) models.

## 3. Small models on good data

Data quality is the cheapest lever in this chapter, but each claim needs its price.

| Work | What it showed | Compute and hardware | Source |
|---|---|---|---|
| TinyStories (Microsoft, 2023) | A synthetic set of short stories using words a 3 to 4 year old knows, written by GPT-3.5 and GPT-4. Models "below 10 million total parameters" or with a single transformer block "still produce fluent and consistent stories" | Models of 1M to 35M parameters, 1 to 8 layers, each "trained on a single V100 GPU within at most 30 hours" | [17] |
| phi-1 (Microsoft, 2023) | 1.3B-parameter code model, 6B tokens of filtered web code plus 1B tokens of GPT-3.5-written textbooks; 50.6% HumanEval pass@1 and 55.5% MBPP; a 350M version still reached 45% HumanEval | 4 days on 8 A100 = 768 A100-hours (4 x 24 x 8, calc:f59f19be) | [18] |
| phi-2 (Microsoft, Dec 2023) | 2.7B parameters; 1.4T tokens (multiple passes over a mix of synthetic and filtered web data); "matches or outperforms models up to 25x larger" on the vendor's chosen benchmarks | 14 days on 96 A100 = 32,256 A100-hours (14 x 24 x 96, calc:a8844880) | [20] |
| BabyLM challenge (2023, findings published 2025) | Language acquired by a child needs under 100M words; LLMs use 3 to 4 orders of magnitude more. From 30+ submissions: the winners used the LTG-BERT architecture and "outperformed models trained on trillions of words"; shorter input sequences and distilling from a teacher helped; curriculum learning (ordering data easy to hard) "was largely unsuccessful" | Fixed data budget (about 100M words) rather than compute | [21] |
| FineWeb and FineWeb-Edu (Hugging Face, 2024) | A 15T-token cleaned web set, and a 1.3T-token subset kept by a classifier trained on LLM ratings of educational value. FineWeb-Edu "can match the final performance of Matrix with almost 10x fewer tokens" on MMLU; MMLU rose from 33% to 37% and ARC from 46% to 57% against other open web datasets in their ablations | The filter itself cost compute: scoring the 15T tokens "required 6,000 H100 GPU hours" | [22] |
| DataComp-LM (2024) | Model-based filtering "is key". DCLM-Baseline trains a 7B model from scratch to 64% 5-shot MMLU with 2.6T tokens, 6.6 points above MAP-Neo with 40% less compute, and is comparable to Llama 3 8B on 53 tasks "while being trained with 6.6x less compute" | 7B model, 2.6T tokens; compute ratios as the paper states them | [23] |
| Dolma / OLMo / OLMo 2 (AI2, 2024 to 2025) | Open corpus (3T tokens) with documented cleaning and mixing; OLMo 2 adds a late "Dolmino" high-quality mix during the final annealing phase and reports sitting "at the Pareto frontier of performance to training compute" | 7B, 13B, 32B models; per-model FLOPs in the paper | [24][25] |

**What the numbers say about compute.** The best-measured savings are: about 10x fewer tokens for equal MMLU (FineWeb-Edu versus a prior web set, in the authors' ablation) [22]; 6.6x less training compute for a similar 53-task average at 7B (DCLM versus Llama 3 8B) [23]. Both are single-paper, single-benchmark-suite results by the data authors, so read them as "a factor of several, not 100". Three costs are not in those ratios. Filtering classifiers cost real compute (6,000 H100-hours for FineWeb-Edu's scoring pass) [22]. Synthetic data (TinyStories, phi) was written by a far larger model (GPT-3.5 or GPT-4), whose training cost is not counted, so it is closer to distillation (section 6) than to free quality. And the phi family was selected and evaluated on benchmarks close to its training style; the evidence for "textbooks beat scale" outside those benchmarks is weaker than the headlines.

**What it means for us.** Under 100M parameters, the best-supported move is simple, tightly filtered data (TinyStories shows this at 1M to 35M parameters on a single V100 [17]). Above that, a model-scored quality filter on a web corpus is worth a factor of several in tokens [22][23], and the filter should be built once and its own cost counted. We do not need 15T tokens: repeating a clean 10B-token set 4 times costs almost nothing in loss [11].

## 4. Numerics: fewer bits per number

Fewer bits means fewer bytes to move and, on tensor cores, more multiplies per second. The question for each format is what it costs in quality, and what the chip actually runs faster.

**FP8 (8-bit floats).** The standard FP8 definition has two encodings: E4M3 (4 exponent bits, 3 mantissa bits, more precision) and E5M2 (5 and 2, more range), proposed jointly by NVIDIA, Arm and Intel; they matched 16-bit training quality up to 175B-parameter language models with unchanged hyperparameters [28]. Microsoft's FP8-LM went further, keeping gradients and optimizer states in 8 bits as well, and on H100s training GPT-175B it cut real memory by 39% and ran 75% faster than a BF16 Megatron-LM baseline [29]. DeepSeek-V3 (671B-parameter mixture of experts, 14.8T tokens, 2,048 H800 GPUs) is the largest public FP8 run: it scales each 1x128 tile of activations and each 128x128 block of weights separately (so one outlier cannot ruin a whole tensor), does the matrix-multiply accumulation at higher precision by promoting partial sums to the GPU's regular cores, and compared with a BF16 baseline over about 1 trillion tokens "the relative loss error of our FP8-training model remains consistently below 0.25%" [26]. It keeps the embedding, output head, mixture gating, normalisation and attention operators in BF16 or FP32, and keeps master weights, weight gradients and optimizer states in higher precision, with Adam moments in BF16 [26]. Full training was 2.788M H800 GPU-hours, of which 2.664M for pretraining (about 54 days on 2,048 GPUs, calc:5da1440a), at a stated $2 per GPU-hour or $5.576M, a figure the report says excludes prior research and ablations [26]. Lesson: FP8 is proven for the big matrix multiplies, but the recipe is not "cast everything"; about a dozen design choices keep it stable.

**FP4 (4-bit floats), 2025.** NVIDIA's NVFP4 pretraining paper trained "a 12-billion-parameter model on 10 trillion tokens" in 4-bit, "the longest publicly documented training run in 4-bit precision to date" [27]. The recipe has four parts: random Hadamard transforms (a fixed scrambling of the numbers that spreads outliers out), two-dimensional scaling so the forward and backward passes see the same quantised weights, stochastic rounding (round up or down at random in proportion to the remainder, so errors average out) for gradients, and keeping sensitive layers in higher precision [27]. NVFP4 uses a scale per block of 16 values plus a tensor-level scale [27]. Result: validation loss tracks the FP8 run with relative error "consistently below 1%" in the stable phase and slightly above 1.5% as the learning rate decays; downstream accuracies "remain largely unaffected" [27]. In that run the first 2 and last 8 blocks (16% of linear layers) stayed in BF16 [27]. Limits: one model family at 12B and 10T tokens, by the vendor that sells the hardware; I know of no equivalent result under 1B parameters [unverified, absence of evidence].

**Ternary weights: BitNet b1.58.** Every weight is -1, 0 or +1. The authors report it "matches the full-precision (i.e., FP16 or BF16) Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance" [30], starting at about 3B parameters, where it is "2.71 times faster and using 3.55 times less GPU memory" at inference [30]. The gain is in running the trained model (adds instead of multiplies, 1.58 bits per weight in memory); the paper reports latency, memory and energy of the trained model and gives no training-compute figure. The benefit for training on a GB10 is therefore small today, and no ternary matrix-multiply hardware exists on it.

**What the GB10 runs fastest.** NVIDIA states only "1 PFLOP at FP4 with sparsity" [2]. A reviewer measured the best dense matrix-multiply speed as 99.8 TFLOPS in BF16 and 207.7 TFLOPS in FP8 (E4M3), a ratio of 2.08 (calc:5d6e8d19) [6]. A user on the NVIDIA developer forum reports a CUTLASS test kernel at 356 TFLOPS NVFP4 and 188 TFLOPS FP8, a ratio of 1.89 (calc:80bc14ab), about 71% of a 500 TFLOPS dense FP4 peak [31]; that is one user's forum post, not a vendor or peer-reviewed number [31]. The same thread shows real inference software only reached 65 tokens per second in NVFP4 against 55 for FP8 on one model, because the software was bandwidth-limited or fell back to slower paths [31]. Read together: FP8 is the format to design for first (2.1x over BF16 on a clean measurement, a mature recipe), and FP4 is a later 1.9x on paper if our own kernels can feed it and the quality recipe [27] holds at small scale.

**Design consequence.** Hold weights and activations in E4M3 for forward matrix multiplies, gradients in E5M2 or E4M3 with per-block scales, accumulate in FP32 inside the tensor core, and keep a higher-precision master copy plus optimizer state, as DeepSeek did [26]. The master copy is where most of the bytes are (section 5).

## 5. Memory traffic: Drake's question

Drake's question: can training stay on the math cores, without moving data from memory into them? Below is the physics, then what has been published, then the arithmetic for one GB10.

### 5.1 What a training step moves

One training step on one batch of T tokens for a model with N parameters does about 6 N T floating-point operations (about 6 per parameter per token, forward plus backward) [7]. Around that math it moves four kinds of data between main memory and the chip:

1. **Weights**: read once in the forward pass and again in the backward pass.
2. **Activations**: the intermediate results of every layer, written in the forward pass and read back in the backward pass (unless thrown away and recomputed).
3. **Gradients and optimizer state**: gradients written once; the optimizer then reads and writes the master weights and its momentum buffers once per step. A common mixed-precision Adam setup costs 2 + 2 + 12 = 16 bytes per parameter just to store (weights, gradients, optimizer state) [37].
4. **The vocabulary projection**: the final layer turns each token into a score for every word-piece (50,257 of them for GPT-2), which is a large temporary table.

At 273 GB/s [1][2] any of these sets a floor on step time that does not depend on how fast the math is. The chip needs this much memory traffic per FLOP to be kept busy: 207.7 TFLOPS / 273 GB/s = 761 FLOP per byte in FP8 (calc:049b452d), or 366 FLOP per byte in BF16 at the measured 99.8 TFLOPS (calc:a6dbe17a) [6]. If a step does fewer FLOPs per byte moved than that, memory is the bottleneck, not the math.

### 5.2 The numbers for a 124M-parameter model on one GB10

My derivation, from published formulas, not a measurement. Model: GPT-2 small, 124M parameters, 12 layers, hidden width 768 [14][44]. Batch: 524,288 tokens per optimizer step (the llm.c 124M run requests "-d 524288", a total batch of about 0.5M tokens per update) [14].

- **Math time.** 6 x 124e6 x 524,288 = 3.9e14 FLOPs. At the measured best-case 207.7 TFLOPS, that is 1.878 s (calc:8ab5e8a9). Real runs reach a fraction of that (section 8).
- **Weights and optimizer state, once per step.** Assume DeepSeek-style storage [26]: FP32 master weights and gradients, BF16 Adam moments, plus FP8 copies, about 23 bytes per parameter moved per step. That is 23 x 124e6 = 2.85 GB, or 10.45 ms at 273 GB/s (calc:1929762d). That is 0.556% of the math time (calc:b9394334). **Weight and optimizer traffic is negligible at this batch size** because it is paid once per step and divided across 524,288 tokens. This is the main reason big-batch training works at all.
- **Activations.** Per layer, the count for a standard transformer layer stored at 16 bits is s*b*h*(34 + 5*a*s/h) bytes (s tokens in a sequence, b sequences, h hidden width, a attention heads) [39]. The second term is the attention score matrix, which FlashAttention removes by never writing it to memory [32]. Dropping it leaves 34 bytes per token per unit of hidden width per layer: 34 x 768 x 12 layers = 313,344 bytes per token (calc:a8be76e3). Written once and read once is 626,688 bytes per token. Over 524,288 tokens that is 329 GB of traffic, or 1.204 s at 273 GB/s (calc:097d18cc). **Activation traffic is the dominant memory cost**: it is 64% of the math time at the measured FP8 peak (1.204 / 1.878). Compute per byte moved is 6 x 124e6 / 626,688 = 1,187 FLOP/B (calc:34ce7f7e), above the FP8 balance point of 761, so a well-overlapped step stays compute-bound, but with only a 1.56x margin before memory time exceeds math time.
- **The vocabulary table.** Logits at 16 bits are 50,257 x 2 = 100,514 bytes per token; written and read once that is 32% of the whole activation traffic above (calc:251e3a5f). This is why the speedrun records fuse the loss calculation into one kernel (records 60, 75, 79 in section 2) [13].
- **Bigger models are better off.** FLOPs per token grow with N, activation bytes only with width x depth. For a 1.3B model (24 layers, width 2048 [44]) the same ratio is 2,334 FLOP/B (calc:9f1a520d), twice as favourable. **Small models are the hard case for memory traffic**, which is awkward since small is where a single GB10 starts.

### 5.3 Could the model live entirely on the math cores?

What the chip stores on-die, from the specs in section 0:
- Registers: 48 SMs x 64 K registers x 4 bytes = 12 MiB [4].
- Shared memory / L1: 48 x 128 KiB = 6 MiB (calc:143886d2), of which at most 100 KB per SM, 4.7 MiB, can be software-managed shared memory [4].
- L2 cache: 24 MB [3], taken as 24 MiB (the slide does not say which unit; a 5% error either way).
- **Total: 42 MiB (calc:582423cc).**

How big a trainable model fits in 42 MiB with nothing else?
- Standard mixed-precision Adam, 16 bytes per parameter [37]: 42 MiB / 16 = 2,752,512 parameters (calc:798c1823), about 2.8 million.
- A lean recipe (BF16 master weights, BF16 gradients, one BF16 momentum buffer as Muon needs [15]; 6 bytes) with one more byte for an FP8 weight copy, 7 bytes per parameter: 6,291,456 parameters (calc:56b4217e), about 6.3 million.

Both numbers assume every register, every byte of shared memory and all of L2 is full of model state, leaving nothing for the numbers being multiplied, the activations, or the program. Registers cannot be addressed as a shared pool across SMs, so the real figure is lower. For comparison, the 124M-parameter model's weights alone in FP8 (124 MB) are 2.8 times the entire on-chip storage (calc:dfc75d0e). **Answer: a GB10 can keep a model of a few million parameters on its math cores. Every model in the range this chapter cares about (50M and up) is 10 to 50 times too big.** The TinyStories-scale models of section 3 (1M to 35M parameters) [17] are the only published models near that size.

### 5.4 Silicon built around Drake's idea

- **Cerebras WSE-3** keeps 44 GB of SRAM on one wafer with 900,000 cores, 125 petaflops of peak AI compute and 21 PB/s of memory bandwidth, per a 2025 academic comparison paper that repeats the vendor's figures [41]. That is 999 times a GB10's 42 MiB (calc:54e42556). Even so, 44 GB at 16 bytes per parameter holds a 2.75-billion-parameter model's training state (calc:72b1b9bd), and the paper describes that for larger models the parameters "can live in external memory (MemoryX or similar) while compute elements on the wafer handle the processing" [41]. In other words, Cerebras's answer for big models is not weights-stationary; it is to stream the weights through the chip. Cerebras-GPT (111M to 13B parameters, trained to Chinchilla ratios on its wafer-scale cluster) is the published training result [43]. It shows that the architecture trains real models; it does not isolate how much of the gain is the SRAM.
- **Groq** puts 220 MiB of SRAM on each chip [42] and, as far as I found, has published inference results only; I found no training results on Groq hardware [unverified, absence of evidence]. Holding a 70B-parameter model in 8-bit weights takes at least 70e9 / (220 MiB) = 303.4 chips (calc:1f54fbe5) before any working memory, which shows the cost of keeping weights in SRAM: a chip per 220 MB.

**Implication.** The two machines that really keep weights on-die do so with 220 MiB (Groq, inference only) to 44 GB (Cerebras) of SRAM per unit. A GB10 has 42 MiB. For us "everything on the math cores" is out of reach for models above a few million parameters; the achievable goal is "stream each tensor through the cores once, touching main memory as little as possible".

### 5.5 What recovers the traffic when the model does not fit

Ranked by what dominates in 5.2, with the best-measured published gains:

1. **Do not store the attention score matrix: FlashAttention.** It tiles attention so intermediate results stay in on-chip SRAM, "to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM" [32]. Measured on A100: 15% end-to-end speedup on BERT-large against the MLPerf 1.1 speed record, 3x on GPT-2 at sequence length 1K [32]. FlashAttention-2 reaches 225 TFLOPs/s per A100 (72% model FLOPs utilization) end to end [33]; FlashAttention-3 reaches 740 TFLOPs/s on H100 in FP16 (75% utilization) and close to 1.2 PFLOPs/s in FP8 [34]. We would write ours in Omega for the GB10. FlashAttention-3 is built around Hopper-specific hardware (asynchronous Tensor Cores and the TMA copy engine) [34]; which of those the GB10 has is a question for the Omega and L279 work, not settled here.
2. **Recompute instead of store.** Throwing activations away and recomputing them in the backward pass trades FLOPs for bytes. Chen et al.: memory for a 1,000-layer network falls from 48G to 7G "with only 30 percent additional running time cost" [38]. Selective recomputation (recompute only the cheap, bulky parts) together with sequence parallelism cuts activation memory 5x and removes over 90% of the recompute overhead; on a 530B-parameter model on 2,240 A100s it reached 54.2% model FLOPs utilization, 29% faster than the 42.1% with full recomputation [39]. In our setting, storing only each layer's input (2 bytes per width unit per layer instead of 34) cuts activation traffic to 1/17 (calc:25920375) (2/34 = 0.0588, calc:25920375) at the cost of one extra forward pass, 8NT instead of 6NT FLOPs, a 33% math increase. For the 124M example that turns 1.204 s of traffic into about 0.07 s (1.204 x 1/17) and 1.878 s of math into about 2.50 s (1.878 x 8/6); the step becomes clearly compute-bound.
3. **Fuse adjacent operations so intermediate results never leave the chip.** Liger Kernel (Triton kernels that fuse operations and chunk the loss) reports on average "a 20% increase in training throughput and a 60% reduction in GPU memory usage" against Hugging Face's implementations [40]. The NanoGPT speedrun's fused kernels (linear + ReLU², softcapped cross-entropy, QK-norm + RoPE) are the same idea measured on the 124M model [13] (section 2).
4. **Persistent kernels / megakernels** remove the gaps between kernel launches and keep data on-chip across operations. Hazy Research ran Llama-1B's whole forward pass in one kernel on an H100 at 78% of memory bandwidth against at most 50% for vLLM and SGLang, "almost 2.5x faster than vLLM and over 1.5x faster than SGLang" at batch size 1 [35]. MPK compiles a model into one persistent kernel automatically, reporting up to 1.7x lower end-to-end latency [36]. **Both are inference results; I found no published training megakernel.** Our L272 stream is building the decode-side equivalent on the GB10; the training-side version is open ground.
5. **ZeRO and optimizer-state sharding do not help one chip.** ZeRO cuts per-device model state from 16Ψ bytes to about 16Ψ / N_d by splitting it over N_d devices [37]; with N_d = 1 the saving is zero. What does help on one chip is the same trick used the other way: smaller state per parameter (BF16 moments [26], one momentum buffer with Muon [15]).

**Which one recovers most?** For the 124M to 1B models we can train on one GB10, the numbers in 5.2 say activations are 64% of math time in traffic and weights and optimizer under 1%. So the most traffic is recovered by (a) recomputation of the bulky layers plus (b) fused kernels that also remove the logits table (32% of that traffic), with FlashAttention covering attention. Each has measured gains on other hardware (3x on GPT-2 attention on A100, 29% end-to-end from selective recompute at 530B scale, 20% throughput and 60% memory from fusion) [32][39][40], but I found no measurement of the combination on a GB10-class chip; L279 is to supply that.

## 6. Spending parameters differently

Four ideas change what one unit of compute buys: use only some of the parameters per token (mixture of experts), copy skill from a bigger model (distillation), start small and grow (model growth), and let a small model think longer when it answers (test-time compute).

### 6.1 Mixture of experts (MoE)

An MoE layer holds many parallel feed-forward blocks ("experts") and sends each token through only one or a few of them, so total parameters and per-token FLOPs come apart.
- **Switch Transformer (Google, 2021).** Based on T5-Base and T5-Large, "up to 7x increases in pre-training speed with the same computational resources" [45], and it showed sparse models can train in bfloat16 [45]. A trillion-parameter version got a 4x speedup over T5-XXL [45].
- **DeepSeekMoE (2024).** A 16B-parameter model "comparable performance with LLaMA2 7B, with only about 40% of computations" [46]; at 145B, comparable to DeepSeek 67B using 28.5% of the computation (their own preliminary result) [46].
- **DeepSeek-V3.** 671B total parameters, 37B active per token [26], i.e. 5.5% of the parameters touched per token (37 / 671 = 0.0551, calc:9ad67bd4).

These are FLOP savings measured against a dense model of equal quality. They cost more memory: every expert's weights and optimizer state still exist and must be stored. On one GB10 with 128 GB, standard Adam state at 16 bytes per parameter [37] fills the memory at about 8 billion parameters (128e9 / 16), before activations. MoE is a way to buy quality per FLOP, not per byte. Because model-state traffic is under 1% of a big-batch step at 124M (section 5.2), spreading it across eight times as many parameters would still leave it a few percent; the limit would be capacity, not bandwidth. A small MoE is a real option for us only after the dense baseline works.

### 6.2 Distillation

Train a small student to match a larger teacher's output probabilities, not just the correct next word.
- **MiniLM (2020).** The student kept "more than 99% accuracy on SQuAD 2.0 and several GLUE benchmark tasks using 50% of the Transformer parameters and computations of the teacher" [47].
- **Gemma 2 (Google, 2024).** The 2B and 9B models were trained "with knowledge distillation instead of next token prediction" and are "competitive alternatives to models that are 2-3 times bigger" [48]. The paper's small-scale test is the most relevant to us: with a 7B teacher and 500B training tokens, validation perplexity (lower is better) fell from 23 to 21 at 200M parameters, 19 to 17 at 400M and 17 to 15 at 1B [48]. That is about 10% lower perplexity at the sizes we would train.
- **DeepSeek-R1 distilled models (2025).** A 7B student fine-tuned on reasoning traces from DeepSeek-R1 scores 55.5% on the AIME 2024 maths contest, "surpassing QwQ-32B-Preview", and the 32B student scores 72.6% [49]. On the same Qwen2.5-32B base, applying reinforcement learning directly gave 47.0% on AIME 2024 against 72.6% from distillation [49]. The authors conclude "direct distillation from DeepSeek-R1 outperforms applying RL on it" [49].

The cost nobody headlines: the teacher's forward pass. Distilling a 124M student from a 27B teacher on every token costs the teacher's 2 x 27e9 = 54 GFLOP per token on top of the student's own 6 x 124e6 = 0.74 GFLOP, a total of 73.6x the student-only cost (calc:d23ee67d; my arithmetic). Online distillation from a large teacher is therefore far more expensive than the student's own training. It pays only if teacher outputs are computed once and reused (stored top-k probabilities), or the teacher is also small, as Gemma's 7B-to-2B test [48]. Gemma and R1 do not report the teacher cost in these comparisons.

### 6.3 Model growth: train small, then grow

- **Stacking Your Transformers (2024).** A depthwise stacking operator (train a shallow model, then copy its layers to make a deeper one and keep training): "compared to a conventionally trained 7B model using 300B tokens, our G_stack model converges to the same loss with 194B tokens, resulting in a 54.6% speedup" [50]. Tested up to 7B parameters after growth and 750B tokens [50]. Earlier progressive stacking for BERT (Gong et al., ICML 2019) is the origin of the idea; it appears in the reference list of [50].
- **Sparse upcycling (Google, 2022).** Turn a trained dense model into an MoE by copying its feed-forward blocks into experts: upcycled T5 models "significantly outperform their dense counterparts ... using only ~50% of the initial dense pretraining sunk cost" and beat MoEs trained from scratch on the same budget [51].
- **LLaMA Pro (2024).** Add new transformer blocks to a trained Llama2-7B (to 8.3B) and tune only the new blocks on code and maths, "without catastrophic forgetting" [52]. This is about adding skills cheaply, not about the first training run.

Fit for us: stacking is the one with a clean measured saving at pretraining (35% fewer tokens to the same loss; 300 to 194 = 0.647 of the tokens, calc:b4855a42). It makes a staged plan natural: train a shallow 124M-class model quickly, grow it, continue. The 7B result is one lab on one setup; the effect at 100M scale is not shown in that paper [unverified at small scale].

### 6.4 Test-time compute instead of parameters

- **Snell et al. (2024).** A smaller model given extra inference-time compute, spent adaptively per question, can "outperform a 14x larger model" in a FLOPs-matched comparison, but only "on problems where a smaller base model attains somewhat non-trivial success rates" [54]. Allocating the compute adaptively improved efficiency by more than 4x over best-of-N [54].
- **s1 (2025).** 1,000 curated reasoning examples plus "budget forcing" (make the model keep thinking by appending "Wait") took Qwen2.5-32B-Instruct to beat o1-preview on competition maths by up to 27%, and extra thinking moved AIME24 from 50% to 57% [53].
- o1 and R1 are the production versions of the same trade [49].

Fit for us: test-time compute multiplies decode cost, and decode is the memory-bound side of the GB10 (273 GB/s). It does not reduce the cost of the first model. It is a lever for after a base model exists.

## 7. Optimizers and schedules

An optimizer decides how to turn gradients into weight updates. AdamW is the long-time default. The claims below are speedups over a tuned AdamW in steps or compute to reach the same loss.

| Method | What it does, in plain words | Claimed gain | Where measured | Source |
|---|---|---|---|---|
| Muon | Replaces each weight-matrix update with the nearest "evenly spread" matrix using a few small matrix multiplies; one momentum buffer; applied to hidden 2D weights only | 1.35x on the NanoGPT speedrun; 10 vs 13.3 8xH100-hours to a GPT-2-XL-level HellaSwag score at 1.5B parameters; FLOP overhead below 1% | 124M speedrun and 1.5B run, 8 H100 | [15] |
| Muon at scale (Moonshot) | Adds weight decay and a per-matrix update scale so Muon runs without retuning | "~2x computational efficiency compared to AdamW with compute optimal training"; Moonlight, a 3B/16B-parameter MoE trained on 5.7T tokens | scaling-law runs plus one large run, hardware not stated in the abstract | [60] |
| Muon, practical view | Muon keeps data efficiency at very large batch sizes and moves the compute-time Pareto frontier over AdamW | Shown at sizes up to 4B parameters; no single headline factor in the abstract | up to 4B parameters | [61] |
| Shampoo | Keeps a small preconditioning matrix per tensor dimension (a smarter, direction-aware step size) | "converging considerably faster than commonly used optimizers"; per-step cost comparable to SGD in the authors' words | 2018 vision and language models | [55] |
| SOAP | Runs Adam in Shampoo's rotated coordinate system | In the large-batch regime, over 40% fewer iterations and over 35% less wall-clock than AdamW, about 20% better than Shampoo | 360M and 660M-parameter language models | [56] |
| Sophia | Divides the gradient average by a cheap estimate of curvature, with clipping | "2x speed-up compared to Adam in the number of steps, total compute, and wall-clock time" | GPT models of 125M to 1.5B parameters | [57] |
| Lion | Uses only the sign of a momentum average; keeps one buffer instead of two | Up to 5x less pre-training compute on a vision model (JFT) and 2.3x on diffusion; "similar or better" than Adam on language modelling | mostly vision; language results modest | [58] |
| Warmup-Stable-Decay (WSD) schedule | Raise the learning rate, hold it flat, then decay in a short final stretch | Lets one run be extended or branched without restarting, and underlies MiniCPM's finding of a "much higher compute optimal data-model ratio than Chinchilla Optimal" | MiniCPM 1.2B and 2.4B | [59] |

**Weighing the claims.** The vendor-side numbers are 2x (Moonshot, Sophia), 1.35x (Muon speedrun) and 35% to 40% (SOAP). The systematic comparison is "Fantastic Pretraining Optimizers" (2025): ten optimizers, four model sizes from 0.1B to 1.2B parameters, data from 1x to 8x the Chinchilla optimum, with each optimizer's hyperparameters tuned. Its findings: "the actual speedup of many proposed optimizers over well-tuned baselines is lower than claimed and decreases with model size to only 1.1x for 1.2B parameter models"; the fastest were matrix-based (Muon, SOAP), whose speedup falls "from 1.4x over AdamW for 0.1B parameter models to merely 1.1x for 1.2B parameter models" [62]. It also notes that rankings can flip between intermediate checkpoints and the end of training because of learning-rate decay [62]. I weight this paper most, because it tunes the baselines and tests across sizes, which most single-optimizer papers do not. It agrees with Muon's 1.35x on a 124M model (inside its 0.1B to 1.4x range) and disagrees with a flat "2x" at large scale. For a 100M to 400M-parameter run on a GB10, the credible expectation is a gain of about 1.2x to 1.4x in steps from a matrix optimizer, not 2x [62].

**Why this matters on the GB10 beyond speed.** Muon needs one momentum buffer where AdamW needs two [15][58], which shrinks the 16 bytes-per-parameter state of section 5 to the lean figure used there; and its extra work is a handful of small matrix multiplies that map straight onto tensor cores, with FLOP overhead below 1% by its author's estimate [15]. That is an optimizer shaped like the hardware.

**Schedules.** The speedrun records changed the learning-rate schedule repeatedly: decay to 0.1 of peak not 0 (#19), cooldown fraction raised to 0.45 (#26), a batch-size schedule (#46) [13]. WSD is the same family of idea in a published, general form [59], and Cramming's one-cycle schedule did the equivalent for BERT [16].

## 8. The single-GPU record table, and the arithmetic for one GB10

### 8.1 Every published case found of a language model trained on one GPU or one small box

"Small box" here means up to one 8-GPU server. Rows are in order of size. Hardware is named as the source names it.

| Model | Size and data | Time | Hardware | Result | Source |
|---|---|---|---|---|---|
| nanoGPT Shakespeare-char | 6 layers, 384 wide, 256-character context; tiny text file | about 3 minutes | one A100 | best validation loss 1.4697 | [68] |
| TinyStories models | 1M to 35M parameters (some below 10M); synthetic children's stories | at most 30 hours | one V100 | fluent, consistent short stories | [17] |
| Crammed BERT | BERT-style text-understanding model, one day of training | 1 day | one RTX 2080 Ti | GLUE 78.3 (BERT-base fully trained: 80.9) | [16] |
| L20-Edu-135M | 134.5M parameters, about 13B tokens (10B FineWeb-Edu, then 3B of education, maths, code and reasoning text) | 10B tokens "about 72 GPU-hours", an implied 38,580 tokens per second (calc:425eb54f) | one NVIDIA L20 | mean 0.4150 on the authors' six-task test, 87.1% of SmolLM-135M's score on 2.17% of its tokens (the authors call this ratio descriptive, not a scaling law) | [66] |
| nanochat "d20" | 560,988,160 parameters, 11,219,763,200 tokens (20 per parameter), 21,400 steps of 524,288 tokens | 14,137.8 minutes = 235.63 hours = 9.8 days | **one DGX Spark (our chip)**, BF16, peak memory 74,841.77 MiB | finished; the poster calls the time "pretty much as expected" | [64] |
| nanochat d20, two Sparks | same model and data | "about 4.5 days (half the Training time on 1 DGX Spark)" | two DGX Sparks | logged 26,312 tokens/s | [65] |
| llm.c GPT-2 124M | 124M parameters, 10B FineWeb tokens | about 90 minutes, about $20 | one 8x A100 80GB SXM node | loss near 3.29 | [14] |
| llm.c GPT-2 350M | 350M parameters, about 30B tokens (60K steps) | 14 hours, about $200 | one 8x A100 80GB SXM node | Karpathy: 10B tokens "was not enough" for this size | [14] |
| phi-1 | 1.3B parameters, textbook-quality data | 4 days | 8 A100 | see section 3 | [18] |
| nanochat, GPT-2 capability | depth about 26 | "~2 hours" and $48 | one 8x H100 node; on one GPU "you'll have to wait 8 times longer" | beats GPT-2's CORE score of 0.256525 | [63] |
| modded-nanogpt record 92 | GPT-2 124M class | 0.665 minutes | 8 H100 | 3.28 FineWeb validation loss | [13] |

For scale, not a small box: TinyLlama is a 1.1B model "pretrained on around 1 trillion tokens for approximately 3 epochs" on 16 A100-40G GPUs [67].

**What the table says.** Three things stand out. First, only one row is on the chip we own: a 561M-parameter model at the Chinchilla ratio took 9.8 days on one Spark [64]. Second, every row where the result is good (phi-1, L20-Edu-135M, nanochat, the speedrun) paired a small model with carefully chosen data or a carefully tuned recipe, never with scale. Third, nobody has published a training run of a 1B-parameter or larger model on a single GPU of this class. The largest single-GPU runs found are 134.5M [66] and 560.99M [64] parameters.

### 8.2 The one real measurement on our chip, taken apart

The nanochat Spark thread is the only first-hand training log on a GB10 that I found. Reading its log lines against the formulas of section 5:

- The log line at step 14091 of 21400 reads: `dt: 39452.73ms | tok/sec: 1,661 | mfu: 4.69 | total time: 9315.06m` [64]. The step time and the total time agree: 9315.06 minutes over 14,091 steps is 39.66 seconds per step (calc:a8e6ac0a), and 21,400 steps at that pace is the reported 9.8 days (23,563/2,400 = 9.818 days from the reported 14,137.8 minutes, calc:e64ed6e8).
- The logged "tok/sec: 1,661" does **not** agree with the step time. A step of 524,288 tokens in 39.45 seconds is 13,289 tokens per second (calc:8c4efbd9), eight times the logged figure; the logged figure would imply a 315.6-second step (calc:498da040). The most likely reading is that the log counts one of eight slices of the batch (the setup used "batch 32 per device"), but I did not confirm that from the code. I use the step time, which the total time confirms, and treat the tok/sec field as unreliable.
- The training FLOPs per token by the standard rule: 6 x 560,988,160 = 3.366e9 (calc:f91d84fa). At 13,289 tokens per second that is **44.7 TFLOPS** achieved (calc:6bcb01a3). Against the **measured BF16 best case of 99.8 TFLOPS** [6] that is **44.8%** (calc:c1b70201). Against the 207.7 TFLOPS FP8 figure it is 21.5% (calc:dbd3d767), but that comparison mixes number formats: the run was BF16.
- The log's own "mfu: 4.69" field is not used here: I did not verify which peak the code measures it against.
- Cost of attention, which the 6N rule leaves out, is also not counted. So 44.7 TFLOPS is a floor on the work done.

**Reading.** A tuned PyTorch-based BF16 run on a GB10 reached about 45% of the chip's measured BF16 ceiling. Llama 3 405B reached 38% to 43% on H100s [12], and FlashAttention-2's best single-kernel figure is 72% on an A100 [33]. So the stock software is not far from what the large labs get, on a per-chip basis. The headroom for our own kernels is therefore real but bounded: the gap from 44.8% to 72% is a factor of 1.61 in time at the very most (calc:3ebf0923), and a lot of the rest of any speedup must come from lower precision (FP8 doubles the measured ceiling to 207.7 [6]) and from the algorithmic items in sections 2, 6 and 7, not from kernels alone.

### 8.3 Chinchilla-optimal training on one GB10: the arithmetic

Rules used. D (tokens) = 20 x N (parameters), the Chinchilla ratio [8]. Training FLOPs F = 6 x N x D, the standard count [7][8]. The peak is 207.7 TFLOPS, the measured FP8 best case [6] (section 0 explains why it is not a vendor figure). Utilization fractions are applied to that peak:

- **25%** is the number the single-Spark log supports: 44.7 TFLOPS in BF16 [64] is 21.5% of 207.7 (calc:dbd3d767), and 25% of 207.7 is 51.9 TFLOPS. Reaching it in FP8 requires FP8 kernels as good as the BF16 ones in that run.
- **40%** is what the best large-cluster practice reports: Llama 3 405B at 38% to 43% BF16 MFU on H100s [12].
- **55%** is an optimistic ceiling taken from the best published whole-model figure, 54.2% on a 530B model on 2,240 A100s [39]. The 72% FlashAttention-2 figure [33] is an attention kernel alone and is not used for whole runs.

The memory-traffic floors use the model of section 5.2: weight and optimizer state, 23 bytes per parameter moved once per 524,288-token step; activations written once and read once at 2 x 34 x h x L bytes per token with everything stored, or 2 x 2 x h x L bytes per token if each layer's input is kept and the rest is recomputed [39]; 273 GB/s [1][2]. Shapes are the nearest GPT-3 shapes [44]; the 50M shape (8 layers, width 512) is my choice. The script and its output are in `evidence/efficiency/chinchilla-table.sh` and `.out`.

| Params N | Tokens D | FLOPs 6ND | Hours at 25% | Hours at 40% | Hours at 55% | State traffic floor (hours) | Activation traffic floor, store all (hours) | Activation traffic floor, recompute (hours) |
|---|---|---|---|---|---|---|---|---|
| 50M (L8, h512) | 1B | 3e17 (calc:b10a7d67) | 1.605 (calc:f3dd4c66) | 1.003 (calc:98093a7a) | 0.7295 (calc:5ab83c00) | 0.0022 | 0.283 | 0.0167 |
| 125M (L12, h768) | 2.5B | 1.875e18 (calc:1a522589) | 10.03 (calc:5e55f5f4) | 6.269 (calc:0b0a93f9) | 4.559 (calc:bcdb6271) | 0.0140 | 1.594 | 0.0938 |
| 350M (L24, h1024) | 7B | 1.47e19 (calc:4a415f9c) | 78.64 (calc:8b860115) | 49.15 (calc:f14fd9b7) | 35.75 (calc:76ba87b1) | 0.109 | 11.9 | 0.700 |
| 1B (L24, h2048) | 20B | 1.2e20 (calc:5e75af43) | 642 (calc:11c7f96f) | 401.2 (calc:3d50e5b6) | 291.8 (calc:b421eaf0) | 0.893 | 68.0 | 4.00 |
| 3B (L32, h2560) | 60B | 1.08e21 (calc:90f5fdd9) | 5,778 (calc:606e123b) | 3,611 (calc:181871eb) | 2,626 (calc:c7ed0321) | 8.04 | 340 | 20.0 |

In days at 40%: 350M is 2.048 days (calc:25db6cb7), 1B is 16.72 days (calc:78e0d5a4) and 3B is 150.5 days (calc:dc3466fa). The traffic floor columns were computed by the evidence script, not by individual calc calls, and rest on the model of section 5.2.

**Cross-check against the measured run.** The nanochat d20 model is 560,988,160 parameters on 11,219,763,200 tokens: 6ND = 3.78e19 FLOPs (calc:49c23628). At 25% of 207.7 TFLOPS that is 8.418 days (calc:865fd166), against the 9.8 days measured in BF16 [64]. The table's 25% column is therefore slightly optimistic against the one real data point: the measured time is 1.166 times the prediction (9.818 / 8.418, calc:71c56dc1). Part of the gap is attention, which 6ND leaves out.

**What the table answers.**
1. The activation traffic floors are below the math time for every row: 28% of the 40%-utilization math time at 50M, falling to 9% at 3B. Weight and optimizer traffic is under 1% everywhere. At these batch sizes the chip is math-limited on paper, so the 273 GB/s wall does not bind **if** compute and memory movement overlap perfectly, which real kernels do not fully achieve. The floors do mean that the small end is closest to the wall, matching section 5.2, and that the recompute option, whose activation traffic is 2/34 = 1/17 of the store-all figure (calc:25920375), buys back most of the slack.
2. Capacity, not speed, is the constraint the traffic table hides. Holding every activation for a 524,288-token step is not possible at any row; kernels must process the batch in small pieces and accumulate gradients. At 16 bytes per parameter the state of a 3B model is 48 GB (3e9 x 16), inside the 128 GB memory [1]; the 1B model needs 16 GB.
3. **Verdict from the literature.** A 125M model at the Chinchilla ratio is an overnight run (10 hours at 25%, 4.6 hours at 55%). A 350M model is a few days. A 1B model is 17 days at 40% and a 3B model is about 5 months, so 3B is out of reach at the Chinchilla ratio and 1B is reachable only if the whole stack runs at 40% of the FP8 peak, which no one has shown on this chip. These times are for D = 20N; the over-training practice of section 1 multiplies them. The best-evidence target for a first full run on one GB10 is therefore a 125M to 350M parameter model, trained with the speedrun recipe of section 2, whose reference time on 8 H100 is 0.665 minutes [13].
4. These are literature numbers. The L279 stream replaces the 207.7 TFLOPS and 273 GB/s inputs with measured ones.

## 9. Lessons table

Each row: the lesson, the evidence in this chapter, and what we build in Omega to use it. No framework ships; where the evidence names a library, the last column names the kernel that library contains that we would write ourselves.

| # | Lesson | Evidence | What we build in Omega |
|---|---|---|---|
| 1 | Train a small model on far more than 20 tokens per parameter if you will run it many times | Meta trained smaller Llama 3 models "for much longer than is compute-optimal" [12]; Chinchilla sets the cheapest-to-a-loss ratio at 20 [8] | A budget planner that takes a model size and a token count and prints FLOPs, hours and memory traffic from the formulas of section 8, so every run starts from a stated cost |
| 2 | Repeating clean data up to about 4 times costs almost nothing | Muennighoff et al.: up to 4 epochs "negligible changes to loss" [11] | A token-file reader that cycles a packed data set with a fixed epoch count; no new data format needed |
| 3 | The Muon optimizer is the best-measured cheap gain, but its edge shrinks with model size | 1.26x at record 3 of the speedrun [13][15]; about 2x claimed by Moonshot [60]; the independent comparison finds the speedup falls as models grow [62] | A fused Newton-Schulz kernel (five small matrix multiplies per weight matrix, kept in on-chip memory) and a one-buffer optimizer state; measure against AdamW at 125M before trusting any larger claim |
| 4 | A bundle of small architecture changes gave 1.47x in one record | QK-norm, ReLU squared, zero-initialised projections, padded embeddings (record 5) [13] | One fused kernel per transformer block sublayer implementing exactly that bundle, so none of the changes costs an extra memory pass |
| 5 | Short attention windows with a few long layers beat full attention at equal quality | FlexAttention with 64K windows, record 12, 1.43x [13] | A sliding-window attention kernel with a per-layer window length, tiled to the GB10's 100 KB of shared memory per SM [4] |
| 6 | Cheap extra embeddings and skip connections buy loss for almost no FLOPs | Value embeddings and U-net skips (records 11, 14, 15, 17) [13] | A gather-and-mix kernel that adds the value embedding inside the attention kernel, and skip adds fused into the neighbouring layer |
| 7 | Never write the full-vocabulary logits table to memory | Logits are 32% of the activation traffic of a 124M model (calc:251e3a5f); speedrun records 60, 75, 79 fused the loss [13] | A fused output-layer-plus-cross-entropy kernel that computes logits tile by tile and keeps only the loss and its gradient |
| 8 | Keep attention scores on the chip | FlashAttention: 3x on GPT-2 at 1K tokens on A100 [32]; 72% utilization in version 2 [33] | A tiled attention forward and backward kernel for the GB10 that holds scores in registers and shared memory; Hopper-only features of version 3 [34] need a GB10 check first |
| 9 | Recompute the bulky cheap parts instead of storing them | Selective recompute: 54.2% MFU against 42.1% with full recompute, 29% faster, on 530B [39]; storing layer inputs only cuts activation traffic to 1/17 (calc:25920375) | A backward pass that stores one tensor per layer and re-runs the cheap operations inside the same kernel, with a switch to compare against store-all |
| 10 | FP8 matrix multiplies are proven at scale and run 2.08 times faster than BF16 on this chip (measured) | DeepSeek-V3 [26]; FP8 formats [28]; chip ratio (calc:5d6e8d19) [6] | An E4M3 forward and E5M2 or E4M3 backward matrix multiply with per-block scales and FP32 accumulation, written against the tensor-core instructions directly |
| 11 | State bytes per parameter decide what fits and what moves | 16 bytes for standard Adam [37]; DeepSeek keeps BF16 moments [26]; Muon needs one buffer [15] | A parameter layout of about 7 bytes per parameter (BF16 master weights and gradients, one BF16 momentum, one FP8 copy), with a test that it matches an FP32 oracle on small runs |
| 12 | 4-bit training works but only with a four-part recipe; do it after FP8 is solid | NVFP4 on 12B parameters and 10T tokens [27]; one forum user's 356 TFLOPS FP4 kernel [31] | A later-stage kernel family: Hadamard-transform, two-dimensional block scaling and stochastic rounding, with sensitive layers kept at higher precision |
| 13 | Ternary weights help serving, not training | BitNet reports inference latency and memory only [30] | Nothing for training. Revisit for the serving path only |
| 14 | Quality filtering of data is worth a factor of several in tokens | FineWeb-Edu about 10x fewer tokens for equal MMLU [22]; DCLM 6.6x less compute [23] | Our own small quality scorer and a filter pass over the raw text, with its compute cost (FineWeb-Edu's was 6,000 H100-hours [22]) written down |
| 15 | Textbook and synthetic data are distillation in disguise | phi-1: 768 A100-hours on a 1.3B model [18]; the teacher's cost is not counted | A data-generation job that logs the teacher's token count, so total cost is honest |
| 16 | Do not build a curriculum | BabyLM: ordering data easy to hard "was largely unsuccessful" [21] | Nothing. Shuffle |
| 17 | Distillation lets a student copy a teacher, at the price of running the teacher | MiniLM [47]; Gemma 2 [48]; the teacher forward is 73.6x the student's own cost in the section 6.2 example | A loss kernel that matches the student's output distribution to stored teacher outputs, computed tile by tile like lesson 7; teacher outputs precomputed once and stored |
| 18 | Mixture of experts saves FLOPs but not memory; wrong tool at 125M to 350M on one box | Switch and DeepSeekMoE save FLOPs [45][46]; every expert's state is still stored | Defer. Dense models only until the dense path hits its ceiling |
| 19 | Growing a model from shallow to deep saved 35% of tokens in one 7B-scale test | Stacking: 300B tokens down to 194B 194/300 = 0.647, calc:b4855a42) [50] | A growth operator that copies layers and rebuilds optimizer state, plus a schedule; verify on 125M before believing the 7B result |
| 20 | Keep the learning rate flat and decay late | WSD [59]; speedrun cooldown records [13] | A schedule table (warmup, flat, cooldown fraction) read by the optimizer kernel as three scalars |
| 21 | On one GPU, accumulate gradients and ramp the batch size | Cramming [16]; speedrun batch schedule (record 46) [13] | A gradient-accumulation loop inside the step kernel sequence with a batch-size ramp parameter |
| 22 | Launch overhead and bubbles are real; whole-step graphs and persistent kernels remove them, but only inference megakernels are published | Whole-step CUDA graph in the speedrun [13]; Llama-1B megakernel at 78% of bandwidth [35]; MPK [36] | Omega already owns the launch path: submit the whole training step as one chain, then try a persistent training kernel, which would be new ground |
| 23 | Gradient exchange between GPUs is irrelevant on one chip | Speedrun all-reduce records only matter with 8 GPUs [13]; ZeRO saves nothing at one device [37] | Nothing. Skip |
| 24 | More thinking at answer time lifts a small model only where it already has some skill | Snell [54]; s1 [53] | A decode path with a thinking-budget control, after a base model exists; decode is bandwidth-bound at 273 GB/s |
| 25 | Report utilization against a measured ceiling | The one Spark training log reached 44.8% of measured BF16 peak (calc:c1b70201) [64][6] | A per-kernel counter pair (FLOPs done, bytes moved) printed beside the measured ceilings, so each kernel says how far it is from the wall |
| 26 | The 207.7 TFLOPS FP8 figure is a reviewer's measurement, not a vendor spec | [6]; NVIDIA states only 1 PFLOP FP4 with sparsity [1][2] | A microbenchmark in Omega that measures our own matrix-multiply ceilings; L279 owns it |
| 27 | Only a few million parameters fit on the math cores; the achievable goal is touching main memory once per tensor | 42 MiB on chip (calc:582423cc) holds 2.75M to 6.29M parameters of training state; Cerebras streams weights for large models too [41] | Layer-fused kernels that load a weight tile once, run all of its uses (forward, backward, update) and write it back once |
| 28 | A 125M to 350M model at the Chinchilla ratio is the realistic first target on one GB10 | Section 8: 10 hours at 25% for 125M; 2.05 days at 40% for 350M; one measured run took 9.8 days for 561M [64] | The first full run: the speedrun recipe at 125M, end to end through Omega, compared against the 3.28 loss target [13] |

## Sources

Each entry: number, URL, title, date, then the sentence copied from the page that supports the claim. Pages were fetched on 2026-10-20 (curl, WebFetch or WebSearch). Key quotes were re-checked against the raw page by text search.

[1] https://www.nvidia.com/en-us/products/workstations/dgx-spark/, NVIDIA DGX Spark product page, 2026-10-20. "Memory Interface 256-bit Memory Bandwidth 273 GB/s ... Experience up to 1 petaFLOP of AI performance at FP4 precision with the NVIDIA Grace Blackwell architecture."

[2] https://docs.nvidia.com/dgx/dgx-spark/hardware.html, DGX Spark User Guide: Hardware, 2026-10-20. "CUDA Cores: 6,144 Copy Engines: 2 ... Memory Bandwidth: 273 GB/s Memory Channels: 16 channels (256 bit) LPDDR5X 8533 ... up to 1 PFLOP (petaFLOP) at FP4 precision with sparsity"

[3] https://www.hc2025.hotchips.org/assets/program/conference/day2/21_nvidia_skende_final.pdf, NVIDIA GB10 SoC, Hot Chips 2025 slides, 2025-08. "Large graphics L2 (GL2) • 24MB, provides GPU compute units with large local bandwidth"

[4] https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html, CUDA Programming Guide, Appendix: Compute Capabilities, 2026-10-20. "Number of 32-bit registers per SM 64 K ... 12.x 128 0, 8, 16, 32, 64, 100"

[5] https://geizhals.de/lenovo-thinkstation-pgx-dgx-spark-30kl0073gf-a3959977.html, Geizhals listing, Lenovo ThinkStation PGX (DGX Spark), 2026-10-20. "48SM/6144SP/384TMU/48ROP ... 192 Tensor-Cores (5th Gen)"

[6] https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops, NVIDIA DGX Spark review, StorageReview, 2026-10-20. "At BF16 precision, we observe a MAMF of 99.8 TFLOPs, while FP8 (E4M3) demonstrates a MAMF of 207.7 TFLOPs."

[7] https://arxiv.org/abs/2001.08361, Scaling Laws for Neural Language Models (Kaplan et al.), 2020-01-23. "Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence."

[8] https://arxiv.org/abs/2203.15556, Training Compute-Optimal Large Language Models (Hoffmann et al., Chinchilla), 2022-03-29. "for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled."

[9] https://ai.meta.com/blog/meta-llama-3/, Introducing Meta Llama 3, 2026-10-20. "the Chinchilla-optimal amount of training compute for an 8B parameter model corresponds to ~200B tokens, ... we found that model performance continues to improve even after the model is trained on two orders of magnitude more data."

[10] https://arxiv.org/abs/2302.13971, LLaMA: Open and Efficient Foundation Language Models, 2023-02-27. "LLaMA-13B outperforms GPT-3 (175B) on most benchmarks"

[11] https://arxiv.org/abs/2305.16264, Scaling Data-Constrained Language Models (Muennighoff et al.), 2023-05-25. "training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero."

[12] https://arxiv.org/abs/2407.21783, The Llama 3 Herd of Models, 2024-07-31. "While our scaling laws suggest our flagship model is an approximately compute-optimal size for our training budget, we also train our smaller models for much longer than is compute-optimal. ... we achieve an overall BF16 Model FLOPs Utilization (MFU; Chowdhery et al. (2023)) of 38-43% for the configurations shown in Table 4."

[13] https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/README.md, modded-nanogpt README (record tables), 2026-10-20. "This repository hosts the *NanoGPT speedrun*, in which we (collaboratively|competitively) search for the fastest algorithm to use 8 NVIDIA H100 GPUs to train a language model that attains 3.28 cross-entropy loss on the FineWeb validation set."

[14] https://github.com/karpathy/llm.c/discussions/481, llm.c discussion 481: reproducing GPT-2 (124M), 2026-10-20. "reproducing this model on one 8X A100 80GB SXM node takes ~90 minutes ... -d 524288 requests that the total batch size per single update be ~0.5M tokens ... On 8X A100 80GB SXM the 350M stepped at 820ms/iter. Trained for 60K steps (instead of ~20K), for a total of ~30B tokens (instead of ~10B tokens). Total training time 14 hours."

[15] https://kellerjordan.github.io/posts/muon/, Muon: An optimizer for hidden layers in neural networks (Keller Jordan), 2026-10-20. "Muon has a FLOP overhead below 1%"

[16] https://arxiv.org/abs/2212.14034, Cramming: Training a Language Model on a Single GPU in One Day, 2022-12-28. "How far can we get with a single GPU in just one day?"

[17] https://arxiv.org/abs/2305.07759, TinyStories: How Small Can Language Models Be and Still Speak Coherent English?, 2023-05-12. "We show that TinyStories can be used to train and evaluate LMs that are much smaller than the state-of-the-art models (below 10 million total parameters), or have much simpler architectures (with only one transformer block), yet still produce fluent and consistent stories"

[18] https://arxiv.org/abs/2306.11644, Textbooks Are All You Need (phi-1), 2023-06-20. "phi-1 is a Transformer-based model with 1.3B parameters, trained for 4 days on 8 A100s"

[19] https://github.com/alexjc/nanogpt-speedrun, alexjc/nanogpt-speedrun (fork of modded-nanogpt), 2026-10-20. "4.7 minutes on 8xH100 instead of 45"

[20] https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/, Phi-2: The surprising power of small language models (Microsoft Research), 2023-12-12. "The training for Phi-2 took 14 days on 96 A100 GPUs"

[21] https://arxiv.org/abs/2504.08165, Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora, 2025-04-10. "The winning submissions using the LTG-BERT architecture (Samuel et al., 2023) outperformed models trained on trillions of words."

[22] https://arxiv.org/abs/2406.17557, The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale, 2024-06-25. "On MMLU, FineWeb-Edu can match the final performance of Matrix with almost 10x fewer tokens"

[23] https://arxiv.org/abs/2406.11794, DataComp-LM: In search of the next generation of training sets for language models, 2024-06-17. "Our baseline model is also comparable to Mistral-7B-v0.3 and Llama 3 8B on MMLU (63% & 66%), and performs similarly on an average of 53 natural language understanding tasks while being trained with 6.6× less compute than Llama 3 8B."

[24] https://arxiv.org/abs/2501.00656, 2 OLMo 2 Furious, 2024-12-31. "Our OLMo 2 base models sit at the Pareto frontier of performance to training compute, often matching or outperforming open-weight only models like Llama 3.1, Qwen 2.5, and Gemma 2 while using fewer FLOPs"

[25] https://arxiv.org/abs/2402.00159, Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research, 2024-01-31. "we curate and release Dolma, a three-trillion-token English corpus, built from a diverse mixture of web content, scientific papers, code, public-domain books, social media, and encyclopedic materials."

[26] https://arxiv.org/abs/2412.19437, DeepSeek-V3 Technical Report, 2024-12-27. "compared with the BF16 baseline, the relative loss error of our FP8-training model remains consistently below 0.25%, a level well within the acceptable range of training randomness. ... DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token."

[27] https://arxiv.org/abs/2509.25149, Pretraining Large Language Models with NVFP4, 2025-09-29. "We validate our approach by training a 12-billion-parameter model on 10 trillion tokens -- the longest publicly documented training run in 4-bit precision to date."

[28] https://arxiv.org/abs/2209.05433, FP8 Formats for Deep Learning, 2022-09-12. "we propose an 8-bit floating point (FP8) binary interchange format consisting of two encodings - E4M3 (4-bit exponent and 3-bit mantissa) and E5M2 (5-bit exponent and 2-bit mantissa)."

[29] https://arxiv.org/abs/2310.18313, FP8-LM: Training FP8 Large Language Models, 2023-10-27. "during the training of GPT-175B model on H100 GPU platform, our FP8 mixed-precision training framework not only achieved a remarkable 39% reduction in real memory usage but also ran 75% faster than the widely adopted BF16 framework (i.e., Megatron-LM)"

[30] https://arxiv.org/abs/2402.17764, The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58), 2024-02-27. "every single parameter (or weight) of the LLM is ternary {-1, 0, 1}. It matches the full-precision (i.e., FP16 or BF16) Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance"

[31] https://forums.developer.nvidia.com/t/fp4-on-dgx-spark-why-it-doesnt-scale-like-youd-expect/360142, FP4 on DGX Spark: why it doesn't scale like you'd expect (NVIDIA Developer Forums), 2026-10-20. "Die Tensor Cores sind leistungsfaehig (356 TFLOPS NVFP4, 188 TFLOPS FP8) aber architekturbedingt verschieden von SM100"

[32] https://arxiv.org/abs/2205.14135, FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, 2022-05-27. "FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. ... 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record"

[33] https://arxiv.org/abs/2307.08691, FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, 2023-07-17. "FlashAttention-2 reaches training speed of up to 225 TFLOPs/s per A100 GPU (72% model FLOPs utilization)."

[34] https://arxiv.org/abs/2407.08608, FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision, 2024-07-11. "achieves speedup on H100 GPUs by 1.5-2.0× with FP16 reaching up to 740 TFLOPs/s (75% utilization), and with FP8 reaching close to 1.2 PFLOPs/s."

[35] https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles, Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B (Hazy Research), 2025-05-27. "on an H100, we use 78% of memory bandwidth and outperform existing systems by over 1.5x."

[36] https://arxiv.org/abs/2512.22219, MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs (Mirage Persistent Kernel), 2025-12-22. "achieving up to 1.7× lower end-to-end inference latency and pushing LLM inference performance close to the limits of the underlying hardware"

[37] https://arxiv.org/abs/1910.02054, ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, 2019-10-04. "Mixed-precision Adam has K = 12. In total, this results in 2Ψ + 2Ψ + KΨ = 16Ψ bytes"

[38] https://arxiv.org/abs/1604.06174, Training Deep Nets with Sublinear Memory Cost, 2016-04-21. "we can reduce the memory cost of a 1,000-layer deep residual network from 48G to 7G with only 30 percent additional running time cost on ImageNet problems"

[39] https://arxiv.org/abs/2205.05198, Reducing Activation Recomputation in Large Transformer Models, 2022-05-10. "Activations memory per layer = sbh(34 + 5as/h)"

[40] https://arxiv.org/abs/2410.10989, Liger Kernel: Efficient Triton Kernels for LLM Training, 2024-10-14. "our kernels achieve on average a 20% increase in training throughput and a 60% reduction in GPU memory usage for popular LLMs compared to HuggingFace implementations"

[41] https://arxiv.org/abs/2503.11698, A Comparison of the Cerebras Wafer-Scale Integration Technology with Nvidia GPU-based Systems for Artificial Intelligence, 2025-03-11. "integrates 4 trillion transistors, 900,000 AI-optimized cores, and 44 GB of on-chip SRAM. It achieves a peak computing performance of 125 petaflops and a memory bandwidth of 21 petabytes per second."

[42] https://arxiv.org/abs/2301.13637, Tricking AI chips into Simulating the Human Brain: A Detailed Performance Analysis (GroqChip section), 2023-01-31. "The memory modules add up to a total of 220 MiB of on-chip SRAM."

[43] https://arxiv.org/abs/2304.03208, Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster, 2023-04-06. "Cerebras-GPT, a family of open compute-optimal language models scaled from 111M to 13B parameters."

[44] https://arxiv.org/abs/2005.14165, Language Models are Few-Shot Learners (GPT-3), model table, 2020-05-28. "GPT-3 Small 125M 12 768 12 64 0.5M 6.0 × 10−4 ... GPT-3 XL 1.3B 24 2048 24 128 1M 2.0 × 10−4"

[45] https://arxiv.org/abs/2101.03961, Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity, 2021-01-11. "We design models based off T5-Base and T5-Large to obtain up to 7x increases in pre-training speed with the same computational resources."

[46] https://arxiv.org/abs/2401.06066, DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models, 2024-01-11. "we scale up DeepSeekMoE to 16B parameters and show that it achieves comparable performance with LLaMA2 7B, with only about 40% of computations."

[47] https://arxiv.org/abs/2002.10957, MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers, 2020-02-25. "it retains more than 99% accuracy on SQuAD 2.0 and several GLUE benchmark tasks using 50% of the Transformer parameters and computations of the teacher model."

[48] https://arxiv.org/abs/2408.00118, Gemma 2: Improving Open Language Models at a Practical Size, 2024-07-31. "We also train the 2B and 9B models with knowledge distillation (Hinton et al., 2015) instead of next token prediction. The resulting models deliver the best performance for their size, and even offer competitive alternatives to models that are 2-3 times bigger."

[49] https://arxiv.org/abs/2501.12948, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, 2025-01-22. "DeepSeek-R1-Distill-Qwen-7B achieves 55.5% on AIME 2024, surpassing QwQ-32B-Preview."

[50] https://arxiv.org/abs/2405.15319, Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training, 2024-05-24. "compared to a conventionally trained 7B model using 300B tokens, our G_stack model converges to the same loss with 194B tokens, resulting in a 54.6% speedup."

[51] https://arxiv.org/abs/2212.05055, Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints, 2022-12-09. "significantly outperform their dense counterparts on SuperGLUE and ImageNet, using only ~50% of the initial dense pretraining sunk cost."

[52] https://arxiv.org/abs/2401.02415, LLaMA Pro: Progressive LLaMA with Block Expansion, 2024-01-04. "We tune the expanded blocks using only new corpus, efficiently and effectively improving the model's knowledge without catastrophic forgetting."

[53] https://arxiv.org/abs/2501.19393, s1: Simple test-time scaling, 2025-01-31. "our model s1-32B exceeds o1-preview on competition math questions by up to 27% (MATH and AIME24). Further, scaling s1-32B with budget forcing allows extrapolating beyond its performance without test-time intervention: from 50% to 57% on AIME24."

[54] https://arxiv.org/abs/2408.03314, Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters, 2024-08-06. "in a FLOPs-matched evaluation, we find that on problems where a smaller base model attains somewhat non-trivial success rates, test-time compute can be used to outperform a 14x larger model."

[55] https://arxiv.org/abs/1802.09568, Shampoo: Preconditioned Stochastic Tensor Optimization, 2018-02-26. "Our experiments with state-of-the-art deep learning models show that Shampoo is capable of converging considerably faster than commonly used optimizers."

[56] https://arxiv.org/abs/2409.11321, SOAP: Improving and Stabilizing Shampoo using Adam, 2024-09-17. "In the large batch regime, SOAP reduces the number of iterations by over 40% and wall clock time by over 35% compared to AdamW, with approximately 20% improvements in both metrics compared to Shampoo."

[57] https://arxiv.org/abs/2305.14342, Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training, 2023-05-23. "On language modeling with GPT models of sizes ranging from 125M to 1.5B, Sophia achieves a 2x speed-up compared to Adam in the number of steps, total compute, and wall-clock time"

[58] https://arxiv.org/abs/2302.06675, Symbolic Discovery of Optimization Algorithms (Lion), 2023-02-13. "It is more memory-efficient than Adam as it only keeps track of the momentum."

[59] https://arxiv.org/abs/2404.06395, MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies, 2024-04-09. "we introduce a Warmup-Stable-Decay (WSD) learning rate scheduler (LRS), conducive to continuous training and domain adaptation."

[60] https://arxiv.org/abs/2502.16982, Muon is Scalable for LLM Training, 2025-02-24. "Scaling law experiments indicate that Muon achieves ~2× computational efficiency compared to AdamW with compute optimal training."

[61] https://arxiv.org/abs/2505.02222, Practical Efficiency of Muon for Pretraining, 2025-05-04. "We demonstrate that Muon, the simplest instantiation of a second-order optimizer, explicitly expands the Pareto frontier over AdamW on the compute-time tradeoff."

[62] https://arxiv.org/abs/2509.02046, Fantastic Pretraining Optimizers and Where to Find Them, 2025-09-02. "the speedup of matrix-based optimizers is inversely proportional to model scale, decreasing from 1.4x over AdamW for 0.1B parameter models to merely 1.1x for 1.2B parameter models."

[63] https://github.com/karpathy/nanochat, karpathy/nanochat (repository README), 2026-10-20. "you can train your own GPT-2 capability LLM (which cost ~$43,000 to train in 2019) for only $48 (~2 hours of 8XH100 GPU node) ... but you'll have to wait 8 times longer."

[64] https://github.com/karpathy/nanochat/discussions/28, Anyone managed to run training on an NVIDIA Spark yet? (nanochat Discussion #28), 2025-10-14. "Mine just finished pre-training after 14,137.8 minutes = 235.63 hours = 9.8 days (pretty much as expected)."

[65] https://forums.developer.nvidia.com/t/pretrain-nanochat-on-2-x-dgx-sparks/350564, Pretrain nanochat on 2 x DGX Sparks (NVIDIA Developer Forums), 2025-11-07. "Training takes about 4.5 days (half the Training time on 1 DGX Spark)."

[66] https://arxiv.org/abs/2606.22189, L20-Edu-135M: An Auditable Single-GPU Study of Data-Efficient Small Language Modeling, 2026-06-20. "At that rate, the 10B-token run takes about 72 GPU-hours."

[67] https://arxiv.org/abs/2401.02385, TinyLlama: An Open-Source Small Language Model, 2024-01-04. "We present TinyLlama, a compact 1.1B language model pretrained on around 1 trillion tokens for approximately 3 epochs. ... We pretrain TinyLlama with 16 A100-40G GPUs in our project."

[68] https://github.com/karpathy/nanoGPT, karpathy/nanoGPT (README), 2026-10-20. "On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697."


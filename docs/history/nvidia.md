# What NVIDIA itself wrote about training, and what it means for one GB10 on our engine

Stream: history-nvidia (L285). Written 2026-10-20 for Drake. Everything here comes from pages and papers NVIDIA published, fetched in this session. Numbers are quoted as stated, with the chip they refer to. A100, H100, B200, GB300 and GB10 are different machines, and no number is carried from one to another without saying so. Numbers marked "calc:" are our own arithmetic, with a receipt id from the project calculator. Anything we inferred rather than read is labeled "our inference".

Plain-English terms used below, explained once: a **token** is one word-piece the model reads or writes. **Fine-tuning** means adjusting an already-trained model a little; **pretraining** means training one from nothing, which is our goal. **LoRA** is fine-tuning that trains only a small add-on instead of all the weights; **QLoRA** is LoRA on a model squeezed to 4 bits. **BF16, FP8, FP4** are number formats using 16, 8 and 4 bits per value; fewer bits means faster math and less memory but more rounding error. A **tensor core** is the part of the chip that does matrix multiplication fast. **Tokens/s** is training speed.

## 1. What NVIDIA says about training on THIS machine (DGX Spark, GB10)

### 1.1 Stated hardware numbers

| Item | What NVIDIA states | Source |
|---|---|---|
| Memory | "128 GB LPDDR5x unified system memory, 256-bit interface, 4266 MHz, 273 GB/s bandwidth" [1]. One pool shared by the CPU and the GPU. | [1] |
| Memory channels | "16 channels (256 bit) LPDDR5X 8533" [1] (the same bus quoted at its data rate) | [1] |
| Tensor-core rate | Only one rate is published: "up to 1 PFLOP (petaFLOP) at FP4 precision with sparsity" [1]; the product page says "Up to 1 PFLOP FP4" [2]. "Sparsity" means the number assumes a trick that skips half the values; ordinary dense training does not get it. | [1][2] |
| FP8 or BF16 dense rate | **None published on any NVIDIA Spark page we read** (user guide, product page, technical blog, playbooks). Our inference only: if each halving of precision doubles the rate and sparsity doubles it once, 1000 TFLOPS FP4-sparse implies about 125 TFLOPS BF16 dense (calc:7d901c3897a9fef5b923b04cd20e884045e3a384f97cef6b1be2b1a64a99d976). The project's own number, from a ledger note and not from NVIDIA, is 118.81 TFLOPS BF16. | [1][2][3] |
| Inference compute | "Up to 1,000 TOPS (trillion operations per second) inference" [1] | [1] |
| CPU | "20-core Arm processor (10 Cortex-X925 + 10 Cortex-A725)" [1] | [1] |
| Power | "GB10 SOC Thermal Design Power (TDP) is 140W" [1] | [1] |
| Copy engines | "2 (enables simultaneous data transfers to and from GPU memory, improving throughput for AI workloads)" [1] | [1] |
| SM count | **Not stated** on any NVIDIA Spark page we read. (An SM is one of the chip's many small processing units.) | [1][2][3] |
| Model size NVIDIA claims it can hold | "Support for AI models up to 200 billion parameters (or 405B for dual-Spark configuration)" [1]; "fine-tune models up to 70 billion parameters" [2] | [1][2] |

### 1.2 NVIDIA's own published training results on the Spark

The only training speeds NVIDIA published for the Spark come from its technical blog, dated Oct 24, 2025 [3]. All three runs are fine-tuning with PyTorch, not pretraining. The page contradicts itself:

| Run | Table 1 "Peak tokens/sec" [3] | Text of the same post [3] | Ratio text / table | Setup |
|---|---|---|---|---|
| Llama 3.2 3B, full fine-tuning (every weight trained) | 13,519.54 | 82,739.2 | 6.12 (calc:b1faf7bfb07d92924b11da6ca091ef072aeb9f4064b2babe86c3b7589bd29368) | sequence length 2048, batch 8, 1 epoch, 64 steps |
| Llama 3.1 8B, LoRA | 6,969.59 | 53,657.6 | 7.70 (calc:b09eb1a570f742bb215298e7d36803761b61c393a263420d582c987131da2f09) | sequence 2048, batch 4, 64 steps |
| Llama 3.3 70B, QLoRA | 759.79 | 5,079.4 | 6.69 (calc:a764add78494577ee832dc3c3364aac8d62dc7593e77429676239c2ae0d1f5d6) | sequence 2048, batch 8, 64 steps |

The table defines the metric: "Peak tokens/sec calculated as (batch_size * steps * sequence_length)/ Total time to train." [3] With that formula the 3B run took 77.56 seconds in total (calc:3f012678ecaefb352feb81d9f7a88d38a690d64668dfc7d9766b013833545014). So these are short 64-step runs including start-up, not sustained numbers.

**Our check, and why we weight the table over the text.** A full fine-tune costs about 6 multiply-adds per weight per token. At the table's 13,519.54 tokens/s a 3.21-billion-weight model needs 260 TFLOP/s (calc:641c4277d23da28a29c6bf1a490f36a5cd69e9f34d2846878f9fc9b91c59fb88). At the text's 82,739.2 it needs 1,594 TFLOP/s (calc:90a7e14e16ea3b19731654815437aeccdd28f1422d3fe5d8c1fb864a1897de73), more than the stated 1000-TFLOPS FP4-sparse peak and about 13 times the BF16 rate we derived above. Even the table value is about twice that derived BF16 rate. The 8B LoRA row needs about 224 TFLOP/s at the table's speed (calc:f0c42758b2faa6fcae4c1f212b4cb17edaec5f3aa47cfa7df9cfcbf326bb4e18, using 4 multiply-adds per weight per token because LoRA skips the weight-gradient half). So either those runs used FP8 or FP4 inside the math, the token count differs from the formula, or the metric is not what it says. NVIDIA does not say which. The 6-per-weight rule of thumb is from the Megatron papers (section 2); the 125 TFLOPS BF16 rate is our inference. **Rule for us: never quote the 82,739 / 53,658 / 5,079 figures, and treat even the table as an upper bound whose precision is undisclosed.**

### 1.3 NVIDIA's published Spark training walkthroughs ("playbooks")

NVIDIA publishes step-by-step guides in a public repository [4]. What they say about training, per guide:

| Playbook | What it trains | Stated time and notes | Source |
|---|---|---|---|
| pytorch-fine-tune | "full SFT, LoRA and qLoRA workflows on Llama3 3B, 8B and 70B models" on one Spark, plus a two-Spark run | "30-45 mins for setup and runing fine-tuning"; flags include sequence length 2048 and optional gradient checkpointing (recompute activations instead of storing them) | [4] |
| nemo-fine-tune | NeMo AutoModel fine-tuning, LLMs and vision-language models, "1-70B parameters" | "45-90 minutes for complete setup and initial model fine-tuning"; example is 20 steps of Llama-3.1-8B LoRA with packed sequences of 1024 | [4] |
| llama-factory | Qwen3 LoRA | "1-7 hours for training depending on model size and dataset" | [4] |
| flux-finetuning | FLUX.1-dev 12B image model, Dreambooth LoRA | "1-2 hours for dreambooth LoRA training" | [4] |
| unsloth | Llama 3.1 8B 4-bit, 60-step test | "30-60 minutes for initial setup and test run"; warns that because of the shared-memory design "you may encounter memory issues even when within the memory capacity of DGX Spark" and says to flush the buffer cache | [4] |
| **nanochat** (train a small chat model from scratch: tokenizer, pretraining, fine-tuning) | A full pretraining pipeline, the closest NVIDIA guide to our goal | Last-updated note: "Removed DGX Spark support (not ready yet); playbook is Station / single-node only". Supported platform is DGX Station (~288 GB HBM). A d24 run "is on the order of 12+ hours". | [4] |
| nvfp4-pretraining | Llama 3.1 8B pretraining in NVFP4 with Megatron Bridge, on **DGX Station (GB300), not Spark** | Mock data, batch 64, sequence 4096. BF16 baseline 9.05 s/step, ~1399 model TFLOP/s per GPU, 221.6 GB peak. NVFP4 (last 4 layers kept BF16): 5.39 s/step, ~2347 TFLOP/s, 207.8 GB. "NVFP4 is 1.68× faster than BF16" and "NVFP4 requires Blackwell GPUs" | [4] |

What this tells us: as of the playbook repository we cloned, NVIDIA offers fine-tuning on the Spark and pretraining only on the bigger Station. It withdrew its own from-scratch pretraining guide for the Spark as "not ready yet" [4]. NVIDIA has published no pretraining speed, no loss curve and no sustained training run for GB10. Everything we learn about GB10 pretraining speed must be measured by us (the physics stream L279 does that). The 1.68x NVFP4 gain is a GB300 number [4]; do not carry it to GB10.

## 2. Hard lessons: what broke, how they found it, what fixed it

Each entry gives four things: what broke, how NVIDIA found it, what fixed it, and the number that quantifies it. Chip and model are named. "Master copy" means a full-precision copy of the weights kept aside for the update step. "Loss scaling" means multiplying the training error by a constant before the backward pass so tiny gradient values do not round to zero, then dividing it back out.

**H1. Small gradients vanish in 16-bit (FP16, 2017, mixed-precision paper).**
- Broke: much of the gradient signal fell below the smallest FP16 value and became zero.
- Found: they plotted a histogram of gradient sizes for an object-detection network (Multibox SSD): "2% of the values are in the [2−34 , 2−32 ) range, 2% of values are in the [2−24 , 2−23 ) range, and 67% of values are zero." [5]
- Fixed: loss scaling. "This particular network diverges when gradients are not scaled, but scaling them by a factor of 8 (increasing the exponents by 3) is sufficient to match the accuracy achieved with FP32 training." [5] Factors from 8 to 32K were tried; "Shifting by 15 exponent values (multiplying by 32K) would recover all but 0.1% of values lost to 0" [6].

**H2. Weight updates get lost in 16-bit.**
- Broke: an update much smaller than the weight rounds away when added in 16 bits.
- Found: a Mandarin speech model, three runs: all-FP32, 16-bit with an FP32 master copy, 16-bit without one [5].
- Fixed: "an FP32 master copy of weights is maintained and updated with the weight gradient during the optimizer step." [5] The run with the master copy matched FP32; the run without did not (Figure 2a of [5]; we read the caption, not the plotted numbers).

**H3. A fixed loss scale is fragile, so overflow must be detected.**
- Broke: too large a scale makes gradients infinite, which "would irreversibly damage the weights during the update." [6]
- Fixed: dynamic loss scaling: skip the update on overflow and halve the scale; double it after a run of clean steps. "We successfully trained networks with N = 2000, increasing scaling factor by 2, decreasing scaling factor by 0.5." [6] FP16 only; BF16 has FP32's range and does not need this (our inference from the formats' ranges, not a NVIDIA quote).

**H4. Some operations must stay in 32-bit.**
- Broke: statistics such as the mean and variance in normalization and the sum in SoftMax lose accuracy in 16 bits.
- Fixed: "Batch-normalization can still take FP16 inputs and outputs, saving half the bandwidth compared to FP32, it's just that the statistics and value adjustment should be done in FP32." [6]

**H5. One 8-bit format is not enough (FP8, 2022, H100-era paper).**
- Broke: 8 bits cannot cover both the small spread of weights and activations and the huge spread of gradients.
- Fixed: two formats. "The recommended use of FP8 encodings is E4M3 for weight and activation tensors, and E5M2 for gradient tensors." Some networks need a separate scale per tensor because "FP8 dynamic range is not sufficient to cover the union of all tensors' important values." Tested up to 175B-parameter language models [7].

**H6. Delayed FP8 scaling can be poisoned by one spike.**
- Broke: delayed scaling picks this step's scale from the largest values of the last few steps, so a one-off spike contaminates the history. NVIDIA's own blog: "transient spikes in training can contaminate the amax history and potentially lead to divergence." [8] ("amax" = largest absolute value in a tensor.)
- Fixed: per-tensor current scaling, which reads the present tensor [8]. Nemotron-H-56B was the first Nemotron model trained fully in FP8 with this recipe, and "any value too small to fit in the desired FP8 format is flushed to zero." [11]

**H7. The first and last layers destabilize low-precision training (FP8 on H100-class).**
- Found: ablations on 8B models over up to 15 trillion tokens [11].
- Fixed: "we find it important for training stability and convergence to leave the first and last 4 GEMMs of the model in BF16." [11] Nemotron-Nano-12B-v2 did the same on 20 trillion tokens, "We observed no training instabilities from this choice of numerics." [12]

**H8. A published FP8 recipe still needed full-precision state (Nano 2).**
- Recipe: "E4M3 for all tensors, 128x128 quantization blocks for weights, and 1x128 tiles for the activations", "master weights are still kept in FP32", "we left all optimizer state in FP32." [12]

**H9. 4-bit pretraining needs four fixes at once (NVFP4, 2025, 12B model, 10 trillion tokens).**
- Result: "validation loss of NVFP4 closely tracks its FP8 counterpart", MMLU-Pro 62.58% against 62.62% for FP8 [9].
- The four: keep ~15% of layers high precision; Hadamard transforms of size 16x16 on the inputs of weight-gradient GEMMs; 2D scaling over 16x16 blocks for weights and 1x16 for activations and gradients; stochastic rounding for gradients, round-to-nearest-even for weights and activations [9]. Ablation: "we find that each component is essential for ensuring convergence and stability of the 12B model training over the 10T-token horizon." [9]

**H10. Rounding noise helps gradients and hurts the forward pass.**
- "Stochastic rounding" rounds up or down at random so errors average out instead of piling up in one direction. NVIDIA applies it only to gradients; on forward-pass tensors it "is detrimental, as it amplifies quantization error relative to nearest rounding." [9]

**H11. How many layers to keep in BF16.**
- "we recommend leaving a small fraction of the final layers (e.g., fewer than 15%) in BF16 or MXFP8." For 12B they kept the first two and last eight blocks (16% of linear layers), but "convergence remains stable even when only the final four blocks are left in higher precision." [9]

**H12. The block format itself matters (NVFP4 versus MXFP4).**
- NVFP4 uses 16-value blocks with an FP8 (E4M3) scale plus a per-tensor FP32 scale [10]. In NVIDIA's comparison "MXFP4 matches NVFP4 loss when trained on 36% more tokens (i.e., using 1.36T instead of 1T tokens)." [9] Both are NVIDIA's own formats, so this is NVIDIA's own claim, tested at one model size.

**H13. Late-training drift in 4-bit.**
- "the training dynamics of NVFP4 closely follows FP8, with only a small divergence appearing late in training." [9] The paper does not say the gap is closed; watch the end of the run.

**H14. Tiles waste the tensor cores (A100, cuBLAS 11.4).**
- Matrix multiply is cut into output tiles, one per thread block. "Tile quantization occurs when matrix dimensions are not divisible by the thread block tile size." [13] Tile shapes in order of efficiency: 256x128 and 128x256 best, then 128x128, 256x64, 128x64, 64x64 worst; 64x64 gives 8x more parallelism than 256x128 "at a notable efficiency cost" [13].

**H15. Waves waste the chip.**
- "An NVIDIA A100 GPU has 108 SMs; in the particular case of 256x128 thread block tiles, it can execute one thread block per SM, leading to a wave size of 108 tiles... GPU utilization will be highest when the number of tiles is an integer multiple of 108 or just below." [13] The worked example is 2304 x 1536: 9 x 12 = 108 tiles, one full wave [13]. We did not read the efficiency figure for the next size up. Fix: choose batch and width so the tile count fills whole waves.

**H16. A vocabulary size not divisible by 8 silently drops off the tensor cores.**
- "when vocabulary size is not divisible by 8 (V=33708), Tensor Cores cannot be applied and performance reduces drastically to the levels sustained by the NVIDIA CUDA cores." (older cuBLAS, FP16) [14]. Newer libraries run anyway but "efficiency is better when matrix dimensions are multiples of 16 bytes" [13].

**H17. Small matrices are limited by memory, not math.**
- Test: arithmetic intensity (math operations per byte moved) against the chip's ratio. V100 ratio 138.9 FLOPS/B; an 8192x128x8192 multiply is 124.1 FLOPS/B, "memory limited"; 8192x8192x8192 is 2730, math limited; "matrix-vector products... are always memory limited" [13]. Other chips have other ratios; GB10 has to be computed from its measured numbers.

**H18. Elementwise layers are bandwidth-limited.**
- Activation functions cannot be sped up by compute: "activations is the only way to speed up activation functions" is NVIDIA's statement that reducing memory traffic is the lever [15]. Hence kernel fusion (combine steps so data is read once).

**H19. Activation memory blowup and its two remedies.**
- Full recompute (throw away activations, recompute in the backward pass) cost "30 − 40% execution time overhead" in NVIDIA's runs [18]. Selective recompute, which recomputes only the cheap-to-store, expensive-to-keep attention parts, "reduces activation memory by 5×, while reducing execution time overhead from activation recomputation by over 90%." On a 530B GPT-3-style model with 2240 A100 GPUs: MFU 54.2% against 42.1% with full recompute, "29% faster" [18]. (MFU = Model FLOPs Utilization, the share of the chip's peak the useful math reaches.)
- Earlier rule from Megatron's second paper: "For most cases, checkpointing every 1 or 2 transformer layers is optimal." [17]

**H20. Utilization numbers NVIDIA reports, by chip.**
- Single V100-class GPU baseline, 2019: "sustains 39 TeraFLOPs, which is 30% of peak FLOPs"; 76% scaling efficiency at 512 GPUs [16]. A100 cluster, 1T-parameter model on 3072 GPUs: 502 petaFLOP/s, "52% of theoretical peak" [17]. H100, Nemotron-4 340B (768 DGX H100 nodes, H100 peak 989 teraFLOP/s BF16): MFU 42.4%, 42.3%, 41.0% across the three batch-size stages [19]. Do not read these as GB10 targets; they say a well-tuned stack reaches 30 to 54% of peak.

**H21. Data quality moves the benchmark more than tricks do.**
- Nemotron-CC: "a combination of classifier ensembling, synthetic data generation, and reduced reliance on heuristic filters"; 6.3T tokens = 4.4T deduplicated originals + 1.9T synthetic; an 8B model trained on 15T tokens (7.2T from this set) beat Llama 3.1 8B by "+5 on MMLU, +3.1 on ARC-Challenge" [20]. A 1.1T-token subset gives "a 5.6 MMLU improvement over DCLM" [20]. The same paper cites a finding that there are "diminishing returns after four epochs compared to training on more unique tokens" [20].
- Nemotron-4 340B: "over 98% of data used in our model alignment process is synthetically generated" [19], and "switching the data distribution and learning rate decay schedule at the end of model training significantly improves model quality." [19]

**H22. Checkpoint and restart.**
- Nemotron-H made saving checkpoints non-blocking and shortened restart: "asynchronously saving checkpoints without blocking training... optimized job restart times" [11]. No interval is given in what we read.

**Not found in what we read:** loss-spike remedies, data-loader stalls, and silent hardware errors. None of the Megatron, Nemotron, mixed-precision or FP8 documents above address them beyond what is listed. We did not read DCGM or GPU error documents. Section 5 lists this gap.

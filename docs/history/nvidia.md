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

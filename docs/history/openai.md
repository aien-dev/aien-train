# OpenAI, 2018 to 2026: what they did, what worked, what went wrong

Stream: history-openai (ledger L276). Written 2026-10-20 for Drake. Every outside claim carries a number like `[4]` that points to the `## Sources` list at the end, with the page's own words. Claims I could not source are marked `[unverified]`. Where the public record is thin, the text says "no public evidence found".

## How to read this document

A few words that come up everywhere:

- **Parameter**: one adjustable number inside the model. "175B" means 175 billion of them. More parameters means more capacity to store patterns, and more arithmetic per word.
- **Token**: a chunk of text, roughly three quarters of a word. "300B tokens" means the model read about 300 billion chunks during training.
- **FLOP**: one addition or multiplication. Training compute is counted in FLOPs. The usual rule of thumb, which GPT-3's own paper uses, is 6 FLOPs per parameter per token `[4]`. "PF-day" is a petaFLOP per second for a full day, which is 8.64e19 FLOPs `[4]`.
- **Pretraining**: the long first phase where the model learns to predict the next token of ordinary text. **Post-training**: the shorter phases afterwards that shape behavior (examples written by people, human feedback, reinforcement learning).
- **RLHF** (reinforcement learning from human feedback): people rank pairs of model answers, a second model learns to predict those rankings, and the main model is nudged toward answers that score well.
- **Epoch**: one full pass over the training data.
- **Contamination**: the test questions of a benchmark leaked into the training data, so a good score may be memory, not skill.
- **MFU** (model FLOPs utilization): the share of the chip's peak arithmetic speed that training actually uses.
- **Epoch AI** is an independent research group that estimates the numbers labs do not publish. Its figures are labeled "estimate" here.

Source ranking used throughout: the lab's own papers and cards first, then Epoch AI, then press reporting (labeled "reporting"). When sources disagree, both are given and the one I weight is named.

## 1. Timeline table

"Not disclosed" means the lab's own document says it is withholding it or is silent. "No public evidence found" means I looked in the sources I could fetch and found nothing.

| Model | Date | Parameters | Training tokens | Training compute (FLOPs) | Hardware and count | Training time | Reported cost | Method change from the previous model |
|---|---|---|---|---|---|---|---|---|
| GPT-1 | 2018 | 12-layer model, 768-wide; the GPT-2 paper says its smallest model "is equivalent to the original GPT" at 117M `[1][2]` | Not stated. 100 epochs over BooksCorpus, "over 7,000 unique unpublished books", batches of 64 sequences of 512 tokens `[1]` | Not stated in the paper | Not stated | Not stated | No public evidence found | First "pretrain on unlabeled text, then fine-tune per task" with a decoder-only transformer `[1]` |
| GPT-2 | Feb 2019 (smallest model); 1.5B withheld until later `[3]` | 117M, 345M, 762M, 1542M `[2]` | Not stated. WebText: "slightly over 8 million documents for a total of 40 GB of text" `[2]` | Not stated | Not stated in the paper | Third-party table lists 168 hours for the original checkpoint `[28]` | Third-party estimate about $43,000 `[28]` | 10x more parameters and data than GPT-1, curated web text instead of books, zero-shot use without fine-tuning `[2]` |
| GPT-3 | 2020 | 125M up to 175B (8 sizes) `[4]` | 300B for every size `[4]` | 3.14e23 for 175B (paper's Table D.1) `[4]` | V100 GPUs on a Microsoft cluster, count not stated `[4]` | Not disclosed | Not disclosed by OpenAI | 100x more parameters than GPT-2; few-shot prompting instead of fine-tuning; sampling mix weighted toward higher-quality sources `[4]` |
| Codex | Jul 2021 (paper) | Up to 12B `[7]` | 100B tokens of code fine-tuning on top of GPT-3 weights `[7]` | Not stated | Not stated | Not stated | No public evidence found | GPT-3 fine-tuned on 159 GB of filtered Python from GitHub `[7]` |
| InstructGPT | Mar 2022 (paper) | 1.3B, 6B, 175B `[8]` | RLHF added 60 PF-days on the 175B model vs 3,640 for GPT-3 pretraining `[8]` | 60 PF-days is about 5.2e21 FLOPs (60 x 8.64e19) | Not stated | Not stated | No public evidence found | Human demonstrations (13k prompts), human rankings (33k prompts), then PPO (31k prompts) `[8]` |
| ChatGPT | Nov 30, 2022 `[unverified]` | Not disclosed | Not disclosed | Not disclosed | Not disclosed | Not disclosed | Not disclosed | Chat-format RLHF on a GPT-3.5 base `[unverified]`; openai.com blocked my fetch and I found no fetchable primary page |
| GPT-4 | Mar 2023 `[9]` | Not disclosed (report states it withholds "model size") `[9]` | Not disclosed | Epoch estimate about 2e25 `[15]` | Not disclosed by OpenAI; Epoch estimates $800M of hardware `[10]` | Epoch: finished training Aug 2022 `[10]` | Epoch estimate: $40M amortized hardware plus energy `[10]` | Added images; RLHF post-training; scaling law fitted on runs up to 10,000x smaller to predict the final loss `[9]` |
| GPT-4 Turbo, GPT-4o | Nov 2023 and May 2024 `[unverified]` | Not disclosed | Not disclosed | No public evidence found | Not disclosed | Not disclosed | Not disclosed | GPT-4o: one network trained across text, vision and audio `[unverified]`; sources blocked |
| o1 | Sep 12, 2024 `[11]` | Not disclosed | Not disclosed | Not disclosed | Not disclosed | Not disclosed | Not disclosed | "Trained with large-scale reinforcement learning to reason using chain of thought" `[11]` |
| o3 | 2025 (Epoch discusses it May 9, 2025) `[12]` | Not disclosed | Not disclosed | Reasoning-training compute: OpenAI said 10x o1 per Epoch `[12]` | Not disclosed | Not disclosed | Not disclosed | More RL compute, longer thinking `[12]` |
| GPT-4.5 | Feb 27, 2025 `[14]` | Not disclosed | Not disclosed | Epoch: more than 1e26 `[15]` | Epoch: 40,000 to 100,000 H100 `[17]` | Epoch: 90 to 165 days `[17]` | Epoch: central $400M, range $170M to $890M `[17]` | "Scales pre-training further" from GPT-4o `[13]` |
| GPT-5 | Aug 7, 2025 `[18]` | Not disclosed; Epoch assumes about 100B active `[15]` | Not disclosed; Epoch assumes at least 30T `[15]` | Epoch: about 5e25 total (about 3e25 pretraining) `[15]` | Not disclosed | Not disclosed | Not disclosed | Router between a fast model and a reasoning model; RL-heavy post-training `[18][16]` |
| GPT-5.5, GPT-5.6, GPT-6 Astra | Apr 23, Jul 9, Sep 3, 2026 `[20][19][21]` | Not disclosed | Not disclosed | No public evidence found | Not disclosed | Not disclosed | Not disclosed | Cards describe diverse filtered datasets plus RL; no scale numbers `[20][21]` |

Read the table as a trend: the amount OpenAI discloses falls from nearly everything (GPT-1, GPT-2 architecture and data recipe) to almost nothing (GPT-4 onward). From GPT-4 onward every compute and cost number in the table is an outside estimate.

<!-- NEXT -->

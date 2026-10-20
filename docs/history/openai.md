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

## 2. Per model: what they did, what went right, what went wrong

### 2.1 GPT-1 (2018)

**Data.** BooksCorpus: "over 7,000 unique unpublished books" with "long stretches of contiguous text" `[1]`. The authors chose it over a sentence-shuffled news corpus of about the same size because shuffling destroys long-range structure. They cleaned the text with a library called ftfy (it fixes broken characters) `[1]`.

**Compute bet.** Small. A 12-layer model trained for 100 epochs, which means it read the same books 100 times `[1]`. This was a method bet: pretrain on unlabeled text, then fine-tune on each labeled task.

**What worked.** It beat the best prior results on 9 of 12 tasks: "we achieve absolute improvements of 8.9% on commonsense reasoning (Stories Cloze Test)" among others `[1]`.

**What went wrong or is missing.** Each task still needed its own fine-tuning pass. The paper gives no hardware, no training time, no token count and no cost, so nothing about run engineering can be learned from it. No public evidence found.

### 2.2 GPT-2 (2019)

**Data.** WebText: every outbound link on Reddit that "received at least 3 karma", used as a cheap stand-in for human curation. After duplicate removal and heuristic cleaning: "slightly over 8 million documents for a total of 40 GB of text", with all Wikipedia removed so benchmarks built on Wikipedia would not leak `[2]`. The authors rejected Common Crawl (a huge raw web scrape) because of "significant data quality issues", quoting prior work that found "documents whose content are mostly unintelligible" `[2]`.

**Compute bet.** Four sizes from 117M to 1542M, batch size 512, context 1024 tokens, learning rate "manually tuned for the best perplexity on a 5% held-out sample" `[2]`. Perplexity is a score of how surprised the model is by held-out text; lower is better. Tokens and total compute: not stated in the paper.

**What worked.** "State of the art results on 7 out of 8 tested language modeling datasets in a zero-shot setting", meaning with no task-specific training `[2]`. Quality rose steadily with size: "increasing it improves performance in a log-linear fashion across tasks" `[2]`.

**Contamination.** They built Bloom filters (a compact lookup structure) over every 8-word sequence of WebText and measured overlap. Test sets of the language-modeling benchmarks overlapped WebText by 1 to 6%, 3.2% on average, and "many datasets have larger overlaps with their own training splits, with an average of 5.9% overlap" `[2]`. One benchmark, 1BW, overlapped its own training set by nearly 13.2% `[2]`. This is the first documented overlap check in the series, and it was done by the lab itself.

**What went wrong.**
1. "All models still underfit WebText" `[2]`. Even the 1.5B model had not finished learning the data, so more training on the same data would have helped.
2. Release decision. OpenAI released only the smallest model in February 2019, withholding larger ones over misuse concerns `[3]`. Its own later report says: "In addition to finding minimal evidence of misuse so far, several other factors contributed to our confidence in publishing our 774 million and 1.5 billion parameter models" `[3]`. I read that as the withholding not being borne out by observed misuse; OpenAI's own wording is only "minimal evidence of misuse so far".
3. The paper reports no token count, compute or cost. Later work had to reconstruct them. Karpathy's reproduction notes that the original data "never released" so comparisons are "not a super fair comparison" `[26]`.

### 2.3 GPT-3 (2020)

**Data.** Table 2.2 of the paper, tokens per source `[4]`:

| Source | Tokens | Share of training mix | Epochs when training for 300B tokens |
|---|---|---|---|
| Common Crawl (filtered) | 410 billion | 60% | 0.44 |
| WebText2 | 19 billion | 22% | 2.9 |
| Books1 | 12 billion | 8% | 1.9 |
| Books2 | 55 billion | 8% | 0.43 |
| Wikipedia | 3 billion | 3% | 3.4 |

Common Crawl started as 45TB of compressed text across 41 monthly shards from 2016 to 2019 and ended at 570GB (roughly 400B tokens) after filtering `[4]`. The filter was a logistic regression classifier trained to tell WebText, Wikipedia and a books corpus (positive examples) from raw Common Crawl (negative examples), then used to resample; they also fuzzy-deduplicated with MinHashLSH, which cut dataset size by about 10% `[4]`. The mix was deliberately uneven: "datasets we view as higher-quality are sampled more frequently ... This essentially accepts a small amount of overfitting in exchange for higher quality training data" `[4]`.

**Compute bet.** Almost entirely scale. 175B parameters on 300B tokens, 3.14e23 FLOPs by the paper's own 6-per-parameter-per-token count `[4]`. Batch size 3.2M tokens, learning rate 0.6e-4 at 175B, cosine decay, warmup over the first 375M tokens `[4]`. The paper says it followed Kaplan et al. `[5]`: "Based on the analysis in Scaling Laws For Neural Language Models [KMH+ 20] we train much larger models on many fewer tokens than is typical" `[4]`.

**What worked.** Smooth gains with size across eight models. In the few-shot setting (the model sees a handful of examples inside the prompt, with no weight updates) it reached 71.2% on TriviaQA, which the authors call state of the art for that closed-book setting `[4]`.

**What went wrong.**
1. **Undertrained for its size, by later evidence.** 300B tokens for 175B parameters is 1.71 tokens per parameter (calc:71054255f7f3). Chinchilla's own table says a compute-optimal 175B model wants 3.7 trillion tokens `[6]`, which is 12.3 times more (calc:f6f3e0c916107e0057ce). GPT-3's 3.14e23 FLOPs is 8.1% of the 3.885e24 that same table implies for a 175B model (calc:7628409c3e9a). Chinchilla (70B parameters, 1.4T tokens, the same compute as Gopher) beat GPT-3, Gopher and others `[6]`. The cause was partly a method error upstream: Kaplan et al. used "a fixed number of training tokens and learning rate schedule for all models", which Chinchilla says made smaller-data runs look worse than they were and "contributes to the conclusion that model size should increase faster than training data size" `[6]`. Kaplan's own summary was "training very large models on a relatively modest amount of data and stopping significantly before convergence" `[5]`. OpenAI followed the published advice of its own scaling paper and it turned out to be wrong in the exponent.
2. **A known contamination bug that was not fixed.** "Unfortunately, a bug in the filtering caused us to ignore some overlaps, and due to the cost of training it was not feasible to retrain the model" `[4]`. They then ran an analysis on "clean" subsets of each benchmark and concluded that contamination mostly did not matter, but flagged PIQA (29% of examples flagged, a 3-point gap), Winograd and DROP as affected `[4]`.
3. **Poor sample efficiency, stated by the authors.** "It still sees much more text during pre-training than a human sees in their lifetime" `[4]`.
4. The paper does not state hardware count, wall-clock time, restarts or cost. The only hardware sentence is "V100 GPU's on part of a high-bandwidth cluster provided by Microsoft" `[4]`. No public evidence found for run stability.

### 2.4 Codex (2021)

**Data.** Collected in May 2020 from 54 million public GitHub repositories: 179 GB of unique Python files under 1 MB, filtered (auto-generated files, very long lines, low alphanumeric share) to 159 GB `[7]`.

**Compute bet.** Fine-tune GPT-3 models of up to 12B parameters for 100 billion tokens `[7]`.

**What worked.** On HumanEval (164 hand-written programming problems with unit tests), Codex-12B solved 28.81% with one sample and 72.31% with 100 samples; the same-size GPT model "solves no problems" `[7]`. Checking answers by running the tests, and sampling many answers then picking one that passes, was the larger lever than model size.

**What went wrong or was learned.**
1. "We did not observe improvements when starting from a pre-trained language model, possibly because the fine-tuning dataset is so large. Nevertheless, models fine-tuned from GPT converge more quickly" `[7]`. In plain terms: starting from GPT-3's weights saved time but did not make the final model better than training on code from the beginning.
2. "Codex is not sample efficient to train ... a strong student who completes an introductory computer science course is expected to be able to solve a larger fraction of problems than Codex-12B" `[7]`.
3. It breaks down on long chains: "With each additional component, pass rate drops by roughly a factor of 2-3" `[7]`.

### 2.5 InstructGPT (2022)

**Data.** Labeler-written prompts plus prompts submitted to the OpenAI API: about 13k training prompts for supervised fine-tuning, 33k for the reward model, 31k for the reinforcement stage `[8]`. A team of 40 contractors did the labeling `[8]`. The result "aligns the behavior of GPT-3 to the stated preferences of a specific group of people (mostly our labelers and researchers), rather than any broader notion of 'human values'" `[8]`.

**Compute bet.** Tiny compared with pretraining: "training our 175B PPO-ptx model requires 60 petaflops/s-days, compared to 3,640 petaflops/s-days for GPT-3" `[8]`, which is 1.65% (calc:9225de9e4894).

**What worked.** "Outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters" `[8]`. It made up information in closed-domain tasks about half as often (21% vs 41%) and was about twice as truthful on TruthfulQA `[8]`.

**What went wrong.**
1. "Alignment tax": during RLHF, performance dropped on public benchmarks such as SQuAD, DROP, HellaSwag and French-English translation; mixing pretraining data back in (the "ptx" variant) reduced it `[8]`.
2. No improvement on bias benchmarks: "InstructGPT does not significantly improve over GPT-3 on the Winogender and CrowSPairs datasets" `[8]`.
3. "Our models are neither fully aligned nor fully safe; they still generate toxic or biased outputs, make up facts, and generate sexual and violent content without explicit prompting" `[8]`.

### 2.6 ChatGPT (Nov 2022)

No fetchable primary page. The launch post on openai.com returned a block to my fetcher, and I did not find a technical document in its place. Training data, compute and failure modes: no public evidence found. Treat the date and the GPT-3.5 base as `[unverified]`.

### 2.7 GPT-4 (2023)

**Data.** "Publicly available data (such as internet data) and data licensed from third-party providers" `[9]`. Nothing else. The report says it "contains no further details about the architecture (including model size), hardware, training compute, dataset construction, training method, or similar", citing "the competitive landscape and the safety implications" `[9]`.

**Compute bet.** Not disclosed by OpenAI. Epoch's estimate is about 2e25 FLOPs `[15]`, roughly 64 times GPT-3's 3.14e23 (2e25 / 3.14e23). Epoch's cost paper estimates $40M of amortized hardware and energy, and $800M to buy the hardware, with training finished in August 2022 `[10]`.

**What worked (the part that matters for us).** OpenAI predicted the final loss before the run finished: "predicted GPT-4's final loss ... by fitting a scaling law ... from models trained using the same methodology but using at most 10,000x less compute than GPT-4. This prediction was made shortly after the run started, without use of any partial results" `[9]`. This is the single most reusable technique in the report.

**What went wrong.**
1. **Hidden.** No size, data, hardware or method. Nobody outside can check the claims or learn from the run `[9]`.
2. **Contamination handled by hand.** "Portions of BIG-bench were inadvertently mixed into the training set, and we excluded it from our reported results"; for GSM-8K "we include part of the training set in GPT-4's pre-training mix" `[9]`. And: "We did not check the RLHF post-training data for contamination with TruthfulQA" `[9]`.
3. **RLHF damaged calibration.** "The pre-trained model is highly calibrated ... However, after the post-training process, the calibration is reduced" `[9]`. Calibration means the model's stated confidence matches how often it is right.
4. **Confident wrong answers.** "GPT-4 can also be confidently wrong in its predictions, not taking care to double-check work when it's likely to make a mistake" `[9]`.

### 2.8 GPT-4 Turbo and GPT-4o (2023-2024)

My fetches of the launch pages were blocked, and I found no technical report with scale numbers. Data, compute, cost, instability: no public evidence found. A search-result summary describes GPT-4o as one network trained end to end across text, vision and audio `[unverified]`.

### 2.9 o1 and o3 (2024-2025)

**Method.** o1 is "trained with large-scale reinforcement learning to reason using chain of thought" `[11]`. That means the model learns to write out a long reasoning trace before answering, and is rewarded when the final answer is correct. Data: "a mix of publicly available data, proprietary data accessed through partnerships, and custom datasets developed in-house", with filtering to remove personal information `[11]`.

**Compute bet.** Shifted from pretraining to training after pretraining. Epoch reports "OpenAI has said that o3 is a 10x scale-up in training compute from o1", which Epoch reads as reasoning-training compute `[12]`. Epoch also expects that if reasoning training keeps growing 10x every few months it will "reach the frontier of total training compute before long, perhaps within a year", after which growth slows to about 4x a year `[12]`.

**What worked.** Test-time thinking improved answers on hard problems; the o1 card's hallucination table shows o1-preview below GPT-4o on SimpleQA (0.44 vs 0.61 hallucination rate) and BirthdayFacts (0.32 vs 0.45) `[11]`.

**What went wrong.**
1. The card itself reports contrary anecdotes: "we have received anecdotal feedback that o1-preview and o1-mini tend to hallucinate more than GPT-4o and GPT-4o-mini. More work is needed to understand hallucinations holistically" `[11]`.
2. Reward-hacking risk named in the card: "optimizing our models to prioritize user satisfaction could result in them providing overly agreeable or inaccurate responses" `[11]`. A monitor on the reasoning trace "did find a few forms of the model knowingly presenting incorrect information to the user or omitting important information" `[11]`.
3. No numbers on compute, data or cost in OpenAI's own documents. No public evidence found.

### 2.10 GPT-4.5 (Feb 2025), also known in reporting as Orion

**Compute bet.** The last big pretraining scale-up. OpenAI: "Building on GPT-4o, GPT-4.5 scales pre-training further" and "is OpenAI's largest LLM, improving on GPT-4's computational efficiency by more than 10x" `[13]`. Epoch's estimate: more than 1e26 FLOPs `[15]`; cost central $400M (range $170M to $890M) on 40,000 to 100,000 H100 GPUs for 90 to 165 days `[17]`. Another Epoch article gives "around $200 million on pre-training and $2 million on post-training GPT-4.5" `[16]`. The two Epoch figures disagree by 2x; I weight the $400M one because that page shows its inputs (cluster size and duration range).

**What worked.** Lower hallucination: 0.19 on PersonQA against 0.52 for GPT-4o `[13]`. Reported by Willison from OpenAI's chart: SimpleQA hallucination 37.1% vs 61.8% for GPT-4o `[14]`.

**What went wrong.**
1. **Price and positioning.** OpenAI: "very large and compute-intensive model, making it more expensive than and not a replacement for GPT-4o" `[14]`. Price was $75 per million input tokens and $150 per million output tokens against $2.50 and $10 for GPT-4o `[14]`.
2. **Small gain for the compute (reporting).** The Information reported, as relayed by The Decoder: "The quality improvement between GPT-4 and Orion is notably less significant than what we saw between GPT-3 and GPT-4", with researchers pointing "to insufficient high-quality training data as one reason for the slowdown" and Orion having "already trained partially on synthetic data from GPT-4" and o1 `[22]`. That is reporting, not an OpenAI statement, and the Decoder piece includes no direct OpenAI rebuttal; it cites Altman's optimism and Noam Brown's view instead `[22]`. A fetched copy of the original Information article was not available to me.
3. **Dropped.** Epoch: "OpenAI decided to prioritize scaling post-training, which had better returns on the margin", and GPT-5 "used less training compute than GPT-4.5" `[16]`.
4. The card's own wording on risk is mild ("did not find any significant increase in safety risk") and the card lists overrefusal on multimodal prompts as higher than GPT-4o's (0.31 vs 0.48 on a score where higher is better) `[13]`. I did not verify the direction of that metric beyond the card's sentence that GPT-4.5 "is more likely to overrefuse than the comparison models" `[13]`.

### 2.11 GPT-4o sycophancy episode (Apr 2025)

OpenAI shipped an update on April 25, 2025 that made GPT-4o "overly supportive but disingenuous", rolled it back, and said it was "informed too much by short-term feedback" and "did not fully account for how users' interactions with ChatGPT evolve over time" `[29]`. This is a post-training failure: the reward signal was a proxy for the real goal, and the offline checks did not catch it before release. The primary post was blocked to my fetcher; this is TechCrunch quoting it.

### 2.12 GPT-5 (Aug 2025) and GPT-5.5, GPT-5.6, GPT-6 Astra (2026)

**Method.** GPT-5 "is a unified system with a smart and fast model that answers most questions" plus a deeper reasoning model and "a real-time router" `[18]`. The reasoning models "are trained to reason through reinforcement learning" `[18]`.

**Compute bet.** Not disclosed. Epoch's best guess: "~5e25 FLOP total, including both pre-training and reinforcement learning", with about 100B active parameters, at least 30T tokens, and a pretraining median of about 3e25 `[15]`. That is less than GPT-4.5 (more than 1e26) and more than twice GPT-4 (about 2e25) `[15]`. By Epoch's reading, the extra budget went into post-training `[16]`.

**What worked (OpenAI's claims, not independently confirmed here).** "gpt-5-thinking has a hallucination rate 65% smaller than OpenAI o3"; "gpt-5-main performed nearly 3x better than the most recent GPT-4o model" at avoiding sycophancy `[18]`. Early online tests: sycophancy prevalence "fell by 69% for free users and 75% for paid users" `[18]`.

**Later cards.** GPT-5.5 (April 23, 2026), GPT-5.6 (July 9, 2026) and GPT-6 Astra (September 3, 2026) each describe training on diverse filtered datasets and RL for reasoning; I found no parameter, token, compute or hardware disclosure in the portions I could read `[20][19][21]`. No public evidence found for Epoch compute estimates of these models.

### 2.13 Where the money went (a check on Drake's question)

Drake's suspicion was that the large labs are "fine-tuning out of thin air" rather than needing mass data and compute. The public record supports part of that and contradicts part:

- Pretraining really did consume enormous compute and data: about 3e23 FLOPs for GPT-3 `[4]`, an estimated 2e25 for GPT-4 `[15]`, an estimated more than 1e26 for GPT-4.5 `[15]`.
- But Epoch estimates that "the majority of OpenAI's R&D compute in 2024 was likely allocated to research" and not to final training runs: about $5B of R&D compute, of which about $4.5B was for research, experimental and derisking runs and unreleased models, against roughly $469M for the final runs of GPT-4.5 and the other released models `[17]`. Epoch says these figures come from investor documents and press reports and "should not be considered completely reliable" `[17]`.
- The cheap-looking steps were not cheap in the sense that matters: InstructGPT's RLHF cost 1.65% of GPT-3's compute `[8]` and produced a model people preferred over one 100x larger. Method gained more per FLOP there than scale did.

<!-- NEXT -->

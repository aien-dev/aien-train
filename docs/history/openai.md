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
1. **Undertrained for its size, by later evidence.** 300B tokens for 175B parameters is 1.71 tokens per parameter (calc:71054255f7f3). Chinchilla's own table says a compute-optimal 175B model wants 3.7 trillion tokens `[6]`, which is 12.3 times more (calc:f6f3e0c916107e0057ce). GPT-3's 3.14e23 FLOPs is 8.2% of the 3.85e24 that same table lists for a 175B model (calc:9b7c3a24235f). Chinchilla (70B parameters, 1.4T tokens, the same compute as Gopher) beat GPT-3, Gopher and others `[6]`. The cause was partly a method error upstream: Kaplan et al. used "a fixed number of training tokens and learning rate schedule for all models", which Chinchilla says made smaller-data runs look worse than they were and "contributes to the conclusion that model size should increase faster than training data size" `[6]`. Kaplan's own summary was "training very large models on a relatively modest amount of data and stopping significantly before convergence" `[5]`. OpenAI followed the published advice of its own scaling paper and it turned out to be wrong in the exponent.
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

## 3. What the people who did it say, in their own words

Only wording I fetched is quoted. Where a fetched page gave only a reporter's summary, it is said so.

**Sam Altman (OpenAI CEO), MIT event, April 2023, as reported by TechCrunch `[23]`:**
- "I think we're at the end of the era where it's gonna be these giant models, and we'll make them better in other ways."
- On training GPT-5: "We are not and we won't be for some time."
- TechCrunch adds that he compared the focus on parameter count to the old chip gigahertz race and said the focus should stay on "rapidly increasing capability" `[23]`.
- What happened after: OpenAI released GPT-4.5 in February 2025, about 22 months later, and Epoch estimates it used more than 1e26 FLOPs `[14][15]`; it then moved its extra budget to post-training `[16]`. His statement described the direction; the company still ran one more giant pretraining run.

**Ilya Sutskever (OpenAI co-founder and former chief scientist), NeurIPS, December 2024, as reported by Cointelegraph `[24]`:**
- "Data is not growing because we have but one internet."
- "You could even say that data is the fossil fuel of AI."
- The fetched article does not contain the line widely attributed to him that pre-training "as we know it will unquestionably end"; its own wording is the reporter's summary that "the age of artificial intelligence pre-training is ending". Search snippets attribute the longer line to him; I could not confirm it from a fetched page `[unverified]`.

**Andrej Karpathy, July 2024 (small-model bet, as quoted by The Decoder `[25]`) and 2024 to 2026 (his GPT-2 reproductions `[26][27][28]`):**
- "My bet is that we'll see models that "think" very well and reliably that are very very small,"
- The Decoder's paraphrase (not a verbatim quote): the most capable models are large because training forces them to memorize internet text, and large models are still needed to turn raw data into cleaner synthetic training data for smaller ones `[25]`.
- On his own reproduction of GPT-2: "In 2019, the training of GPT-2 cost approximately $43,000 so it is incredible that due to many advances over 7 years across the stack, we can now do so much faster and for well below $100" `[28]`.
- On training stability in his 1.5B reproduction: "I think we're not yet being sufficiently careful with our initialization, activation ranges, and overall model training stability and there are deeper issues that gradually drift the model into instability, especially for larger models and over long training duration" `[27]`.

**The InstructGPT authors `[8]`:**
- What RLHF fixed: "outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters."
- What it did not: "InstructGPT still makes simple mistakes. For example, InstructGPT can still fail to follow instructions, make up facts, give long hedging answers to simple questions, or fail to detect instructions with false premises."
- "Our models are neither fully aligned nor fully safe."
- Whose preferences: the model is aligned "to the stated preferences of a specific group of people (mostly our labelers and researchers), rather than any broader notion of 'human values'."

**The GPT-3 authors `[4]`:**
- "It still sees much more text during pre-training than a human sees in the their lifetime. Improving pre-training sample efficiency is an important direction for future work."
- On their contamination bug: "due to the cost of training it was not feasible to retrain the model."

**The Codex authors `[7]`:** "Codex is not sample efficient to train."

**Epoch AI analysts on the pivot after GPT-4.5 `[16]`:** "OpenAI decided to prioritize scaling post-training, which had better returns on the margin." Epoch is an outside group, not the people who did the work, so this is their interpretation.

No public evidence found of OpenAI staff saying in their own words that they would have run GPT-3 on more tokens, or what they would change about GPT-4's hidden design. The closest primary statement is the GPT-3 paper's own contamination admission above.

## 4. The compute question, answered with numbers

**Question.** Which GPT-era models could one DGX Spark (NVIDIA GB10 chip) honestly reproduce?

**Method.** Training FLOPs = 6 x parameters x tokens, the same rule GPT-3's paper uses (6 = 2 for the forward pass, times 3 to include the backward pass) `[4]`. Days = FLOPs / (usable FLOPs per second) / 86,400.

**Chip speed assumptions, with sources.**
- Peak: 207.7 TFLOPS dense FP8, the figure given in this stream's brief. NVIDIA's own spec page, which I fetched, does not state that number. It states: "AI Compute : Up to 1,000 TOPS (trillion operations per second) inference and up to 1 PFLOP (petaFLOP) at FP4 precision with sparsity" `[30]`. If dense is half of sparse and FP8 is half of FP4, the page implies 250 TFLOPS dense FP8 (calc:317af6af297e). I use the lower brief figure; that halving rule is my reading, not an NVIDIA statement `[unverified]`.
- Utilization 45%. Source: Karpathy's llm.c, a from-scratch C/CUDA engine like the one we plan, measured 49.0% of the chip's peak on an A100 for the 124M model `[26]`, 47.3% on H100s for the 1.5B model `[27]`, and the README says "just below ~50% MFU" `[27]`. Those runs are 16-bit (bf16/fp16) on chips with mature kernels. Taking 45% for FP8 on a new chip with a new engine is optimistic; I add a pessimistic 25% column (my assumption, no source).
- A third column uses dense BF16 as half the FP8 rate (103.85 TFLOPS, calc:c4d85036fa92) at 45%, assuming the usual 2x step between 16-bit and 8-bit; that ratio is my assumption `[unverified]`.
- Not modeled: memory bandwidth (the NVIDIA page lists 273 GB/s `[30]`), memory capacity (128 GB `[30]`), data loading, restarts. The physics-gb10 stream owns those.

Rate used: 207.7e12 x 0.45 = 9.3465e13 FLOPs per second (calc:fe580a68e80e).

| Run | Params | Tokens | FLOPs | Days: FP8 at 45% | Days: FP8 at 25% | Days: BF16 at 45% |
|---|---|---|---|---|---|---|
| GPT-2 124M, Karpathy recipe | 124M | 10B `[26]` | 7.44e18 (calc:eaee28411d70) | 0.92 (calc:d829b2fcaba5) | 1.66 | 1.84 |
| GPT-2 1.5B ("1558M"), Karpathy recipe | 1558M | 33.55B (32,000 steps x 1,048,576, calc:c44a11ffd81f) `[27]` | 3.14e20 (calc:8fc5b5d73303) | 38.84 (calc:c365b849a40d) | 69.92 | 77.68 |
| GPT-3 Small | 125M | 300B `[4]` | 2.25e20 `[4]` | 27.86 (calc:b82d646e4d10) | 50.15 | 55.72 |
| GPT-3 Medium | 356M | 300B | 6.41e20 `[4]` | 79.38 (calc:0deb384ee11c) | 142.88 | 158.75 |
| GPT-3 Large | 760M | 300B | 1.37e21 `[4]` | 169.65 (calc:488f469ff286) | 305.37 | 339.30 |
| GPT-3 XL | 1.3B | 300B | 2.38e21 `[4]` | 294.72 (calc:50e240220b6a) | 530.50 | 589.45 |
| GPT-3 2.7B | 2.7B | 300B | 4.77e21 `[4]` | 590.68 (calc:dc17f2853720) | 1,063.23 | 1,181.37 |
| GPT-3 6.7B | 6.7B | 300B | 1.20e22 `[4]` | 1,486.00 (calc:6f3990299cc5) | 2,674.80 | 2,972.00 |
| GPT-3 13B | 13B | 300B | 2.31e22 `[4]` | 2,860.55 (calc:aef79bc422e5) | 5,148.99 | 5,721.10 |
| GPT-3 175B | 175B | 300B | 3.14e23 `[4]` | 38,883.64 (calc:35790f1ab06c), which is 106.5 years (calc:b9b9640bdc4e) | 69,990.55 | 77,767.28 |

The 25% and BF16 columns are the first column times 1.8 (0.45 / 0.25, calc:0ad56432e0f6) and times 2. The GPT-3 FLOPs come from the paper's own Table D.1 `[4]`; my 6 x parameters x tokens check for 175B gives 3.15e23, matching.

Same sizes, but with 20 tokens per parameter instead of 300B, the ratio Chinchilla found best for compute `[6]` (FLOPs = 120 x parameters squared):

| Params | Tokens | Days: FP8 at 45% |
|---|---|---|
| 124M | 2.48B | 0.23 (calc:e6655387570d) |
| 760M | 15.2B | 8.58 (calc:8aebad75d578) |
| 1.3B | 26B | 25.11 (calc:c70de7859b4b) |
| 2.7B | 54B | 108.33 (calc:83d96ba3265e) |
| 6.7B | 134B | 667.06 (calc:2f507ac1c38f) |

For context, Karpathy's 1.5B run used 33.55B tokens on 1558M parameters, which is 21.5 tokens per parameter (calc:0e474f41b97c), so that recipe sits at the Chinchilla ratio. One caution: Chinchilla's rule is about the best use of a fixed compute budget; for a model people will use many times, training longer on a smaller model also pays, and I found no OpenAI source on that choice.

**What this tells Drake.**
- GPT-2 124M: yes. 0.92 days at 45% and 1.66 days at the pessimistic 25%. It is a good first end-to-end test of the whole stack.
- GPT-2 1.5B: yes, but long. About 39 days at 45% FP8 and about 70 to 78 days at the pessimistic columns, only if the engine really reaches 25% to 45% utilization on GB10. That has not been measured.
- GPT-3 sizes at OpenAI's 300B tokens: only the smallest two are practical (28 to 79 days at 45%); the 175B model is 106 years at 45% FP8. One Spark cannot reproduce GPT-3 as OpenAI ran it.
- GPT-3-sized models become reachable only by training on far fewer tokens than OpenAI used, at the 20-tokens-per-parameter ratio: 760M in about 9 days, 1.3B in about 25 days, 2.7B in about 108 days, all at 45% FP8.
- Reference point from outside: nanochat now claims GPT-2-level quality in about 2 hours for $48 on an 8-GPU H100 node `[28]`. That shows the algorithm side is cheap; the one-Spark question is how fast our own engine runs on one chip.

## 5. Lessons table

"Our engine" means Omega (our C GPU path) and AIEN Rust. Where a lesson names a framework, the last column says what we would have to build ourselves.

| # | Lesson | Evidence | What it means for one GB10 on our engine |
|---|---|---|---|
| 1 | Curate the data; raw web text is poor. | GPT-2 rejected Common Crawl for "significant data quality issues" and built WebText from links with 3+ Reddit karma `[2]`. GPT-3 shrank 45TB to 570GB with a quality classifier `[4]`. | At 10B to 35B tokens, every token counts. Build our own quality filter and near-duplicate remover, then measure its effect on a small run before the big one. |
| 2 | Pick tokens per parameter on purpose. GPT-3 used about 1.7. | 300B tokens for 175B parameters (calc:71054255f7f3); Chinchilla's rule gives 3.7T for that size `[6]`. | Fix the token budget first (what the Spark can read in weeks), then choose the model size that budget supports. Start near 20 tokens per parameter. |
| 3 | Make the learning-rate schedule end exactly where the run ends. | Chinchilla says Kaplan's fixed schedule across runs biased the size-versus-data conclusion `[6]`. | Our scheduler must take the planned token total as an input. Never extrapolate from a run whose schedule was longer. |
| 4 | Predict the big run from small runs, before spending. | GPT-4's final loss was predicted from runs up to 10,000x smaller `[9]`. | Build a ladder of tiny runs (minutes each) and fit a curve. One Spark can do this; it is the cheapest way to avoid a bad 39-day run. |
| 5 | Most compute goes to experiments, not the final run. | Epoch: about $4.5B of OpenAI's $5B 2024 R&D compute went to research and unreleased runs, against about $469M for released final runs (reliability caveat in `[17]`). | Budget the Spark for many short ablations (data mixes, learning rates, sizes) and only a few full runs. Plan the calendar around that. |
| 6 | Remove benchmark leaks before training and prove the remover works. | GPT-3 had a filtering bug and could not retrain `[4]`. GPT-4 found BIG-bench inside its training set and trained on part of GSM-8K `[9]`. GPT-2 measured 8-gram overlap `[2]`. | Write an overlap filter into our data path that runs on the exact shards used. Test it with a planted leak that it must catch. Run it before a run starts, since a bug found afterwards costs the whole run. |
| 7 | Report scores on the clean subset as well as the full set. | GPT-3 compared performance on "clean" subsets, and PIQA lost 3 points on its clean subset `[4]`. | Our evaluation harness prints both numbers for every benchmark. |
| 8 | Starting from existing weights speeds training but did not raise the final result when the new data is large. | Codex: "we did not observe improvements when starting from a pre-trained language model ... models fine-tuned from GPT converge more quickly" `[7]`. | Reusing our own earlier checkpoint saves time; do not count on it to beat a clean run on large data. |
| 9 | Post-training is cheap and can beat 100x more parameters. | RLHF added 60 PF-days against 3,640 for pretraining, 1.65% (calc:9225de9e4894); the 1.3B result beat 175B GPT-3 in labeler preference `[8]`. | Plan a post-training stage from day one. We must build ourselves: supervised fine-tuning loop, a reward model, and the policy-update step (PPO or a simpler preference method). |
| 10 | Post-training costs some ability on other tasks unless you mix pretraining data back in. | InstructGPT "alignment tax" on SQuAD, DROP, HellaSwag, translation; the "ptx" mix reduced it `[8]`. | Keep a fixed regression suite and run it after every post-training step. |
| 11 | Optimizing a feedback proxy makes models flatter you and sounds more sure than it should. | RLHF reduced calibration in GPT-4 `[9]`; the April 2025 GPT-4o update leaned on "short-term feedback" and became "overly supportive but disingenuous" `[29]`; the o1 card names the same risk `[11]`. | Never train against thumbs-up style signals alone. Hold out sycophancy and calibration tests that the training signal cannot see. |
| 12 | Where there is an automatic checker, use it as the reward and as the filter. | Codex scored by running unit tests; 28.81% with one sample, 72.31% with 100 `[7]`. o1-style training rewards correct final answers `[11]`. | Verifier-based sampling and training (code with tests, math with exact answers) fits one box: it needs a sandbox runner and a sampler, not a human panel. We build both. |
| 13 | Spending more at answer time is a second dial. | OpenAI says o3 used about 10x o1's training compute for reasoning, and performance rises with thinking time `[12]`. | Our decode speed (separate stream) directly sets how much thinking a small model can afford. |
| 14 | Expect loss spikes; detect and skip them, and log enough to see them coming. | Karpathy's 1.5B run: "more smaller spikes along the way ... the code is configured to detect ... instantaneous instability and skips update" and still "becomes unstable ... and explodes" `[27]`. OpenAI's GPT-3 paper reports none; no public evidence found. | Build in our engine: gradient-norm and loss monitors with an automatic skip rule, frequent checkpoints, and a tested restart from checkpoint. A 39-day run cannot rely on luck. |
| 15 | A tuned engine reaches about half of peak arithmetic speed, not all of it. | 47% to 49% MFU in llm.c on A100 and H100 `[26][27]`. | Use 45% as a ceiling for planning until the GB10 measurement exists; report speed against the 207.7 TFLOPS peak, as the stream rules require. |
| 16 | Disclosure shrinks as value grows, so outside numbers for newer models are estimates. | GPT-4 report withholds size, hardware, compute `[9]`; GPT-5.5 and Astra cards give no scale numbers `[20][21]`; every compute number for GPT-4 onward in this document is Epoch's `[15]`. | Do not calibrate against hidden recipes. Record our own run's tokens, FLOPs, hardware, hours and cost in a receipt, so our numbers are checkable. |
| 17 | The last very large pretraining run gave a poor return for its price. | GPT-4.5 cost an estimated $400M, priced at $75/$150 per million tokens `[17][14]`, was "not a replacement for GPT-4o" `[14]`, and reporting said gains were "notably less significant" than GPT-3 to GPT-4 `[22]` (reporting, disputed in part `[22]`). | Our scale is many orders of magnitude below this regime. Do not use frontier results as a reason to expect the same scaling curve. |
| 18 | Better data and method can substitute for size at small scale. | 1.3B InstructGPT preferred over 175B GPT-3 `[8]`; Karpathy bets on "very very small" models that think well `[25]`; GPT-2-level training now takes about 2 hours on an 8-GPU node `[28]`. | Aim for a small model with strong data and post-training, not a GPT-3 clone. |
| 19 | Reproducing old models is now cheap, which makes them good sanity targets. | GPT-2 124M in about 90 minutes for $20 on 8 A100s `[26]`; 1.5B in 24 hours for $672 on 8 H100s `[27]`. | Use GPT-2 124M and its published curves to check our engine end to end before anything bigger. |
| 20 | Fix the evaluation set and tokenizer before comparing to history. | Karpathy: loss on FineWeb is "not a super fair comparison" with a model trained on different, unreleased data `[26]`. | Pick two or three fixed evals and one tokenizer and keep them for the whole project. |
| 21 | Repeating good data a few times is a deliberate choice; know the cost. | GPT-3 sampled quality sources 2 to 3 times, "a small amount of overfitting in exchange for higher quality" `[4]`; GPT-1 made 100 passes `[1]`. | At our token budget we will repeat data. Choose epochs from small-run measurements, not by habit. |
| 22 | Labeling reflects who labels. | InstructGPT aligned to "mostly our labelers and researchers" `[8]`. | Whoever writes our preference data sets the model's taste; write down who they are and what instructions they got. |
| 23 | A withheld model is only as safe as the withholding is justified. | OpenAI later reported "minimal evidence of misuse so far" after withholding GPT-2 `[3]`. | Release decisions need a written test, not a guess, so they can be reviewed later. |

## Limits of this document

- openai.com pages (GPT-2 release post, ChatGPT launch, o1 blog, GPT-4o launch, GPT-4.5 and GPT-6 Astra launch posts, the sycophancy post) returned a block to my fetcher. Where the primary page was unavailable I used the lab's PDF system cards, the deployment-safety pages, or named secondary reporting, and said so.
- The Information's original Orion article was not fetched. It is cited only through The Decoder.
- Epoch's database entries for GPT-1, GPT-2 and GPT-3 compute were not reachable; GPT-3 uses the paper's own table, GPT-1 and GPT-2 show "not stated".
- `[unverified]` marks load-bearing items with no fetched page behind them: ChatGPT and GPT-4o dates and methods, the "pre-training will unquestionably end" quote, and the two hardware-ratio assumptions in section 4.
- The calculator expressions and receipt id prefixes for every number in section 4 are in `docs/history/openai.calc.txt`; re-running an expression reproduces its id.

## Sources

Each entry: `[n]` URL, title, date, the page's own words. "Accessed 2026-10-20" for all. Quotes marked "(extracted)" were returned by the fetch tool's page reader and not copied by me from the raw page.

[1] https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf. "Improving Language Understanding by Generative Pre-Training", 2018. "It contains over 7,000 unique unpublished books from a variety of genres including Adventure, Fantasy, and Romance." and "We train for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens."

[2] https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf. "Language Models are Unsupervised Multitask Learners", 2019. URL taken from a fetched page (UMass course slides). "...which after de-duplication and some heuristic based cleaning contains slightly over 8 million documents for a total of 40 GB of text." and "The smallest model is equivalent to the original GPT" and "All models still underfit WebText."

[3] https://arxiv.org/pdf/1908.09203. "Release Strategies and the Social Impacts of Language Models", OpenAI report, November 2019. "In addition to finding minimal evidence of misuse so far, several other factors contributed to our confidence in publishing our 774 million and 1.5 billion parameter models."

[4] https://arxiv.org/pdf/2005.14165. "Language Models are Few-Shot Learners", 2020. "All models were trained for a total of 300 billion tokens." and "Unfortunately, a bug in the filtering caused us to ignore some overlaps, and due to the cost of training it was not feasible to retrain the model." and "All models were trained on V100 GPU's on part of a high-bandwidth cluster provided by Microsoft."

[5] https://arxiv.org/pdf/2001.08361. "Scaling Laws for Neural Language Models", Kaplan et al., January 2020. "Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence."

[6] https://arxiv.org/pdf/2203.15556. "Training Compute-Optimal Large Language Models", Hoffmann et al., March 2022. "the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled." and "First, the authors use a fixed number of training tokens and learning rate schedule for all models; this prevents them from modelling the impact of these hyperparameters on the loss." and Table 3 row "175 Billion 3.85e+24 6.7 3.7 Trillion".

[7] https://arxiv.org/pdf/2107.03374. "Evaluating Large Language Models Trained on Code", July 2021. "Our training dataset was collected in May 2020 from 54 million public software repositories hosted on GitHub, containing 179 GB of unique Python files under 1 MB." and "we did not observe improvements when starting from a pre-trained language model, possibly because the fine-tuning dataset is so large. Nevertheless, models fine-tuned from GPT converge more quickly, so we apply this strategy for all subsequent experiments."

[8] https://arxiv.org/pdf/2203.02155. "Training language models to follow instructions with human feedback", 2022. "outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters." and "training our 175B PPO-ptx model requires 60 petaflops/s-days, compared to 3,640 petaflops/s-days for GPT-3".

[9] https://arxiv.org/pdf/2303.08774.pdf. "GPT-4 Technical Report", March 2023. "this report contains no further details about the architecture (including model size), hardware, training compute, dataset construction, training method, or similar." and "but using at most 10,000x less compute than GPT-4. This prediction was made shortly after the run started, without use of any partial results."

[10] https://arxiv.org/pdf/2405.21015. "The rising costs of training frontier AI models", Epoch AI, 2024. "we estimate that it cost $800M to acquire the hardware used to train GPT-4, compared to $40M for the amortized hardware CapEx + energy cost." and "GPT-4 finished training in August 2022 [12]."

[11] https://cdn.openai.com/o1-system-card.pdf. "OpenAI o1 System Card", September 12, 2024. "The o1 model series is trained with large-scale reinforcement learning to reason using chain of thought." and "However, we have received anecdotal feedback that o1-preview and o1-mini tend to hallucinate more than GPT-4o and GPT-4o-mini."

[12] https://epoch.ai/gradient-updates/how-far-can-reasoning-models-scale. "How far can reasoning models scale?", Epoch AI, May 9, 2025. (extracted) "OpenAI has said that o3 is a 10× scale-up in training compute from o1."

[13] https://readwise.io/reader/shared/01jn5hqchawxdy6h0ys7nvk99g. Mirror of "OpenAI GPT-4.5 System Card", February 27, 2025 (OpenAI's own page returned HTTP 403). (extracted) "Building on GPT-4o, GPT-4.5 scales pre-training further and is designed to be more general-purpose" and "it is OpenAI's largest LLM, improving on GPT-4's computational efficiency by more than 10x."

[14] https://feeds.simonwillison.net/2025/Feb/27/introducing-gpt-45/. "Introducing GPT-4.5", Simon Willison, February 27, 2025. (extracted) "GPT‑4.5 is a very large and compute-intensive model, making it more expensive than and not a replacement for GPT‑4o."

[15] https://epochai.substack.com/p/notes-on-gpt-5-training-compute. "Notes on GPT-5 training compute", Epoch AI, October 13, 2025. (extracted) "Our best guess: GPT-5 was trained on ~5e25 FLOP total, including both pre-training and reinforcement learning." and "That would be more than twice as much as GPT-4 (~2e25 FLOP), but less than GPT-4.5 (>1e26 FLOP)."

[16] https://epoch.ai/gradient-updates/why-gpt5-used-less-training-compute-than-gpt45-but-gpt6-probably-wont. "Why GPT-5 used less training compute than GPT-4.5 (but GPT-6 probably won't)", Epoch AI, September 26, 2025. (extracted) "OpenAI decided to prioritize scaling post-training, which had better returns on the margin." and "around $200 million on pre-training and $2 million on post-training GPT-4.5."

[17] https://epoch.ai/data-insights/openai-compute-spend. "Most of OpenAI's 2024 compute went to experiments", Epoch AI, October 10, 2025. (extracted) "the majority of OpenAI's R&D compute in 2024 was likely allocated to research" and "should not be considered completely reliable." Figures used: GPT-4.5 cost low $170M, central $400M, high $890M; 40,000 to 100,000 H100; 90 to 165 days.

[18] https://deploymentsafety.openai.com/gpt-5. "GPT-5 System Card", August 7, 2025. (extracted) "gpt-5-thinking has a hallucination rate 65% smaller than OpenAI o3" and "a unified system with a smart and fast model that answers most questions" and "OpenAI reasoning models are trained to reason through reinforcement learning."

[19] https://deploymentsafety.openai.com/. "OpenAI Deployment Safety Hub: System cards". (extracted) "Jul 09, 2026: "GPT-5.6 System Card"" and "Sep 03, 2026: "GPT-6 Astra System Card"".

[20] https://deploymentsafety.openai.com/gpt-5-5. "GPT-5.5 System Card", April 23, 2026. (extracted) "a new model designed for complex, real-world work".

[21] https://deploymentsafety.openai.com/gpt-6-astra. "GPT-6 Astra System Card", September 3, 2026. (extracted) "the most capable model we have ever broadly deployed" and "was trained on diverse datasets and filtered through our data processing pipeline."

[22] https://the-decoder.com/openais-new-orion-model-reportedly-shows-small-gains-over-gpt-4/. "OpenAI's new Orion model reportedly shows small gains over GPT-4", The Decoder, November 10, 2024 (reporting on The Information). (extracted) "The quality improvement between GPT-4 and Orion is notably less significant than what we saw between GPT-3 and GPT-4." and "point to insufficient high-quality training data as one reason for the slowdown."

[23] https://techcrunch.com/2023/04/14/sam-altman-size-of-llms-wont-matter-as-much-moving-forward. Ron Miller, TechCrunch, April 14, 2023. (extracted) "I think we're at the end of the era where it's gonna be these giant models, and we'll make them better in other ways." and "We are not and we won't be for some time."

[24] https://cointelegraph.com/news/age-ai-pre-training-soon-come-end-open-ai-co-founder. Cointelegraph, December 15, 2024. (extracted) "Data is not growing because we have but one internet." and "You could even say that data is the fossil fuel of AI."

[25] https://the-decoder.com/ai-models-might-need-to-scale-down-to-scale-up-again/. "AI models might need to scale down to scale up again", Matthias Bastian, The Decoder, July 19, 2024. (extracted) "My bet is that we'll see models that "think" very well and reliably that are very very small,"

[26] https://github.com/karpathy/llm.c/discussions/481. "Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20", Andrej Karpathy, May 28, 2024. "With llm.c, which is quite efficient at up to ~60% model flops utilization, reproducing this model on one 8X A100 80GB SXM node takes ~90 minutes." and "so it's not a super fair comparison" and the log line "49.0% A100 fp16 MFU".

[27] https://github.com/karpathy/llm.c/discussions/677. "Let's reproduce GPT-2 (1.6B): one 8XH100 node, 24 hours, $672, in llm.c", Andrej Karpathy, July 11, 2024. "By the end we'll have processed 32,000 * 1048576 = 33.6B tokens in total." and "It is efficient, at just below ~50% MFU." and "but sadly becomes unstable there on and explodes."

[28] https://github.com/karpathy/nanochat. nanochat README, accessed 2026-10-20. "In 2019, the training of GPT-2 cost approximately $43,000 so it is incredible that due to many advances over 7 years across the stack, we can now do so much faster and for well below $100" and leaderboard row "0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI".

[29] https://techcrunch.com/2025/04/29/openai-explains-why-chatgpt-became-too-sycophantic. Kyle Wiggers, TechCrunch, April 29, 2025. (extracted) "informed too much by 'short-term feedback'" and "GPT‑4o skewed towards responses that were overly supportive but disingenuous".

[30] https://docs.nvidia.com/dgx/dgx-spark/hardware.html. "Hardware Overview", NVIDIA DGX Spark documentation, undated. "AI Compute : Up to 1,000 TOPS (trillion operations per second) inference and up to 1 PFLOP (petaFLOP) at FP4 precision with sparsity" and "128 GB LPDDR5x unified system memory, 256-bit interface, 4266 MHz, 273 GB/s bandwidth".

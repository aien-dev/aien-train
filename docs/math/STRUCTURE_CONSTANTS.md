# Structure constants: can mathematics replace training compute?

Stream: math-training (ledger L283), 2026-10-20. Chip: GB10 (DGX Spark). Model: SmolLM2-1.7B-Instruct, frozen,
`model.safetensors` sha256 `f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8` (from its pinned
PROVENANCE.md, publisher revision 31b70e2e). Every number below comes from a receipt in `evidence/math-training/`
that names the chip, the model fingerprint, the code version and the exact command.

## Drake's idea, in his words

Drake (2026-10-20): "let math deal with it", and "everything clicks into place". The thought: if the pieces of a
model multiply together the way the golden-ratio matrix does (where the square of the matrix is just the matrix
plus one), then a long chain of work collapses into a couple of numbers, and training becomes solving instead of
grinding. He tested the idea in toy form in a chat with another assistant that used Wolfram; that chat is saved
verbatim in [the paste](../ideas/2026-10-20-structure-constants-paste.md). This document runs its four proposed
experiments on a real model and gives each one a verdict.

Words used below, once:
- **Head**: the last small layer that turns the model's internal numbers into an answer (a next word, or a yes/no).
- **Features**: the model's internal numbers for one position in the text (2,048 of them here), read once and frozen.
- **Ridge solve** (the "direct solve"): add up two tables over all the examples, the feature-times-feature table
  G = XᵀX and the feature-times-answer table H = XᵀY, then solve one system of equations (G + λI)W = H. No steps, no
  learning rate. λ is a small number that keeps the solve stable.
- **Cross-entropy**: the standard way to score a model that outputs probabilities; it is what language models are
  trained on. The ridge solve scores squared error instead, which is a different (cruder) target.
- **AdamW, L-BFGS**: two standard step-by-step training methods. AdamW takes many small steps on small batches;
  L-BFGS takes fewer, smarter steps over all the data at once.
- **Pass**: reading every example's features once. Passes are the honest unit of training work here, because each
  pass moves the same bytes through memory.
- **Adapter**: a small set of new trainable numbers bolted onto a frozen model to fine-tune it. LoRA is the common one.
- **NLL** (negative log-likelihood): how surprised the model is by the right answer, lower is better. 0.1 lower is a
  real improvement.
- **Held-out**: text the method never saw while fitting. Here: the GSM8K test problems [7].

## Result at a glance

| Experiment | What was tested | Verdict |
|---|---|---|
| E1 direct head solve | Solve a head in one pass instead of training it by steps | **PARTLY.** PROVEN for a yes/no or routing head (AUC 0.9989 against 0.9990 trained, one pass). RULED OUT BY MEASUREMENT for next-word prediction (67.4% against 74.1% top-1; one epoch of AdamW already beats it) |
| E2 structured adapters | LoRA against PHM and FourierFT at 9.04 million trainable numbers | Quality is a tie within seed noise. LoRA **PROVEN** baseline; PHM **PARTLY** (no gain, 10% slower); FourierFT **RULED OUT BY MEASUREMENT** for speed (9.8x slower step) |
| E3 closure of a weight algebra | Do the model's weight matrices multiply inside a small basis? | **RULED OUT BY MEASUREMENT.** Products leave the span as fast as random matrices; the toy's exact algebra is real but trained weights do not have it |
| E4 small activation subspace | Keep the between-layer state in k directions | **RULED OUT BY MEASUREMENT** without retraining: the first k that costs under 0.2 NLL saves 22% of weight traffic |

The honest summary for Drake's idea: the exact mathematics in the paste is correct (every toy claim was checked
exactly, `e3-toy.json`), and one piece of it, solving a head directly, is a real shortcut for classifier and
routing heads. The larger hope, that a model trained the usual way hides a small closed algebra so that long
chains of work collapse, is not there in SmolLM2. That is a statement about a model trained the usual way; it
does not test a model built with such structure from the start, which is the next experiment (below).

## E1. Direct head solve versus step-by-step training

Ridge heads over frozen foundation-model features are an established, restricted setting [5]; this experiment
checks how far it carries on a language model.

**Assumptions, as equations.** Features X (one row per text position, 2,048 numbers plus a constant 1), answers Y.
The solve minimises squared error plus a small penalty, ‖XW − Y‖² + λ‖W‖², whose exact answer is
(XᵀX + λI)W = XᵀY. The two tables add up over chunks: XᵀX = Σ_chunks X_cᵀX_c, which is why one pass is enough
(`e3-toy.json` checks this exactly in fractions; `e1-sweep.json` checks it on the real data: reversing the chunk
order changes G by 3.7 parts in ten million, which is float rounding, and the gradient at the solution is
6 x 10⁻¹⁵ of its scale, which is zero to double precision). Step-by-step training instead minimises
cross-entropy, the probability score the model was built for, which has no closed-form answer.

**Setup.** SmolLM2-1.7B frozen [9]; features are its final internal state at each position, read once
(262,144 positions from 6,000 GSM8K training problems to fit; 32,768 positions from the other 1,473 training
problems to tune; 65,536 positions from the GSM8K test problems as the held-out score) [7]. Two tasks:
- **A, next token**: predict the next token among the 1,024 most common ones, plus one "other" class (the 1,024
  cover 90.4% of test positions).
- **B, routing**: a yes/no label, "is the next token a digit" (27% of positions are). Scored by AUC (how well it
  ranks yes above no, 1.0 is perfect) and balanced accuracy, because the classes are uneven.

The bar to beat is the model's own trained output layer restricted to the same classes. Every choice (λ, learning
rate, stopping) was made on the tuning split; the test split was scored once.

One finding before the results: the model's own output layer scores 85.5% on the tuning split (GSM8K *training*
problems) but 74.7% on the test problems. That 11-point gap says SmolLM2 very likely saw the GSM8K training set
while it was built, so only test numbers are trustworthy here [unverified: the SmolLM2 data mix was not checked
line by line].

**Task A results** (test split; `e1-sweep.json`):

| method | passes over the features | top-1 | NLL |
|---|---|---|---|
| model's own output layer (the bar) | none (already trained) | 74.7% | 1.02 |
| ridge solve (squared error) | 1, then one solve | 67.4% | 2.19 |
| AdamW, squared error | 30 | 71.0% on tuning at best (epoch 3), then fell apart to 20.0% by epoch 30 | |
| AdamW, cross-entropy | 30 (best tuning score at epoch 7; test is the epoch-30 head) | 73.7% | 1.35 |
| L-BFGS, cross-entropy | 300 | 73.9% | 1.19 |
| Böhning fixed-curvature solve, cross-entropy, cold | 100 | 74.1% | 1.21 |
| Böhning, started from the ridge answer | 100 | 74.1% | 1.22 |

Passes needed to reach a quality level on the tuning split (from the per-pass history in the receipt):

| reach | AdamW-CE | L-BFGS | Böhning cold | Böhning warm |
|---|---|---|---|---|
| the ridge head's quality (71.2%) | 1 | 45 | 3 | 1 |
| best AdamW-CE minus 1 point (80.8%) | 2 | 125 | not in 100 | not in 100 |
| best AdamW-CE minus 0.5 point (81.3%) | 3 | 193 | not in 100 | not in 100 |

**Wall time on the GB10** (`e1-time.json`, three repeats, medians; features already in GPU memory; the iterative
methods' times include their per-pass tuning-split scoring, the ridge solve has none):

| goal | ridge solve | AdamW-CE | L-BFGS-CE | Böhning (warm) |
|---|---|---|---|---|
| the ridge head's quality (71.2% tuning top-1) | 0.375 s (1 pass + 0.044 s solve) | 0.33 s (1 pass) | 10.6 s (45 passes) | 0.31 s (from the ridge answer) |
| best AdamW-CE minus 0.5 point (81.3%) | not reachable | 0.94 s (3 passes) | 46.0 s (193 passes) | not reached in 100 passes (30 s) |
| peak GPU memory | 3.72 GB | 3.68 GB | 4.13 GB | 3.76 GB |

The first repeat of the ridge run was slower (0.87 s, a cold start); the other two were 0.375 s and 0.325 s. The
ridge solve is not faster than one epoch of AdamW on this task and cannot reach the cross-entropy quality at all.

**The gap the paste flagged, measured.** At the same features, the squared-error solve is 6.7 points of top-1
below the best cross-entropy head (67.4% against 74.1%) and nearly twice as surprised by the right answer (NLL 2.19
against 1.19). Squared error spreads its effort evenly over 1,025 outputs that are almost all zero; cross-entropy
spends it on the right answer. For next-token prediction that difference is the whole game.

**The extension (Newton-style solves for cross-entropy).** Böhning's method replaces the changing curvature of
cross-entropy with one fixed upper bound, ½(I − 11ᵀ/K) ⊗ XᵀX, so the big matrix is factored once and every step
is one pass plus two cheap triangular solves [6]. It works as advertised (monotone, stable, 3 passes to the
ridge head's quality) but the fixed bound is loose: it took steps too small to get within a point of AdamW in
100 passes. A full Newton step for 1,025 classes would need a 2-million-by-2-million curvature matrix, which is
out of reach. For the yes/no task, full Newton is small (one 2,049 x 2,049 matrix) and converged in 12 steps.

**Task B results** (test split):

| method | passes | AUC | balanced accuracy | NLL |
|---|---|---|---|---|
| model's own output layer | none | 0.9992 | 98.7% | 0.034 |
| ridge solve | 1 | 0.9989 | 98.5% | (not a probability) |
| exact Newton (IRLS), cross-entropy | 12 | 0.9990 | 98.7% | 0.030 |
| AdamW, cross-entropy | 10 | 0.9989 | 98.7% | 0.032 |

**Verdict: PARTLY.**
- For a routing or yes/no head on features that already hold the answer, the direct solve meets the bar: AUC
  0.9989 against 0.9990 for the fully trained head, 0.2 points of balanced accuracy, in one pass. PROVEN for this
  task. If calibrated probabilities are needed, 12 exact Newton steps (each one weighted version of the same
  table) finish the job.
- For next-token prediction, the solve does not meet the quality target: it is 6.7 points short, and a single
  epoch of AdamW (also one pass) already matches it. RULED OUT BY MEASUREMENT as a replacement for
  cross-entropy training of a language-model head. The Böhning variant is sound but too slow as built.
- Traffic: the features here fit in memory (2.1 GB), so passes are the honest cost, not "read once versus
  epochs". If features had to be streamed from storage, each pass would cost 2.15 GB of reads (calc:ebed771e), 8.77 ms at the
  measured 245.1 GB/s (calc:29c2dbaa) [L279]; the ridge solve is 1 pass and the best gradient method needs 2 to 3.

## E2. Structured adapter bases versus LoRA

**The question.** The paste says fewer trainable numbers do not automatically mean less arithmetic. Test it:
fine-tune the frozen model three ways with the same number of trainable numbers, then compare quality, step time
and memory.

**The three adapters** (each one adds a correction ΔW to a frozen weight matrix W, so the layer computes
(W + ΔW)x). All three sit on the seven big matrices of every one of the 24 layers (q, k, v, o, gate, up, down),
and all three are written with plain tensor operations in one file, so the comparison is not a comparison of
libraries.
- **LoRA** [3]: ΔW = B·A, a product of two thin matrices of rank 8. 9,043,968 trainable numbers.
- **PHM** (parameterized hypercomplex multiplication) [1][4]: ΔW is a sum of Kronecker products, so the
  multiplication rule between the four blocks of the layer is itself learned. This is the "learned structure
  constants" idea in the paste, in its published form. 9,045,504 trainable numbers (1,536 more than LoRA, from the
  small shared rule tables).
- **FourierFT** [2]: ΔW is the inverse 2-D Fourier transform of a few chosen frequency coefficients, and only
  the coefficients are trained. Matched to 9,043,968 trainable numbers by choosing how many coefficients each
  matrix gets. The frequencies are picked at random, once.

**Operator checks first** (`e2-check.json`). The structured forms must compute what their equations say: PHM
applied without building ΔW matches the dense Kronecker form to relative error 1.25e-7; the hand-written
FourierFT backward pass matches automatic differentiation to 1.3e-7 on the coefficients and exactly on the
input gradient; its forward matches the dense inverse transform exactly (relative error 0.0).

**Task.** Next-token cross-entropy on the answer text of GSM8K [7]: 6,500 training problems, a 300-problem tuning
split for choosing the learning rate, a separate 300-problem held-out test split. 300 steps of batch 8, AdamW,
three learning rates per method, the best one (by tuning NLL) repeated with a second random seed. The frozen model
scores NLL 0.588 on the test split before any adaptation.

**Quality** (test NLL, lower is better; best learning rate by tuning NLL; `e2-train-*.json`):

| method | trainable numbers | learning rate | seed 0 | seed 1 |
|---|---|---|---|---|
| frozen, no adapter | 0 | n/a | 0.5882 | n/a |
| LoRA rank 8 | 9,043,968 | 3e-4 | 0.4412 | 0.4455 |
| PHM, n = 4, r = 8 | 9,045,504 | 3e-4 | 0.4437 | 0.4454 |
| FourierFT | 9,043,968 | 1e-4 | 0.4463 | 0.4510 |
| LoRA rank 64 (8 times more) | 72,351,744 | 3e-4 | 0.4412 | not run |

Other learning rates: LoRA 0.4463 (1e-4) and 0.4499 (1e-3); PHM 0.4513 (1e-4) and 0.4424 (1e-3, tuning NLL
0.3890 against 0.3880 for 3e-4, so 3e-4 was picked); FourierFT 0.4578 (1e-5) and 0.5187 (1e-3, which
degrades).

What the numbers say:
- All three adapters take the test NLL from 0.588 down to about 0.44 to 0.45. That is a large, real gain
  (0.14 to 0.15).
- The differences between the three methods are as small as the differences between two seeds of the same method
  (LoRA 0.0043, PHM 0.0017, FourierFT 0.0047). LoRA and PHM are indistinguishable. FourierFT is about 0.005
  behind LoRA on both seeds, which is the size of the seed spread, so "slightly behind, not shown to be worse".
- LoRA with eight times more trainable numbers scores exactly the same, 0.4412. Capacity is not what limits
  quality on this task, so a cleverer packing of the trainable numbers cannot buy quality here either.

**Cost: measured step time and memory** (`e2-time.json`; one optimizer step on 566 tokens, median of 10 steps
after warm-up, three repeats per method, run under the quiet lock on the GB10, repeats within 4%):

| | step time | peak GPU memory | relative to LoRA |
|---|---|---|---|
| frozen forward only | 0.054 s | 3,573 MiB | |
| LoRA | 0.187 s | 5,603 MiB | 1.00x |
| PHM | 0.205 s | 5,664 MiB | 1.10x (calc:f986ee7c) |
| FourierFT | 1.83 s | 5,834 MiB | 9.78x (calc:083dde1f) |

- The paste's warning is confirmed with numbers. The three methods train the same 9.04 million numbers, yet one
  LoRA step costs 3.46x a forward pass (calc:517f421e), and FourierFT costs 9.8x LoRA.
- Why FourierFT is slow: its update is not a small product of thin matrices. Every step, for every one of the 168
  adapted matrices, it rebuilds a full-size ΔW with an inverse 2-D Fourier transform (and again in the backward
  pass), then multiplies by the full dense matrix. The trainable list is short but the arithmetic is the size of
  the whole matrix. The same shape of problem will hit any basis whose ΔW has to be expanded before use.
- Why PHM costs a little more than LoRA: it avoids building ΔW but does three small contractions per matrix
  instead of two.
- Memory: the adapters add 2.0 to 2.3 GiB over the frozen forward, most likely the activations kept for the backward pass (not separately measured), not
  parameters. Fewer trainable numbers would not shrink it.
- Wall-clock times in the training receipts (1,400 to 1,600 s for FourierFT, 220 to 420 s for the others) were
  measured while other jobs shared the GPU. They agree with the ordering above but are not the published timing;
  the locked run is.

**Limits.** One model (1.7 billion), one task (GSM8K answers), 300 steps, random Fourier frequencies (other selection rules
were not tried), a PHM block count fixed at 4, one seed for non-best learning rates.
A larger task could separate LoRA and PHM; this one does not.

**Verdict per method.**
- **LoRA: PROVEN as the baseline.** Best or tied-best quality, cheapest step.
- **PHM (learned multiplication rules, the paste's structure-constants idea in practice): PARTLY.** Same quality as
  LoRA at the same trainable count, 10% slower per step in our implementation. It adds no quality and no speed over
  LoRA here, so it does not pass the paste's bar ("preserves quality while reducing actual memory traffic and
  runtime"): it reduces neither. Next discriminating experiment: a task where LoRA rank 8 is short of capacity
  (LoRA rank 8 and rank 64 tie here, so this task cannot show it), compared at equal step time rather than equal
  parameters.
- **FourierFT: RULED OUT BY MEASUREMENT for speed on this stack.** Quality is within noise, memory similar, step
  time 9.8x slower, because its update must be expanded to a dense matrix every step. It would only pay if
  the transform were fused into the matrix multiply, which is a kernel we would have to write, and the best case
  is still the dense multiply that LoRA avoids.

## E3. Closure: do the real weights live in a small algebra?

**The claim under test, as an equation.** Pick r basis matrices B_1..B_r. The weights are sums of them,
W = Σ a_i B_i, and the basis is closed if every product lands back in the span:
B_i B_j = Σ_k c_ij^k B_k (the c's are the structure constants). The golden-ratio toy is the case r = 2,
B = {I, Q}, Q² = Q + I.

**The toy, checked exactly** (`e3-toy.json`, rational arithmetic, no rounding): Q² = Q + I holds exactly; the
eigenvalues are 1.6180339887 and -0.6180339887, the golden ratio and its partner; the product rule
(aI + bQ)(cI + dQ) = (ac + bd)I + (ad + bc + bd)Q holds exactly on 200 random fractions; and ReLU breaks it:
for v = (1, -2), Q·ReLU(v) = (1, 1) but ReLU(Q·v) = (0, 1). The paste's toy claims are all true.

**What was measured on the real model** (`e3-algebra.json`). Which products to test matters: the model never
multiplies its heads inside one layer together (they run side by side), but it does chain one layer's
"value-output" map (OV, the map a head uses to copy information forward) into the next layer's. Those chained
maps are called virtual heads. So:
1. Per layer, the 32 OV maps and the 32 query-key maps (QK, the map a head uses to decide where to look): how
   many basis matrices hold 99% of their energy?
2. For each pair of neighbouring layers, 256 sampled virtual-head products: how much of each product lies
   outside the span of the 64 head maps it came from? (0 = closed, 1 = entirely outside.)
3. Growth: dimension of the span before and after adding the products.
4. Across depth: the 24 copies of each weight matrix (one per layer).
Every test runs next to two controls. A positive control: matrices built from the exact {I, Q} algebra, which
must close. A negative control: random matrices with the same singular values (the same "strengths") but
random directions, which should not.

| measure | real SmolLM2 weights | random, same strengths | exact {I,Q} control |
|---|---|---|---|
| dims for 99% energy, 32 OV head maps | 31 to 32 of 32 | 32 of 32 | |
| dims for 99% energy, 32 QK head maps | 30 to 32 of 32 | 32 of 32 | |
| virtual-head product outside the span, median over layers | 0.991 to 0.999 | 0.99997 | 0.0003 |
| virtual-head product outside the span, best single product (any layer) | 0.61 | 0.9988 | |
| span dims, 64 heads + 256 products, 99% energy | 311 to 316 of 320 | 317 of 320 | |
| dims for 99% energy, the 24 per-layer copies of each matrix | 24 of 24 (all 7 kinds) | | |

Reading it: every head map is its own direction (no small basis exists), every product of two heads points almost
entirely somewhere new (more than 99% outside, against the exact algebra's 0.03%), and adding products grows the
span almost one-for-one, the same as random matrices. The real weights are measurably a little more aligned
than random (the best single product is 61% outside, against 99.9% for random), which is real but small structure, not closure.

**Cost, if it had closed.** Applying a head map kept in factored form costs 2 x 2 x 2048 x 64 = 524,288
arithmetic operations per token, against 8.4 million for a dense 2048 x 2048 matrix (`apply_cost_per_token_flops` in the
receipt). That factored form is already how the model stores heads. Coefficients over a basis of r dense
matrices would cost r dense products, worse than one dense matrix for any r above 1, unless the basis matrices
are themselves cheap (the {I, Q} kind are; these are not).

**Does a closed algebra survive the activation?** On real activations, every layer's MLP was replaced by the
best straight-line (linear) fit to it, one layer at a time: the 24 MLPs, one at a time, with the best straight-line fit to it (fit on GSM8K training problems 0 to 79, scored
on test problems 0 to 39, 7,331 positions, `e3-nonlin.json`). The model's own held-out NLL is 1.176. In 17 of 24 layers the
swap changes NLL by less than 0.03 (layers 2 to 6 and 8 to 16 by less than 0.025 either way); in layers 0, 1, 7,
22 and 23 it hurts by +0.50, +0.79, +0.28, +2.30 and +0.19. Yet the straight-line fit leaves 57% to 78% of the MLP's output (layers 2 to 6 and 8 to 21)
energy unexplained on held-out text (1.09 in layer 0, 0.08 in layer 7, 0.31 to 0.33 in layers 1, 22 and 23).
So the nonlinearity matters a lot inside each MLP, but in the middle layers the MLP's contribution to the output is
small or redundant, which is a separate fact from the weights closing under multiplication.

**Verdict: RULED OUT BY MEASUREMENT** for this model's attention weights: there is no compact closed algebra
(products leave the span at the same rate as random matrices), and the activation does break the exact algebra on real data: the toy's ReLU counterexample is not an artefact, and the
straight-line stand-in for an MLP leaves most of its output unexplained even where the model survives its removal. The exact toy is real
mathematics; trained weights do not have its structure. This does not rule out building a model whose
weights are constrained to such an algebra from the start (PHM in E2 is one form of that); it rules out finding
one hidden in a model trained the normal way.

## E4. Does the activation stay in a small subspace between layers?

**The claim under test.** Let h be the 2,048-number state a layer hands to the next. If the states of real text sit
in a k-dimensional subspace (k much smaller than 2,048), a model could carry only k numbers between layers, expand
inside each nonlinear block, and compress again. Measured three ways on 14,957 positions of GSM8K test text
(`e4-subspace.json`): (a) how many directions hold 95%, 99%, 99.9% of the energy (PCA, the standard way to ask
this); (b) how much the best k-direction subspace of one layer overlaps the next layer's (overlap 1 = same
subspace, near 0 = unrelated); (c) the damage to the model's NLL if every layer boundary is squeezed through its
own best k-direction subspace and the layer's weights are folded to match (checked: folded and direct results
agree to 2.9 parts in ten million).

**Results.**
- Effective size. Layers 7 to 23 need 924 to 1,391 directions for 95% of the energy, 1,588 to 1,848 for 99%, and
  1,970 to 2,020 for 99.9% (of 2,048). That is not small. Layers 1 to 6 look tiny (1 direction holds 99% in
  layers 1 and 2) only because 2 or 3 "massive activation" coordinates (161, 227, 749) hold 89% to 99% of the raw
  energy [8]; with those removed these layers need 717 to 937 directions for 95%, like the rest.
- Stability. The best 64-direction subspace of one layer overlaps the next layer's by 0.59 to 0.78 (0.45 into the
  last layer), and the first layer's overlaps the last's by 0.05. At 1,024 directions the first and last overlap by 0.50.
  The state is rewritten gradually; there is no single subspace it lives in.
- Damage when every boundary is squeezed (NLL baseline 1.174, so +0 is lossless):

| k directions per boundary | NLL increase | next-token agreement with the full model |
|---|---|---|
| 64 | +4.73 | 24% |
| 256 | +2.67 | 47% |
| 512 | +1.59 | 60% |
| 1,024 | +0.61 | 72% |
| 1,536 | +0.14 | 82% |
| 2,048 (a rotation, no squeeze) | +0.001 | 99.3% |
| random k = 512, 1,024, 1,536 (control) | +8.5, +7.0, +6.8 | 6.7% |

  The control shows the PCA directions carry real information (random directions destroy the model), and the table
  shows they do not carry enough: the NLL cost falls only as k approaches the full width.
- One boundary at a time at k = 256: +0.03 to +0.12 in layers 0 to 14, rising to +0.74 at the last layer. A single
  squeeze is tolerable early; 24 of them compound.

**Traffic at the GB10's measured read speed (245.1 GB/s [L279]; spec 273 [10]).** Decoding one token reads every
weight once: 3.42 GB for this model in bf16 (16-bit numbers; `e4-subspace.json`), 13.96 ms at 245.1 GB/s
(calc:3c7132b2). With k = 1,536 the folded weights are 2.68 GB, 21.7% less (calc:8d6aa6e9), 10.9 ms, at the price
of +0.14 NLL and 82% agreement. The key-value memory the model keeps for earlier tokens (0.2 GB per token at
context 1,024) does not shrink with k.

**Verdict: RULED OUT BY MEASUREMENT as a way to cut traffic without retraining.** No k that is meaningfully small
keeps quality: the first k with under +0.2 NLL (1,536) saves 22% of weight traffic. The measurement does not rule
out training a model with a narrow stream from the start (then the weights are different, not folded), which is
what this test cannot see. Next discriminating experiment: train SmolLM2-sized layers with a rank-512 stream and
measure NLL at equal training tokens.

## What to build in Omega, in order

Nothing here ships from this stream. Only experiments that passed produce kernels. PyTorch was the oracle.

1. **Gram accumulation, G = XᵀX and H = XᵀY in chunks (E1, PASSES for routing and classifier heads).** A
   symmetric rank-k update: bf16 features in, fp32 accumulate on the tensor cores, one pass over the features.
   Parts that exist: tensor-core GEMM generators `src/omega_blackwell_codegen_tc.c`, `_tc2.c`, `_gemm.c` and the
   launcher `src/omega_gpu_tcgemm_api.c` (omega branch `fix/tcgemm-nj`, not yet on omega main `ce42e62`); bf16
   CPU reference `omega_matmul_cpu_oracle_bf16` in `src/omega_blackwell_matmul.h` (main). Missing: the
   transposed-operand (AᵀA) variant, which is the same need as L294's backward GEMM dW = XᵀdY, so build it once.
2. **Linear solve (G + λI)W = H (E1).** A Cholesky factorisation and two triangular solves in binary64, on the CPU
   for widths up to a few thousand (2,049 here). Parts that exist: Cholesky `chol` in `src/estimation/est_kf.c`
   (main, a static function at line 48), and a binary64 ridge readout in `src/reservoir/omega_reservoir_readout.h` (branches
   `research/learning-search-584` and `fix/tcgemm-nj`, not on main). The solve took 0.356 s in PyTorch on the
   GPU (`e1-time.json`); a CPU time was not measured.
3. **Row-weighted Gram, XᵀDX (E1, yes/no task).** Exact Newton (IRLS) reached the trained head's AUC in 12 steps; each
   step is kernel 1 with a per-row scale d_i and kernel 2. Cheap to add once kernel 1 exists.
4. **Skinny low-rank adapter forward and backward (E2, LoRA).** x·Aᵀ then ·Bᵀ with rank 8, and its three
   backward products, using the same transposed-operand GEMM. L294 already scopes the backward GEMMs; add rank 8
   and rank 64 shapes (64 to 2,048 wide) to its acceptance shapes. PHM and FourierFT are **not** on the list.

Not built, with the reason: composition by structure constants (E3, nothing to compose), a folded between-layer
subspace (E4, 22% traffic for +0.14 NLL), the Fourier expansion (E2, 9.8x slower), and a ridge head for next-word
prediction (E1, 6.7 points short).

## Next discriminating experiments

- **E1 (PARTLY):** use the ridge solution as the starting point for cross-entropy training and count passes to the
  trained head's quality. Today AdamW needs 1 epoch from scratch to match ridge and 3 to reach the best head
  minus 0.5 point; if a ridge start cuts the second number, the solve earns a place as a warm start.
- **E2 (PHM PARTLY):** repeat on a task where LoRA rank 8 and rank 64 differ (here they tie), and compare at equal
  step time instead of equal parameter count.
- **E3/E4 (the surviving form of Drake's idea):** train small layers whose weights are constrained to a
  closed algebra (or whose stream is rank 512) from step zero, and measure NLL at equal training tokens against
  an unconstrained model of the same size. The trained-the-usual-way model has no such structure (E3, E4); a
  model built with it is untested.

## Sources

[1] https://arxiv.org/abs/2102.08597
    Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with 1/n Parameters (Zhang et al.), 2021-02-17, accessed 2026-10-20.
    "we propose parameterizing hypercomplex multiplications, allowing models to learn multiplication rules from data regardless of whether such rules are predefined."

[2] https://arxiv.org/abs/2405.03003
    Parameter-Efficient Fine-Tuning with Discrete Fourier Transform (Gao et al.), 2024-05-05, accessed 2026-10-20.
    "we introduce FourierFT, which treats $\Delta W$ as a matrix in the spatial domain and learns only a small fraction of its spectral coefficients. With the trained spectral coefficients, we implement the inverse discrete Fourier transform to recover $\Delta W$."

[3] https://arxiv.org/abs/2106.09685
    LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.), 2021-06-17, accessed 2026-10-20.
    "We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks."

[4] https://arxiv.org/abs/2106.04647
    Compacter: Efficient Low-Rank Hypercomplex Adapter Layers (Karimi Mahabadi et al.), 2021-06-08, accessed 2026-10-20.
    "Specifically, Compacter inserts task-specific weight matrices into a pretrained model's weights, which are computed efficiently as a sum of Kronecker products between shared "slow" weights and "fast" rank-one matrices defined per Compacter layer."

[5] https://arxiv.org/abs/2603.12977
    Exact Federated Continual Unlearning for Ridge Heads on Frozen Foundation Models (Quan, Wu, Montana), 2026-03-13, accessed 2026-10-20.
    "We study this problem in a practically relevant but under-explored regime: a frozen foundation model with a ridge-regression head. The exact optimum depends on the data only through two additive sufficient statistics"

[6] https://www.ism.ac.jp/editsec/aism/44/17.html
    Multinomial Logistic Regression Algorithm (Bohning, Ann. Inst. Statist. Math. 44), 1992 (received 1990-07-23), accessed 2026-10-20.
    "consists of replacing the second derivative matrix by a global lower bound in the Loewner ordering. This bound is used in the Newton-Raphson iteration instead of the Hessian matrix leading to a monotonically converging sequence of iterates."

[7] https://arxiv.org/abs/2110.14168
    Training Verifiers to Solve Math Word Problems (Cobbe et al.), GSM8K, 2021-10-27, accessed 2026-10-20.
    "we introduce GSM8K, a dataset of 8.5K high quality linguistically diverse grade school math word problems."

[8] https://arxiv.org/abs/2402.17762
    Massive Activations in Large Language Models (Sun et al.), 2024-02-27, accessed 2026-10-20.
    "very few activations exhibit significantly larger values than others (e.g., 100,000 times larger). We call them massive activations."

[9] https://arxiv.org/abs/2502.02737
    SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model (Ben Allal et al.), 2025-02-04, accessed 2026-10-20.
    "we overtrain SmolLM2 on ~11 trillion tokens of data using a multi-stage training process that mixes web text with specialized math, code, and instruction-following data."

[10] https://www.nvidia.com/en-us/products/workstations/dgx-spark/
    NVIDIA DGX Spark product page, page as served 2026-10-20, accessed 2026-10-20.
    "Memory Bandwidth 273 GB/s"

[L279] Internal: aien-dev/aien-train branch physics/gb10 at 001ec9a, docs/physics/gb10-training.md line 56 (measured by L151, omega doc/machine-ceiling@3c8ce239, evidence/MACHINE-CEILING-20261020/raw/gpu-tc-peak.txt).
    "| Main memory read, GPU | **245.1 GB/s** | measured (L151)"

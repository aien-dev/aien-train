# Structure constants and "let math deal with it": the source paste (2026-10-20)

This is the verbatim text Drake brought from a chat with another assistant (that assistant ran small checks in
Wolfram). It is kept exactly as pasted, unedited. Its toy claims (the golden-ratio closure, the exact ridge
example, the four-experiment table) are what the math-training stream (ledger L283) tests on a real model.
Results: [docs/math/STRUCTURE_CONSTANTS.md](../math/STRUCTURE_CONSTANTS.md).

---

**Your structure-constants idea gives us a concrete mathematical direction to test.** The key question is whether a large computation can be represented by a much smaller set of quantities that stays sufficient as the computation proceeds.

I tested two examples in Wolfram. One connects directly to your golden-ratio thought; the other can eliminate iterative training for a restricted kind of adaptation.

**1. Structure constants: compute with the coordinates of an operation.**

Suppose useful transformations can be expressed using a small collection of basis matrices:

\[
W=\sum_{i=1}^{r} a_iB_i.
\]

If their products stay within that collection,

\[
B_iB_j=\sum_k c_{ij}^{k}B_k,
\]

then the \(c_{ij}^{k}\) are **structure constants**: a multiplication table for the transformations.

Instead of composing large matrices directly, we can compose their coefficient vectors. This helps when the basis is compact, its multiplication rules are cheap, and applying the resulting transformation is also efficient. A general dense multiplication table could erase the savings.

Here is the golden-ratio connection I checked:

\[
Q=\begin{pmatrix}1&1\\1&0\end{pmatrix},
\qquad Q^2=Q+I.
\]

Wolfram verified that identity and returned eigenvalues \((1+\sqrt5)/2\) and \((1-\sqrt5)/2\). Consequently,

\[
(aI+bQ)(cI+dQ)
=(ac+bd)I+(ad+bc+bd)Q.
\]

**Any sequence of these linear transformations can be represented by just two coefficients.** That is an exact toy example of the “everything clicks into place” property you described.

The useful property is **closure under composition**. The appearance of the golden ratio does not establish an advantage for neural networks.

The obstacle is nonlinear computation. Wolfram also verified a counterexample where applying ReLU before versus after \(Q\) gives different results. We cannot collapse a transformer’s layers through its activations merely because its linear pieces have convenient algebra.

There is relevant research: *Parameterized Hypercomplex Multiplication* learns structured multiplication rules, while **FourierFT** represents fine-tuning updates through selected Fourier coefficients. These support investigating structured parameterizations, but fewer trainable parameters do not automatically mean less total arithmetic. [GitHub](https://github.com/astonzhang/Parameterization-of-Hypercomplex-Multiplications?utm_source=chatgpt.com)

**2. For fine-tuning, the stronger immediate possibility is sufficient statistics.**

Freeze a feature extractor and train a linear output head with a squared-error objective and regularization. Write its features as \(X\), targets as \(Y\), and trainable head as \(W\).

We can accumulate:

\[
G=X^\top X,\qquad H=X^\top Y,
\]

then solve:

\[
(G+\lambda I)W=H,\qquad \lambda>0.
\]

For this specific objective, that replaces repeated gradient-descent updates with **feature extraction, accumulation, and a linear solve**.

I checked a small exact example in Wolfram:

- Accumulating statistics in separate chunks matched processing all examples together.
- The solved weights were \(9/13\) and \(12/13\).
- The objective’s gradient at that solution was exactly zero.

This is close to what you mean by “let math deal with it”: **change the problem into one we can solve directly.** Research on ridge heads over frozen foundation-model features uses this same restricted setting. [arxiv.org](https://arxiv.org/abs/2603.12977?utm_source=chatgpt.com)

It still costs computation. The feature extraction may dominate; \(G\) requires quadratic storage in feature width; numerical conditioning matters. And this does **not** replace general transformer fine-tuning or cross-entropy training. It is especially worth testing for a router, classifier, or regression head whose existing features already contain the needed information.

**3. My deeper hypothesis for Omega: find a representation that remains small throughout the useful calculation.**

A compressed weight file is only part of the opportunity. If every operation expands it into large intermediates, the memory traffic returns.

I would test these possibilities in this order:

| Experiment | What would establish value |
|---|---|
| Frozen features + direct head solve | Meets the task’s quality target with less total training time |
| Structured adapter basis versus LoRA | Preserves quality while reducing actual memory traffic and runtime |
| Compact algebra with structure constants | Useful compositions stay compact, including the cost of applying them |
| Small invariant activation subspace | Intermediate states remain accurately representable without frequent expansion |

For the last two, we must measure where the representation stops being sufficient—particularly across nonlinearities—and how approximation errors accumulate.

**My strongest recommendation is to test direct head solving first, then compact structured adapters.** TURING can choose among the implementations that pass those tests. The mathematical opportunity is to discover which information the task actually needs, preserve that information in a small representation, and avoid repeatedly reconstructing everything else.

The Wolfram results establish the small algebraic examples. Whether Omega’s actual model admits a similarly useful structure remains an experiment, not a demonstrated speedup.

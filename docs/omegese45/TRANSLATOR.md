# Omegese #45 translator (en2goal, goal2en, rep2en)

Issue: [aien-dev/aien-train#45](https://github.com/aien-dev/aien-train/issues/45). The translator is the only place English exists in the product path. It is deterministic, rule-based `.om`. It loads no model and no checkpoint, and it is built and tested separately from the core (`tests/omegese45/translator/run-translator.sh` builds only the units listed below).

Status: IMPLEMENTED and EXERCISED on the GB10 Spark Grace CPU, cores 10-14, nice 19. Receipt: [`evidence/OMEGESE45/translator/`](../../evidence/OMEGESE45/translator/).

## Build units (concatenation, pinned omc bf54b19f, sha256 `62775d2a…3712b`)

| Binary | Units, in order |
| --- | --- |
| en2goal | `translate/otio.om translate/en2goal.om translate/otfrag.om translate/main-en2goal.om` |
| goal2en | `translate/otio.om vocab.om msg.om translate/otmsg.om translate/goal2en.om translate/otfrag.om translate/main-goal2en.om` |
| rep2en | `translate/otio.om vocab.om msg.om translate/otmsg.om translate/rep2en.om translate/otfrag.om translate/main-rep2en.om` |

All paths are under `native/omegese45/`. goal2en and rep2en validate every G and P line with the contract lane's `ocsetgl` and `ocvline` (`vocab.om`, `msg.om`, contract commit fa57fdf). The translator does not re-implement the R45 grammar. All translator names start with `ot`.

## Exit codes

| Code | Tool | Meaning | stdout |
| --- | --- | --- | --- |
| 0 | all | success | the G line or the English sentence |
| 2 | all | usage, or a stdin read or stdout write error | usage line |
| 20 EN_UNSUPPORTED | en2goal | no rule matches, empty or blank input, over 512 bytes, or a byte outside TAB and 0x20..0x7e | `EN_UNSUPPORTED` |
| 21 EN_RANGE | en2goal | a dim not in 1..4095, a seed not in 0..65535, or a malformed number | `EN_RANGE` |
| 22 EN_SHAPE | en2goal | the inner dims differ (k of the left matrix ≠ rows of the right) | `EN_SHAPE` |
| 23 EN_AMBIG | en2goal | more than one reading | `EN_AMBIG` |
| 30 G2E_CONTRADICT | rep2en | P contradicts G or the runtime facts; no success is printed | `G2E_CONTRADICT <what>` |
| 31 G2E_INVALID | goal2en, rep2en | an input is invalid | `G2E_INVALID <what>` |

Dims 257..4095 pass through en2goal: the core refuses them (code `bound`), not the translator.

## en2goal grammar

**Normalization** (`otnorm`):
- A-Z are lowercased.
- Space, TAB and `, ; : ? ! ( ) "` separate words.
- `.` separates words unless a digit is on both sides. In that case it stays inside the word, so `2.5` is a malformed number.
- An `x` next to a digit becomes a word of its own, so `4x8x2` reads as `4 x 8 x 2`.

**Numbers:** a word that starts with a digit, or with `+`/`-` followed by a digit, fills a number slot. The word must be 1..9 decimal digits; anything else is malformed and gives 21. Leading zeros are accepted.

**Rules:** this is the `text otrules` table in `en2goal.om`, verbatim. A rule must match the whole word sequence. `a/b` lists literal alternatives, and `?` marks the word as optional. `#m #k` are the left matrix dims, `#j #n` the right matrix dims, `#s` the seed and `#t` the second seed.

```
m ?please multiply a/an #m by #k matrix by/with/times a/an #j by #n matrix with seed #s
m ?please ?run/do/compute/submit ?a/an/one matmul/matmul_u32 ?job ?of ?size #m x #k x #n ?with seed #s
m goal for review one matmul_u32 job multiplying a/an #m by #k matrix by a/an #j by #n matrix with seed #s
p ?please ?run/do/compute/submit two matmuls/matmul/matmul_u32s/matmul_u32 ?jobs ?of ?size #m x #k x #n ?with seeds #s and #t
p matmuls #m x #k x #n ?with seeds #s ?and #t
p ?please multiply a/an #m by #k matrix by/with/times a/an #j by #n matrix twice with seeds #s and #t
p goal for review two matmul_u32 jobs each multiplying a/an #m by #k matrix by a/an #j by #n matrix with seeds #s and #t in that order
x ?please ?run/do/compute/submit ?a/an/one matmul/matmul_u32 ?job ?of ?size #m x #k x #n ?with seeds #s ?and #t
x ?please ?run/do/compute/submit two matmuls/matmul/matmul_u32s/matmul_u32 ?jobs ?of ?size #m x #k x #n ?with seed #s
x ?please multiply a/an #m by #k matrix by/with/times a/an #j by #n matrix with seeds #s and #t
x ?please ?run/do/compute/submit ?a/an/one matmul/matmul_u32 ?job ?of ?size #m x #k x #n ?with seed #s and/or ?seed #t
x ?please ?run/do/compute/submit ?a/an/one matmul/matmul_u32 ?job ?of ?size #m x #k x #n ?with seed #s seed #t
```

**Decision:**
1. If any `x` rule matches, or two `m`/`p` rules give different readings, the result is 23.
2. If no rule matches, the result is 20.
3. Otherwise a range or malformed-number failure gives 21.
4. Otherwise `j ≠ k` gives 22.
5. Otherwise the G line is printed: `goal mm ushort 0xMMM ushort 0xKKK ushort 0xNNN ulong 0xSSSS end` or `goal pair … ulong 0xSSSS ulong 0xTTTT end`, in lowercase hex.

## goal2en templates (fixed)

- mm: `Goal for review: one matmul_u32 job multiplying a {m} by {k} matrix by a {k} by {n} matrix with seed {s}.`
- pair: `Goal for review: two matmul_u32 jobs, each multiplying a {m} by {k} matrix by a {k} by {n} matrix, with seeds {s1} and {s2} in that order.`
- Invalid G: `G2E_INVALID goal <reason>`, where the reason is the contract reason word (for example `grammar`); exit 31.

The article is always `a`. Every value is decimal. Both sentences are in the en2goal grammar, and the suite checks that re-parsing them gives the same G.

## rep2en

**CLI:** `rep2en FACTS_TSV`. stdin carries the G line, then the P line.

**Facts format:** facts.tsv has 13 TAB-separated columns: job attempt m k n seed exit class code dag_id mjr1_sha256 mjr1_outcome verifier.
- A first row whose first field is `job` is a header.
- A file without data rows means nothing was dispatched.

**Validity:** any of the following gives `G2E_INVALID facts` with exit 31:
- a row that is not 13 fields;
- job, attempt, m, k, n, seed or exit that is not 1..9 decimal digits;
- a class not in ok/refused/fail;
- an ok row whose exit is not 0, whose code is not `-`, or whose dag_id/mjr1_sha256 is not 64 lowercase hex;
- a refused/fail row with exit 0, or with a code that is not one of replay budget worker transport receipt recomp verdict io;
- an empty interior line;
- more than 64 rows or 65536 bytes;
- an unreadable file.

**Templates** (fixed; values come only from G and facts.tsv):

| Case | Sentence |
| --- | --- |
| ok mm | `Completed and verified: one matmul_u32 job multiplying a {m} by {k} matrix by a {k} by {n} matrix with seed {s}; job 1 dag_id {dag1} mjr1_sha256 {sha1}.` |
| ok pair | `Completed and verified: two matmul_u32 jobs, each multiplying a {m} by {k} matrix by a {k} by {n} matrix, with seeds {s1} and {s2}; job 1 dag_id {dag1} mjr1_sha256 {sha1}; job 2 dag_id {dag2} mjr1_sha256 {sha2}.` |
| refused bound | `Not completed: refused before dispatch (code bound): the goal asks for a {m} by {k} by {n} matmul and a dimension is outside the admitted range 1 to 256; nothing was dispatched.` |
| refused C | `Not completed: job {j} was refused (code {C}): {plain}.` |
| fail C | `Not completed: job {j} failed (code {C}): {plain}.` |

{dag}/{sha} are the dag_id/mjr1_sha256 of that job's last facts row; {j} is the job of the last facts row.

**Plain words:**

| Code | Plain words |
| --- | --- |
| replay | the worker refused a replay of an already executed job |
| budget | the job was refused for budget or fuel |
| worker | the worker refused the job |
| transport | the connection to the worker failed |
| receipt | the verifier refused the worker receipt |
| recomp | the verifier recompute failed |
| verdict | the recomputed result did not match the worker result |
| io | a local input or output error occurred |

**Checks, in order:**
1. An invalid G gives `G2E_INVALID goal <reason>`, an invalid P `G2E_INVALID report <reason>`, malformed stdin `G2E_INVALID input` and an invalid facts file `G2E_INVALID facts`, all with exit 31.
2. P ok whose kind or operands differ from G gives `G2E_CONTRADICT operands`.
3. Every facts row:
   - a job outside 1..(1 or 2) gives `G2E_CONTRADICT jobs`;
   - m/k/n differing from G, or a seed differing from G's seed for that job, gives `G2E_CONTRADICT operands`.
4. P ok:
   - every job of G needs a last row with class ok (hence exit 0), otherwise `jobs` (missing) or `class`;
   - if the check passes, the success template is printed.
5. P refused bound: facts must have no rows (`bound` otherwise) and a G dim must be outside 1..256 (`bound` otherwise).
6. P refused/fail C:
   - facts must have rows (`dispatch` otherwise);
   - the last row must have the same class (`class` otherwise) and the same code (`code` otherwise).

A retried job (a fail row, then an ok row for the same job) counts as ok, because the last row decides.

## Measured accuracy and cost (receipt: `evidence/OMEGESE45/translator/suite.txt`)

- **Dev cases** (`tests/omegese45/translator/cases.tsv`; the four required phrasings, boundaries dims 0, 1, 256, 257, 4095, 4096, seeds 65535/65536, and every refusal code): exact 18 of 18.
- **Held-out set** (`tests/omegese45/translator/heldout.tsv`, committed in faeb7ad before the parser existed and never edited after): exact 39, refused differently 6, wrong 0, total 45.
  - Five supported-in-intent phrasings are refused with 20: "using seed", "two 4 by 4 matrices", "run 2 matmuls", a trailing "please", "4 by 8 by 2".
  - "seed 0x10" gets 20 where 21 was expected.
  - No G was ever emitted that differs from the hand-written one.
- **Stability:** 27 distinct G lines; goal2en then en2goal reproduces each one.
- **rep2en:**
  - 6 English cases: ok mm, ok pair, retry-then-ok, bound, refused replay, fail transport;
  - 10 contradiction cases (exit 30);
  - 3 invalid cases (exit 31).
- **Cost per call** (`/usr/bin/time -v`, GB10 Spark Grace CPU, cores 10-14, nice 19): wall 0:00.00 (below the tool's 10 ms resolution) and max RSS 192 KB for each of en2goal, goal2en and rep2en.
- **Negative control:** a mutant en2goal with k and n swapped in the emitted G gives 5 wrong of 18 dev cases, and the suite catches it (`MUTANT_CAUGHT`).

## Not covered (NOT_RUN or out of scope)

- The translator is not trained; there is no model in it by design.
- Phrasings outside the rule table are refused (20), not guessed.
- The demo-side end-to-end use of rep2en with real aienet0-run facts belongs to the demo lane's receipts.

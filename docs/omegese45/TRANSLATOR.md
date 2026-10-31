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
| 21 EN_RANGE | en2goal | a dim above 4095, a seed above 65535, or a malformed number (interface v1.2 §4; dim 0 is accepted) | `EN_RANGE` |
| 22 EN_SHAPE | en2goal | the inner dims differ (k of the left matrix ≠ rows of the right) | `EN_SHAPE` |
| 23 EN_AMBIG | en2goal | more than one reading | `EN_AMBIG` |
| 30 G2E_CONTRADICT | rep2en | P contradicts G or the runtime facts; no success is printed | `G2E_CONTRADICT <what>` |
| 31 G2E_INVALID | goal2en, rep2en | an input is invalid | `G2E_INVALID <what>` |

Dims 0 and 257..4095 pass through en2goal: D can encode 0x000..0xfff, and the core refuses those dims (code `bound`), not the translator (interface v1.2 §4, review finding T3).

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

**Facts format** (interface v1.2 §5): facts.tsv has 13 TAB-separated columns: job attempt m k n seed exit class code dag_id mjr1_sha256 mjr1_outcome verifier.
- The first line must be exactly the header `job attempt m k n seed exit class code dag_id mjr1_sha256 mjr1_outcome verifier` (TAB-separated, LF).
- A file with only the header means nothing was dispatched.

**Validity:** any of the following gives `G2E_INVALID facts` with exit 31:
- a missing or different header line;
- a row that is not 13 fields;
- job, attempt, m, k, n, seed or exit that is not 1..9 decimal digits;
- a class not in ok/refused/fail;
- an ok row whose exit is not 0, whose code is not `-`, whose dag_id/mjr1_sha256 is not 64 lowercase hex, whose mjr1_outcome is not `1` (MJR1 completed), or whose verifier is not 66 lowercase hex starting `02` or `03` (P-256 compressed key);
- a refused/fail row whose code is `-` or not one of replay budget worker transport receipt recomp verdict io, whose mjr1_outcome or verifier is not `-`, or whose dag_id or mjr1_sha256 is neither `-` nor 64 lowercase hex (a non-ok row may carry a real dag_id, and a receipt digest when a receipt exists);
- a class/code/exit triple outside the aienet0-run map: `refused` replay, budget or worker with exit 11; `fail` transport 10, receipt 12, recomp 13, verdict 14, io 15 (an ok row has exit 0);
- an illegal history: jobs must appear in order starting at job 1, each starting with attempt 1; a second row for the same job is legal only as attempt 2 right after an attempt-1 `fail transport`; a row for the next job is legal only after the previous job's last row was ok. So no row follows a job's ok row, every attempt is 1 or 2 and increases within a job, every non-ok row other than one retried transport failure is the last row, and in a refused/fail report every earlier job ended ok;
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
1. An invalid G gives `G2E_INVALID goal <reason>`, an invalid P `G2E_INVALID report <reason>`, malformed stdin `G2E_INVALID input` and an invalid facts file (every rule under **Validity**, header and history included) `G2E_INVALID facts`, all with exit 31.
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

A retried job (an attempt-1 `fail transport` row, then an attempt-2 row for the same job) is decided by its attempt-2 row.

## Measured accuracy and cost (receipt: `evidence/OMEGESE45/translator/suite.txt`)

- **Dev cases** (`tests/omegese45/translator/cases.tsv`; the four required phrasings, boundaries dims 0 (accepted, G with `0x000`), 1, 256, 257, 4095, 4096, seeds 65535/65536, and every refusal code): exact 18 of 18.
- **Held-out set** (`tests/omegese45/translator/heldout.tsv`, committed in faeb7ad before the parser existed and never edited after):
  - Raw, against the committed file (primary number): exact 38, refused differently 6, wrong 1, total 45 (84.44% / 13.33% / 2.22%; WolframLanguageEvaluator: `N[{38/45, 6/45, 1/45}*100, 4]`).
  - The 1 wrong row is row 12 ("a 0 by 6 matrix"), written as `refuse 21` under the old rule. Interface v1.2 §4 then pinned that dim 0 is accepted and refused by the core, so en2goal now emits `goal mm ushort 0x000 ushort 0x006 ushort 0x008 ulong 0x0003 end`. The dated erratum, with its reason, is in [`heldout-errata.tsv`](../../tests/omegese45/translator/heldout-errata.tsv); the suite checks that each erratum's original equals the committed expectation.
  - With the errata applied: exact 39, refused differently 6, wrong 0 (86.67% exact). PASS requires raw wrong ≤ the number of errata rows and wrong 0 with the errata, and the receipt prints both numbers and every erratum.
  - Five supported-in-intent phrasings are refused with 20: "using seed", "two 4 by 4 matrices", "run 2 matmuls", a trailing "please", "4 by 8 by 2".
  - "seed 0x10" gets 20 where 21 was expected.
- **Stability:** 29 distinct G lines (dev plus errata-corrected held-out); goal2en then en2goal reproduces each one.
- **rep2en:**
  - 9 English cases: ok mm, ok pair, retry-then-ok, retry-then-fail, bound, refused replay, fail transport, pair with job 2 refused, a fail row carrying dag_id and digest;
  - 10 contradiction cases (exit 30), including `x-bound-dispatched`, whose facts row matches G (300 2 2, seed 1) so it reaches the `bound` check (`G2E_CONTRADICT bound`; review finding T5);
  - 30 invalid cases (exit 31): invalid G, invalid P and an ok row without digests, then header missing/wrong and one case per v1.2 facts probe of review findings T1 (8: ok outcome, ok verifier forms, non-ok `-` and dag form), T2 (7: exit map, including the old fixture exits 3 and 4) and T4 (10: history and attempts).
- **Cost per call** (`/usr/bin/time -v`, GB10 Spark Grace CPU, cores 10-14, nice 19): wall 0:00.00 (below the tool's 10 ms resolution) and max RSS 192 KB for each of en2goal, goal2en and rep2en.
- **Negative controls:**
  - A mutant en2goal with k and n swapped in the emitted G gives 5 wrong of 18 dev cases, and the suite catches it (`MUTANT_CAUGHT`).
  - Ten rep2en mutants, each disabling one facts check (header, ok outcome, ok verifier, non-ok `-` pair, non-ok dag/digest form, code→class, code→exit, same-job history, next-job history, previous-job-ok history), are each caught by at least one fixture (`REPMUTANT … CAUGHT`).
  - The suite refuses to build with any omc whose sha256 differs from the pin (`OMC_PIN_MISMATCH`, exit 1; receipt `evidence/OMEGESE45/translator/omc-pin-refusal.txt`, review finding T6).

## Not covered (NOT_RUN or out of scope)

- The translator is not trained; there is no model in it by design.
- Phrasings outside the rule table are refused (20), not guessed.
- The demo-side end-to-end use of rep2en with real aienet0-run facts belongs to the demo lane's receipts.

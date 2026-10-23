# PACK5 v1: bounded public-example packing

This implements one component of aien-train5 and train3: integer token packing,
next-token labels, supervised-target masks and causal attention boundaries.
It consumes existing native public-example admission, retaining those owners.
No tokenizer, dataset reader, numerical tensor engine, model training loop,
worker runtime, durable cursor or checkpoint format is implemented here.

## Frozen dependencies and caller

Compose these exact source files, in order, followed by a caller:

1. Omega L775 commit `84d03c5523f39ca1045d1c313c5f62597779462b`,
   `src/learning/benchmark/core.om`, SHA256
   `a5a2e74c815f9ca6a90dbdfbfc7ecc343f0362c824064f2058e9d1d7c119c40c`.
2. Omega L776 commit `d8d14c47647e49774f64b58e3c138335656f175d`,
   `src/learning/curriculum/admit.om`, SHA256
   `533638a9f128853d696644de2bbf7b1859797b170a7e32e54c112ab94c3dadb1`.
3. `native/ecosystem5/pack.om`.

L776 published head is `1a14972c3b2d7494ba445df79967e7963359230e`.
The native compiler SHA256 is
`191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`,
independently rebuilt from omega-bridge
`0fc7d8662cb799e7c206b17c18ea4c3ec2b8fea9` by the compiler qualification lane.
The compilation target is static AArch64 Linux. No foreign runtime is linked.

Callers first use existing `cbegin`, `csubmit` and `cfinish`. Only successful
completed admission with `CState[2]=1` and `LS[18]=1` may be consumed. `CGood`
is the accepted index list in submission order; `CFreeze` is its manifest
snapshot. Quarantined records never enter a packed sequence. Live `LT` changes
after admission cannot alter the consumed frozen split.

These are trusted, single-thread, in-process arrays. The caller must preserve
the owner modules' state and supply measured identities and tokenization.
Public mutable arrays do not authenticate an attacker or prevent concurrent
mutation. Checking a supplied digest does not prove its contents. Existing
custody, tokenizer and execution owners retain those duties.

## Inputs

`DPCfg[4]` holds schema1, vocabulary size1..4294967296, split1(train) or2(dev),
and padding token ID less than vocabulary size. Every accepted record must
belong to the requested split. Mixed accepted train/dev batches are refused.
Protected adaptive/final records are already refused by L775.

`DPPin[16]` holds expected tokenizer SHA256 words0..3, observed tokenizer words4..7,
expected template SHA256 words8..11 and observed template words12..15. Every
identity has four big-endian u64 words representing all32 digest bytes. Both
expected identities must be nonzero and must match all observed words. They
may equal one another; no cryptographic independence is inferred. There is no
identity measurement or hashing implementation in this module.

For accepted task slot `t`, `DPLen[t]` is1..256 and `DPPrompt[t]` is0..DPLen[t].
`DPToken[t*256+j]` is token ID `j`, strictly below vocabulary size. Token zero
is legal. Prompt length counts prefix tokens excluded as prediction targets.
All-prompt examples are valid and contribute no supervised targets. Records
of length1 are valid and contribute no next-token targets. Empty records are
refused. Length/prompt entries for all quarantined or absent slots must be0;
unused token-array cells are ignored. Maximum64 admitted records and256 total
packed token positions. Oversized batches are refused, never truncated.

## Results and failures

`dppack()->ulong` validates the entire input before changing payload arrays.
It clears `DPState[ready,count,records]` at entry. Failure leaves every payload
cell unchanged but invalid, and `dpattend` returns false until a later success.
Do not consume payloads without `DPState[0]=1`.

On success, accepted records are copied in `CGood` order. `DPOut[256]` contains
token IDs; `DPTask[256]` holds original task slots; `DPPos[256]` holds positions
within each record. `DPState[1]` is the actual packed count; `DPState[2]` counts
records. An all-quarantined completed admission yields a valid empty batch.

For position `p` within a record, `DPLabel[p]` is that record's next token when
one exists. `DPMask[p]=1` exactly when the next token exists and its within-record
index is at least the prompt length. Thus the last prompt token may predict
the first completion token, and no final record token predicts the next record.
Prompt-only labels may remain populated but must be ignored when mask0. This
is an integer supervision mask, not a loss computation.

Unused output positions contain the configured padding token in output and
label, mask0, task sentinel18446744073709551615 and position0. Physical padding
IDs inside a real record are ordinary tokens; only explicit record length
defines padding. No special-token semantic is guessed.

`dpattend(q,k)->bool` returns true only for valid packed positions in the same
record with k<=q. Padding and out-of-range indices, including maximumu64,
return false. Helpers `dpvalidate`, `dpinput` and `dpcopy` are internal and do
not independently provide a safe publication API; call `dppack`.

| Code | Meaning |
| --- | --- |
| 0 | Valid packed batch |
| 70 | Admission incomplete, invalid or active |
| 71 | Schema, vocabulary, partition or padding configuration invalid |
| 72 | Zero or mismatched supplied tokenizer/template identity |
| 73 | Invalid/duplicate accepted index, inconsistent membership, or tokens declared for nonaccepted slot |
| 74 | Invalid record or prompt length |
| 75 | Token ID outside the declared vocabulary |
| 76 | Total packed capacity exceeded |
| 77 | Accepted record belongs to another partition |

## Qualification and remaining work

The native author suite drives actual owner admission and passes282622 checks.
Its independent coordinate oracle enumerates196 two-record layouts and their
prompt boundaries, verifies targets and padding, and compares every attention
pair with separately defined partition intervals and a lower triangle. Direct
controls cover readiness, identities, malformed records, mixed partitions,
quarantine, reverse submission order, full capacity and64 records. Three
compiled mutants invert supervision, remove record attention isolation and
bypass partition refusal; they fail checks5163,19 and263254 respectively.

Independent final-source review of088258d passed84378 additional native checks,
reproduced the original282622-check binary, and caught all three compiled
mutants with a separate caller. The full report is preserved under
`evidence/ECOSYSTEM5/independent-review/`. This slice
does not establish tokenizer parity, actual data-source authentication,
shuffling/resume, durable checkpoint recovery, model quality or physical GPU
results. All these remain assigned to train3/5/6 and their existing owners.
No prior first-party implementation or runtime dependency can be retired from
this additive component alone. The complete replacement issues stay open.

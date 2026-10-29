# Omegese #45 contract: R45, messages, binding, MJR1

Status: IMPLEMENTED and EXERCISED on the GB10 Spark Grace CPU host (cores 10-14, nice 19). Receipt:
[`evidence/OMEGESE45/contract/RECEIPT.md`](../../evidence/OMEGESE45/contract/RECEIPT.md) (fresh-clone run; commit named there). Source of truth for the shared
interface is the parent-owned interface v1.2 for [aien-train#45](https://github.com/aien-dev/aien-train/issues/45). This file restates it with
pins and adds what the code fixes. It changes no token, id, grammar rule or exit code.

## 1. Vocabulary pin

- `docs/omegese45/vocab.tsv`: 41 rows, 2188 bytes, `id TAB word TAB class TAB provenance LF`, no header.
- **sha256 `c1d7e15466e6585a776314c5c05f35466b2697d9631ad2e11c7918a434a4c18d`.** Checkpoints and datasets pin this value (exit 52 CKPT_VOCAB on mismatch, pipeline lane).
- `native/omegese45/vocab.om` embeds the same bytes (`text ocvtsv`); driver mode `T` prints them and the suite compares with `cmp`.

R45 is a **fixed lexical vocabulary, not a trained tokenizer**: no merges, no learned statistics, no byte fallback. A
word maps to exactly one id or is rejected. Storage: the table is 2188 bytes; an id needs
`Ceiling[Log2[41]]` = 6 bits (Wolfram: `N[Log2[41],6]` → 5.35755, `Ceiling[Log2[41]]` → 6), stored as one byte per id, so
the longest message (31 ids) is 31 bytes and MAXLEN is 176 bytes in id form. The DFA table is 256×41 = 10496 bytes
(Wolfram: `256*41` → 10496).

## 2. Restricted profile

The only surface is single-space-separated lowercase ASCII words from R45 plus `0x` hex words (`0x` followed by 1..4
digits per the grammar). No free prose, no comment channel, no `note`, no quotes, no numbers in decimal. Rejected bytes
and shapes (driver `V`, exit 40 `INVALID <byte pos> <reason>`): `empty`, `space` (leading, trailing, double), `tab`, `cr`,
`nonascii`, `upper`, `hexcase` (`0X`), `ctrl` (any other byte < 32, including an embedded LF), `unknown`, `long` (> 1024
bytes or > 176 ids), `grammar`, `noend`, `trailing`, `kind`. In id form (driver `D`): `badid` (> 40), `digit` (digit id
not after `0x` or a digit). `<pad>` (id 0) round-trips through the codec but is illegal in every message.

## 3. Grammar (interface §3, unchanged)

```
H := 0|1|…|9|a|…|f
D := ushort 0x H H H          5 ids
S := ulong 0x H H H H         6 ids
C := bound|replay|budget|worker|transport|receipt|recomp|verdict|io     C' := C without bound
G := goal mm D D D S end (24) | goal pair D D D S S end (30)
A := act job mm D D D S end (25) | act refuse bound end (4)
R := res ok end (3) | res refused C' end (4) | res fail C' end (4)
P := rep ok mm D D D S end (25) | rep ok pair D D D S S end (31) | rep refused C end (4) | rep fail C end (4)
```

`msg.om` builds this as a trie DFA (`ocdinit`): state 0 dead, state 1 accept (after `end`, every id dead). Start kinds:
1 G, 2 A, 3 R, 4 P, and **5 "model message"** (`act` or `rep` first, then the A or P automaton). Kind 5 extends the
interface's 1..4 start domain as §3 "grammar-guided decoder" requires; it adds no rule.

## 4. Episode format `oe45 v1` and length facts

First line `oe45 v1`, then one message per line in causal order, final LF; sequence `G (A R?)+ P`; `act refuse` gets no R.

Lengths (Wolfram: `d=5; s=6; {gmm=2+3d+s+1, gpair=2+3d+2s+1, ajob=3+3d+s+1, aref=4, pokmm=3+3d+s+1, pokpair=3+3d+2s+1, naive=gpair+4(ajob+4)+pokpair, maxok=gpair+2(ajob+4+ajob+3)+pokpair, maxfail=gpair+(ajob+4+ajob+3)+(ajob+4+ajob+4)+4}`
→ `{24, 30, 25, 4, 25, 31, 177, 175, 149}`; check `30+2*(3+3*d+s+1+4+3+3*d+s+1+3)+31` → 175):

- **Longest attained episode: 175 ids** — `pair`, each job `res fail transport end` then `res ok end`, then P ok pair.
- **MAXLEN = 176** (`ocmaxi`) is a bound with one id of headroom, not an attained length. Interface v1.1 gave 176 as the
  longest episode by pairing a failed job 2 with a 31-id P; a failed job ends in a 4-id P (total 149). Interface v1.2 §3
  item 3 corrects the longest episode to 175 (contract issue 1, resolved).
- Longest single message 31 ids (P ok pair); decode budget 31 (pipeline exit 53).

## 5. Reference policy

As interface §3 (bound refusal; per-job dispatch; one identical retry after the first `fail transport`; otherwise stop
and mirror R's class and code; P ok copies G's operands). The contract code does not implement the policy; it only
checks messages, binding and receipts.

## 6. Exit codes owned here

| Code | Meaning | stdout |
|---|---|---|
| 0 | ok | mode output |
| 2 | usage (bad mode, malformed id list) | — |
| 40 MSG_INVALID | validator | `INVALID <pos> <reason>` (byte pos for text, id index for `D`) |
| 41 BIND_MISMATCH | binding | `BIND_MISMATCH notjob\|jobindex\|dims\|seed` |
| 42 BIND_BOUND | binding (a dim outside 0x001..0x100) | `BIND_BOUND` |
| 43 MJR1_INVALID | MJR1 parse | `MJR1_INVALID <pos> <reason>` |

Binding order (`ocbcore`): not `act job` → 41 notjob; job index not 1 (`mm`) or 1..2 (`pair`) → 41 jobindex; dims ≠ G →
41 dims; seed ≠ seed of job → 41 seed; else any dim outside 1..256 → 42; else 0. Mismatch is reported before bound.
Lowering (`oclower`, driver `L`): `LOWER m k n seed` in decimal; `act refuse` → 41 notjob.
Exit 43 covers **every static rule of `decode_receipt`** plus `io` (cannot open/read), a superset of the interface's
"wrong length or unknown outcome" (contract issue 3).

## 7. MJR1 field table (aien-protocols-mv `e705fdc9`)

Pins: `crates/aien-microjob/src/lib.rs` L34 `RECEIPT_MAGIC`, L36 `VERSION`, L38 `RECEIPT_LEN` 404, L40 `RECEIPT_SIGNED_LEN` 340,
L103-109 `outcome` 1..5, L887-928 `decode_receipt`; `specs/microjob/SPEC.md` L162-190 (§5 table).

Each row states the condition a valid receipt **must satisfy**; a receipt that violates it is refused with the reason
word and reported position in the last column (exit 43 `MJR1_INVALID <pos> <reason>`). Rules are checked top to bottom.

| Bytes | Field | Required (valid receipt) | Violation → reason, reported pos |
|---|---|---|---|
| len | — | exactly 404 bytes | any other length → `length`, pos = actual length |
| 0..4 | magic | `MJR1` | otherwise → `magic` 0 |
| 4 | version | 1 | otherwise → `version` 4 |
| 7, 84..88, 337..340 | reserved0/1/2 | all zero | any nonzero byte → `reserved` 7 / 84 / 337 |
| 5 | outcome | 1..5 | 0 or 6..255 → `outcome` 5 |
| 6 | attempt | ≠ 0 | 0 → `attempt` 6 |
| 304 | signer_key prefix | 0x02 or 0x03 | otherwise → `signer` 304 |
| 176..208 | source_hash | not all zero | all zero → `zerohash` 176 |
| 240..272 | policy_hash | not all zero | all zero → `zerohash` 240 |
| 144..176 | output_digest | not all zero if COMPLETED (1); all zero otherwise | zero on COMPLETED, or nonzero on 2..5 → `output` 144 |
| 272..304 | verifier_verdict | not all zero if COMPLETED; any value otherwise | all zero on COMPLETED → `verdict` 272 |
| 8..40 | dag_digest | — | printed |
| 40..56 | job_id | — | printed |
| 72..76 | generation (u32 BE) | — | printed |
| 340..404 | signature | not checked (as `decode_receipt`; aienet0-run verifies) | — |

Driver `J` prints `MJR1 <outcome word> <attempt> <generation> <dag hex> <job_id hex> <output hex>`. The real receipt
fixture `tests/omegese45/contract/fixtures/aadc3775…94cb.mjr1` (404 bytes, sha256
`3306d180de1885a2759b324618d11eca523ef4d7aa22c4144813d71187014855`) is copied from aien-protocols-mv `e705fdc9`
`evidence/aienet0-20261028/exercised/artifacts/` (last changed in `b356717d`).

## 8. Vocabulary provenance against omega-mv `bf54b19f`

Checked with omc bf54b19f (sha256 `62775d2a…712b`) by compiling each word as a fn name and as a local, and by reading
`fn kw` (`seed/omc/omc.om` L810; the interface says L808, which is the preceding `note` line — issue 2).

| ids | Words | Finding |
|---|---|---|
| 0 | `<pad>` | reserved, not an Omega word |
| 1-4, 6-12, 14-21 | goal act res rep mm pair job refuse ok refused fail replay budget worker transport receipt recomp verdict io | profile words; none is an Omega keyword (each compiles as a name; `transport` is a 9-byte name → error 13, a length limit, not a keyword) |
| 5 | `end` | Omega keyword, `fn kw` omc.om:1099 (`6581861`) — holds |
| 13 | `bound` | profile code **and** Omega keyword, omc.om:1111 (`431349919586`) — flagged, id unchanged (issue 4) |
| 22 | `ushort` | Omega type keyword omc.om:973, OMEGESE.md:114 — holds |
| 23 | `ulong` | Omega type keyword omc.om:985, OMEGESE.md:114 — holds |
| 24-40 | `0x`, `0`..`f` | Omega hex literals, OMEGESE.md:691-693 — hold **only in profile OM-2** (OMEGESE.md:668 §9.13); an OM-1 unit gives error 6 (issue 5) |

## 9. Build and API

```sh
cat native/omegese45/vocab.om native/omegese45/msg.om [native/omegese45/mjr1.om] <your main.om> | omc > out.elf
```

Order is fixed: `vocab.om` first (globals, codec), `msg.om` second (DFA, binding), `mjr1.om` optional. Names starting `oc`
are reserved for this lane. Call `ocvinit` and `ocdinit` once before use (the driver does).

Globals: `ocin uchar 4096` text in; `ocids ulong 256` ids; `ocipos ulong 256` byte offset per id; `ocout uchar 4096` text
out; `ocerr ulong 2` = [pos, reason]; `ocnids`; `ocmk uchar 41` mask; `ocgv` G operands; `oclv ulong 4` = m k n seed;
`ocrb uchar 512`, `ocpath uchar 4096`, `ocmv ulong 3` = outcome attempt generation.

| fn | returns |
|---|---|
| `ocenc n ulong to long` | id count, or −reason |
| `ocdec n ulong to long` | byte count, or −reason |
| `ocstart kind ulong to ulong`, `ocstep st ulong id ulong to ulong` | state (0 dead, 1 accept) |
| `oclegal st ulong id ulong to bool`, `ocfinal st ulong to bool` | |
| `ocmask st ulong to ulong` | legal count; fills `ocmk` |
| `ocvids n ulong kind ulong to long`, `ocvline n ulong kind ulong to long` | kind 1..4, or −reason (kind 0 infers) |
| `ocsetg n ulong to long`, `ocsetgl n ulong to long` | 0 or −reason |
| `ocbind n ulong job ulong to long`, `ocbindl n ulong job ulong to long` | 0, 41, 42, or −reason |
| `oclower n ulong to long`, `oclowerl n ulong to long` | 0 (`oclv`), 41, or −reason |
| `ocmjr n ulong to long`, `ocmjrf plen ulong to long` | outcome 1..5, or −reason |

Reason codes: 1 empty, 2 space, 3 tab, 4 cr, 5 nonascii, 6 upper, 7 hexcase, 8 ctrl, 9 unknown, 10 long, 11 badid,
12 digit, 13 grammar, 14 noend, 15 trailing, 16 kind, 17 notjob, 18 jobindex, 19 dims, 20 seed. MJR1: 1 length,
2 magic, 3 version, 4 reserved, 5 outcome, 6 attempt, 7 signer, 8 zerohash, 9 output, 10 verdict, 11 io.

Driver (`tests/omegese45/contract/main.om`), stdin `<mode> LF <payload>`: `V` line, `N` kind-letter LF ids, `B` G LF job
LF A, `L` A, `E` line, `D` ids, `T`, `J` path.

## 10. Suite

`OMC=/tmp/om-main/seed/out/omc OUT=<dir> sh tests/omegese45/contract/run-contract.sh` → last line
`OE45_CONTRACT_PASS`. It checks the T table by `cmp`; every id round trip; binding and bound; lowering; the real receipt,
19 refusals built from it by byte edits (including `zerohash` 240 and `verdict` 272) and 3 accepted edits (refused with
zero output and zero verdict; attempt 2 with generation `01 02 03 04` → 16909060, Wolfram `FromDigits[{1,2,3,4},256]`;
signer prefix 0x03); 140 generated valid messages (7 dims × 4 seeds × 5 forms; edges 0x000/0x001/0x100/0x101/0xfff,
seeds 0x0000/0xffff) plus 39 fixed ones, of all four kinds, each with its id count, E→D round trip and every prefix mask
(next id legal, FINAL only after `end`, empty mask after `end`); full N↔V agreement for all 41 ids at every prefix of 11
messages; 32 rejections; dead-state and start masks.

Negative control: 21 one-line mutants built by the script, each of which must fail first at the named check:

| Tag | Unit | Mutation | Killed by |
|---|---|---|---|
| m1 | msg.om | DFA: R `refused` admits the C set (`bound`) | `rej-R-bound` |
| m2 | msg.om | DFA: shared digit transition omits `f` | `bind-bound-101` |
| b1 | msg.om | binding 41: seed always compared with job 1 | `bind-ok2` |
| b2 | msg.om | binding 42: upper admission bound 257 | `bind-bound-101` |
| b3 | msg.om | binding 42: lower admission bound 0 | `bind-bound-000` |
| b4 | msg.om | binding 41: k not compared with G | `bind-dims` |
| b5 | msg.om | binding 41: `act refuse` not reported `notjob` | `bind-notjob` |
| b6 | msg.om | binding 41: job 2 admitted for `mm` | `bind-mm-job2` |
| c1 | vocab.om | codec: hex digit `f` not encoded | `rt40-e` |
| c2 | vocab.om | codec: digit id decoded after a word | `rej-dec-digit` |
| c3 | vocab.om | codec: `0X` reported `upper`, not `hexcase` | `rej-0X` |
| c4 | vocab.om | codec: 177 ids admitted | `rej-overlong-ids` |
| c5 | vocab.om | codec: id 23 (`ulong`) not looked up | `rt23-e` |
| j1 | mjr1.om | MJR1: zero verdict admitted on COMPLETED | `mjr1-zero-verdict` |
| j2 | mjr1.om | MJR1: zero policy_hash admitted | `mjr1-zero-policy` |
| j3 | mjr1.om | MJR1: zero verdict refused when not COMPLETED | `mjr1-refused-zero-verdict` |
| j4 | mjr1.om | MJR1: attempt 0 admitted | `mjr1-attempt-6-0` |
| j5 | mjr1.om | MJR1: outcome 6 admitted | `mjr1-outcome-5-6` |
| j6 | mjr1.om | MJR1: signer prefix 0x03 refused | `mjr1-signer-03` |
| j7 | mjr1.om | MJR1: generation read little-endian | `mjr1-real` |
| j8 | mjr1.om | MJR1: attempt not reported | `mjr1-attempt2-gen` |

## 11. Contract issues

1. Interface v1.1 §3 item 3 example summed to 176 via an unreachable path; attained maximum is 175. Resolved in interface v1.2; MAXLEN 176 kept as a bound.
2. Interface cites `fn kw` at omc.om:808; it is at L810.
3. Exit 43 is a superset of the interface's wording (all `decode_receipt` static rules + `io`).
4. `bound` (id 13) is also an Omega keyword; provenance column says only "code".
5. `0x` and hex digits are Omega syntax only in OM-2 (OMEGESE.md:668); fine for the profile, which is not compiled as Omega.
6. Kind 5 (model message) start and `L` on `act refuse` → 41 notjob are additions, not changes.

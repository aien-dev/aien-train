# omegese45 core CLI (`oe45core`, `oe45gen`)

Pipeline lane of [aien-train#45](https://github.com/aien-dev/aien-train/issues/45). The vocabulary,
grammar and exit codes are fixed by the shared interface (R45, 41 ids; `docs/omegese45/vocab.tsv`
sha256 `c1d7e15466e6585a776314c5c05f35466b2697d9631ad2e11c7918a434a4c18d`). This page documents only
the command line of the two binaries built from `native/omegese45/{data,core}/`.
Training and model details are in [PIPELINE.md](PIPELINE.md).

## Build

```sh
OMC=/path/to/omega-mv/seed/out/omc OMEGA=/path/to/omega-mv OUT=out sh tests/omegese45/core/build.sh
```

`build.sh` checks the omc sha256 and extracts the SHA-256 unit at the pins recorded in
`native/omegese45/core/dependencies.json`. It prints `BUILD_OK` and writes `out/oe45gen`, `out/oe45core`,
`out/oe45test` (the test driver) and `out/oe45mut` (the fault injector). All four are static aarch64 ELF files
produced by omc bf54b19f (sha256 `62775d2a…f3712b`). The unit order is:
SHA unit, `vocab.om`, `msg.om`, `core/model.om`, `core/ckpt.om`, `core/run.om`, then `core/cli.om`
(or `tests/omegese45/core/main.om` for the driver).

## `oe45gen SEED COUNT OUT.oe45 OUT.ids`

Generates COUNT (1..40000) episodes by the reference policy. The only source of randomness is
splitmix64 seeded with SEED.

- `OUT.oe45` is OE45 v1 text. Each episode is a line `oe45 v1` followed by one message per line.
- `OUT.ids` holds one record per episode: one byte L (1..176), then the L ids, one byte each.
- stdout is one line, `oe45gen v1 SEED COUNT` followed by 26 counters, in this order:
  - `mm pair oob retry stop okep`;
  - the dimension edges `e000 e001 e100 e101`;
  - refused × `replay budget worker transport receipt recomp verdict io`;
  - fail × `replay budget worker transport receipt recomp verdict io`.
- Exit codes: 0 ok, 2 usage, 40 a generated message failed the contract validator (a generator defect), 70 I/O.

The issue #45 data comes from `oe45gen 4501 20000` (train) and `oe45gen 4502 1000` (dev).

## `oe45core`

| Command | Effect | stdout |
|---|---|---|
| `init CKDIR TRAIN.ids SEED P1 TOTAL ORDER LRSH` | Writes the step-0 checkpoint (generation 1, slot `ck.a`) bound to `sha256(TRAIN.ids)`. The learning rate is 2^-LRSH; P1 is the length of phase 1 in steps and TOTAL the total number of steps; ORDER seeds the epoch order. | nothing |
| `train CKDIR TRAIN.ids UNTIL EVERY` | Loads the newest valid slot and trains to step min(UNTIL, TOTAL), saving every EVERY steps and at the end. If `sha256(TRAIN.ids)` differs from the binding, exits 54. | `T step phase loss-bits positions` at each save |
| `evaldev CKDIR DEV.ids` | Computes the dev metrics. | `D step episodes positions correct messages exactU exactG` |
| `decode CKDIR U\|G EPFILE` | Decodes the next message after the prefix in EPFILE. | the message words, one line |
| `batch CKDIR U\|G IN OUT` | Decodes every prefix in IN (prefixes are concatenated, each starting at an `oe45 v1` line). | nothing; writes OUT |
| `info CKDIR` | Prints the header of the loaded slot. | `I slot seed step generation P1 total order lrbits algo payload-sha vocab-pin data-sha` |

Decode modes:

- `U` is unguided: argmax over all 41 ids until `end`. The message is then checked with the contract
  validator, and an invalid message exits 50.
- `G` is guided: at each step the argmax is restricted to ids the grammar allows (`ocmask`), so the
  message is valid by construction.
- Both modes stop at the decode budget of 31 ids. Exceeding the budget exits 53 and writes nothing to
  stdout.

A prefix must be OE45 v1 text whose messages are each valid and appear in the order `G (A R)*`.
`act refuse` may appear only as the last A, and the prefix must end in a G or an R. Any other prefix
exits 40.

Each `batch` OUT line is one of:

- `ok <message>`;
- `err 50 <words>`, the invalid unguided message;
- `err 53 <words>`, the 31 ids emitted before the decode budget ran out;
- `err 40` for an invalid prefix.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | ok |
| 2 | usage, or a malformed data file |
| 40 | invalid prefix |
| 50 | DECODE_INVALID |
| 51 | CKPT_REFUSED (no structurally valid slot) |
| 52 | CKPT_VOCAB (the newest valid slot has a different vocab pin) |
| 53 | DECODE_BUDGET |
| 54 | training data differs from the checkpoint binding |
| 70 | file I/O |

## Checkpoint `OE45CK01`

There are two slots, `CKDIR/ck.a` and `CKDIR/ck.b`, each 3,989,908 bytes: a 240-byte header followed by
997,417 binary32 parameters. Every save writes generation g = loaded + 1, to `ck.a` when g is odd and to
`ck.b` when g is even, then calls fsync. The newest valid slot is therefore never the one being
written. A load takes the structurally valid slot with the highest generation, checking:

- size, magic, the architecture words and the payload length;
- the header digest and the payload digest.

The full layout is in the header comment of `native/omegese45/core/ckpt.om`. The suite exercises each
of these behaviors in `tests/omegese45/core/run-core.sh`, section 7:

- fallback when the newest slot is corrupt or truncated;
- 51 when both slots are corrupt or the directory is empty;
- 52 on a wrong vocab pin.

## Tests

```sh
OMC=… OMEGA=… OUT=/tmp/oe45core-suite sh tests/omegese45/core/run-core.sh   # last line OE45_CORE_PASS
```

The receipt is under [`evidence/OMEGESE45/pipeline/`](../../evidence/OMEGESE45/pipeline/).

# GB10 arm of gate 5: NOT_RUN

Status: NOT_RUN. This is not a PASS.

Blocker: Spark scheduling. Drake granted a GPU window on 2026-10-29. One
attempt was queued at about 17:03Z on `/tmp/aien-gb10.lock`. The lock was held
by CodexStackCompare's declared EXCLUSIVE WINDOW (a timing campaign, posted on
`~/handoffs/BOARD.md` at 16:52:54Z). Drake asked the two agents to
coordinate, so the queued attempt was cancelled at about 17:05Z before it
acquired the lock. It never touched the GPU and wrote no receipt. The run
waits for a `SLOT omp-adtrain` post on the board; the board ACK was posted at
17:05:01Z. The chip run itself remains gated by `GB10_CHIP_RUN=1` and
`/tmp/aien-gb10.lock`.

Everything else is ready:

- Script: omega-mv `tests/run_ad_chip.sh` at `965fb7627364274878c35a49e132fae492ff53be`.
- Chip build compile-checked (build only, no device opened) against physics
  `6d7cf0d4d8eb2cda7b512100ff6058e25dbb3ddf`, the value in that branch's
  `physics.lock`, using the clean checkout `~/workspace/physics-sc94`.
- Refusal checks observed: without `GB10_CHIP_RUN=1` the script exits 2, and
  the binary refuses `--chip`.
- Fixture: `frozen/adtrain.fix` (748 bytes; SHA-256 in `../artifacts.sha256`).

Command, once a window is granted (fixture copied outside both trees):

```sh
GB10_CHIP_RUN=1 tests/run_ad_chip.sh --physics-dir ~/workspace/physics-sc94 \
  --fixture /tmp/adtrain-gb10/adtrain.fix --evidence-dir ~/workspace/evidence-out/AD-FIXTURE-GB10
```

PASS requires `AD_FIXTURE_CHIP: exact=…/75 max_ulp=0 verdict=PASS`, or
`PASS_TOL` at max_ulp ≤ 2, and the `--mutant-sign` control must FAIL. Even a
PASS is one fixture on one device; it does not qualify GPU training from `.om`.

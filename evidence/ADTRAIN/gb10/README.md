# GB10 arm of gate 5: EXERCISED (one fixture, one device)

Receipt: `98834f083853d4a405e03e66ae3a7def0f286ff8179b09018f8c090b9fecfdff.json`
(content-addressed by its writer, omega-mv `tests/run_ad_chip.sh`). The two
logs and the stderr are in `blobs/` under their SHA-256 names. The 748-byte
fixture blob equals `../frozen/adtrain.fix` (SHA-256 `4c91d4b9…1bde4`). The
357344-byte chip binary is not copied; its SHA-256 `6c18daa8…65d1c` is in the
receipt.

| field | value |
|---|---|
| run | `20261029T170700Z-965fb7627364`, 17:07:00Z–17:07:08Z, host `spark-b87b`, kernel `7.0.0-1019-nvidia` |
| omega-mv | `965fb7627364274878c35a49e132fae492ff53be`, tree clean before and after |
| physics | `6d7cf0d4d8eb2cda7b512100ff6058e25dbb3ddf` (= `physics.lock`), tree clean before and after |
| chip line | `AD_FIXTURE_CHIP: exact=75/75 max_ulp=0 verdict=PASS`, exit 0 |
| sign-flip control | `word 129 native bc5a0000 C 3c5a0000 ulp 4294967295 …`, exit 1 (FAIL, as required) |
| device calls | stderr `GB10_CALL` rows (simt FMUL/FADD/FSUB/FMNMX, REDUCE_SUM, reduce), all `rc=0` |
| declared tolerance | PASS at max_ulp 0; PASS_TOL at max_ulp ≤ 2; otherwise FAIL |

What it shows: the gradients and values the native `.om` tape computed on the
host for one training step (fixture from mode D) equal, bit for bit, the C M21
tape replaying that step over the M20 GB10 realization on the GB10.

What it does not show: the native `.om` tape does not execute on the GPU (GPU
launch from `.om` is omega-mv #638). This is one fixture on one device, not
qualification of GPU training.

Coordination: the GPU was shared with CodexStackCompare's timing campaign
(`~/handoffs/BOARD.md`). A first attempt queued at about 17:03Z inside their
EXCLUSIVE WINDOW was cancelled before it acquired `/tmp/aien-gb10.lock`; it
wrote no receipt. This run started after their `SLOT omp-adtrain NOW` post
(17:06:34Z). It ran under `GB10_CHIP_RUN=1` and the lock, and touched no
service; 18091 `/health` returned `status ok` before and after. Window
authorised by Drake on 2026-10-29.

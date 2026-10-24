# aien-train#19 host evidence (offline standalone)

Source commit: `d1f7afcc67ec6cef33302d241bece90bdbbd49fb`. Base: train main `ed8b6ea`.
Host: Linux 7.0.0-1019-nvidia aarch64, CPU only. No GPU or simulator ran.
Command: `sh tests/si19/run.sh`. Raw outputs are in `run/` and sealed by
`run/raw.sha256`. `run/build.sha256` lists the ELF, unit and image hashes. The
build files themselves stay in a temporary directory and are not committed.
Wolfram checks of the accounting and exact decodes of every reported metric are
in `wolfram.txt`.

## Results (run/uninterrupted.stdout)
- **Training.** 512 admitted fixture experiences (256 env A, then 256 env B)
  drove 4096 real `afstep` updates: 2048 on M, 1024 on S and 1024 on N.
- **Parameters.** All 8 parameters changed from the seeded init in both MA and
  final M. After A, the model is close to the env-A dynamics, which it never
  saw as rules: W_p->p2 1.00026, W_v->p2 0.99973, W_v->v2 0.96362,
  W_u->v2 0.93549 (true values 1, 1, 1, 1).
- **Held-out A** (128 transitions; none of their (p,v,u) keys appear in
  training). Mean squared error over (p2, v2):

| predictor | MSE |
|---|---|
| model M after A | 0.443491 |
| rules: constant velocity | 1.023438 |
| rules: persistence | 3.984375 |
| frequency mode | 4.476563 |
| nearest-neighbour delta | 5.734375 |
| nearest-neighbour | 11.96875 |
| shuffled-experience control | 79.44 |
| init model | 88.99 |
| reference: true mean dynamics (noise floor, not a baseline) | 0.28125 |

  The model beats every baseline here. It stays above the noise floor.
- **Forgetting on held-out A.** Error was 0.4435 after A. After training on B
  (reversed action) it rises to 1.2529 with bounded A replay and to 3.1193
  without replay. Replay reduces forgetting but does not remove it. One affine
  map cannot represent both action signs; that is a capacity conflict, reported
  as is.
- **Held-out B.**
  - N (B only): 0.2335, close to the floor of 0.2031.
  - M with A replay: 0.8659. This **loses** narrowly to the constant-velocity
    rule (0.8594).
  - MA (trained on A only): 2.7072.
- **Learning curve** (held-out A MSE against experiences: 32, 64, ..., 256):
  1.438, 0.543, 0.423, 0.455, 0.323, 0.353, 0.338, 0.443. The curve is
  non-monotone because SGD runs at a constant step size.
- **Restart.** Runs stopped at event 100 and at event 300 were resumed in a new
  process. Both final 51752-byte images are byte-identical to the uninterrupted
  run (FNV-1a 13533758852048430492), and so are the reports. A damaged image is
  refused (status -5, exit 65), and so is a truncated one (-1).
- **Tests.** `SI19_WORLD_TEST_PASS checks=644630`, including the exact golden
  oracle for two real updates. All six fault-injected units compiled and then
  failed the unchanged test.
- **Cost on this host.** The full driver took 0.01 s wall and 0.01 s user with
  320 KiB max RSS. The test took 0.13 s with 320 KiB. These are host timings
  only: no energy was measured, nothing ran on a GPU, and no speed claim is made.

## Status boundaries
- The data is a seeded FIXTURE, not Omega X1/Cortex experience.
- The held-out sets are development sets, not sealed: G3 sealing is
  BLOCKED_OPERATOR.
- The step consumes the open aien-train#15 native step, and M20-M22 are NOT
  QUALIFIED.
- EST-3 is FAILED, so this result is not a production input.
- Not covered: calibration, curiosity, consent enforcement and a
  pretrained-adviser comparison.

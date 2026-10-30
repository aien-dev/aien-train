# aien-train#19 offline experiential world model (host, standalone)

This is an offline standalone experiment for aien-train#19. It is not M23/M24
(AIEN_0) qualification, not an EST-4+ estimator and not a production input.
EST-3 is FAILED on architecture main, so this module stays outside any runtime,
default build or test target, like the `src/estimation` pattern
(CURRENT_EXECUTION_PLAN Lane 7). M20-M22 remain NOT QUALIFIED. This module adds no
new tensor, autodiff or optimizer. It is a consumer of the aien-train#15
native step, merged on main as c9782bfd, and any T2 result inherits that status. No sealed G3 holdout exists
(sealing is BLOCKED_OPERATOR), so the held-out numbers here are development
results, not sealed results.

## Data source
`native/si19/fixture.om` is a FIXTURE. It is a seeded, resettable micro-world
that stands in for Omega X1 `ExperienceTransition` (omega-mv#699, not on main).
Transitions come from executing fixed dynamics. No model, teacher, corpus or
self-distillation produces them:

- observation: p in -12..12, v in -3..3; action u in {-1,0,1};
- dynamics: p2 = p + v and v2 = v + gain*u + gust, where gust is -1 or +1 with
  probability 1/8 each;
- env A has gain +1 and env B has gain -1 (reversed action);
- a transition is terminal when it leaves the bounds or reaches step 7. A terminal
  record keeps its true successor, and the next record starts from a fresh reset;
- the behaviour policy is seeded uniform random;
- training resets p into -12..-4.

## Model and trainer (`native/si19/world.om`)
- The model is an affine map (p,v,u) -> (p2,v2) with W[3,2] and b[2], 8 binary32
  parameters, randomly initialized from a seed in {-4..3}/16. There is no prior.
- Each update is one call to the pinned `afstep` (`docs/si19/dependencies.json`):
  L720 add32/mul32 forward, sum-squared loss, the explicit gradient and SGD with
  lr = 2^-9. Each batch has two experiences.
- Bounded replay keeps two rings of 64 experience ids. There are 4 updates per
  admitted experience.
- Arms:
  - M is trained on A, then on B with one A-replay sample per batch;
  - N is a copy of M after A, then trained on B only;
  - S is the shuffled-experience control: same ring and schedule, but each row's
    inputs and outcome come from independently drawn experiences;
  - MA and INIT are frozen snapshots.
- Lineage: per arm, `WS` counts how many applied batches used each experience
  id. `wsaffect arm id` answers "did id contribute?" for #20. The counts are
  saved in the checkpoint.
- All resumable state is in WS, TL (the experience log) and FS (fixture
  streams). The checkpoint is a 51752-byte image: magic, version, cell count,
  event, the state, then FNV-1a 64. FNV-1a detects damage but does not
  authenticate. Admission checks length, magic, version, size, checksum and
  semantics (event range, ring counts, update count, finite parameters, ring
  ids, encoded ranges) before it replaces any state.

## Evaluation
For each environment, a separate seed and a wide reset region generate 128
held-out transitions. Every held-out (p,v,u) key is absent from that
environment's training experience, and held-out ids sit outside the training id
range. The metric is mean squared error over (p2,v2), computed in binary32.

The model is compared with these baselines:

- persistence rule;
- constant-velocity rule;
- frequency mode (exact key, falling back to the per-action modal delta);
- nearest-neighbour retrieval, both raw and delta-transfer;
- the shuffled control;
- the init model.

The true-mean-dynamics predictor is shown only as a noise-floor reference (it
uses hidden fixture knowledge). Forgetting is measured as error on held-out A
after B, with and without replay.

## Build and run
`sh tests/si19/run.sh` is disclosed POSIX host glue, kept as the replay
runner. It holds no training, check or verdict logic; all of that is in the .om
units. Its Omega replacement path is an OM2 conductor in .om that performs the
same materialize, hash, build, run and compare steps, as the #14-#16 contracts
do with runner recipes; until that lands the script stays disclosed host glue.
It does the following:

1. Materializes the four pinned sources from Git objects and checks their
   SHA-256 values (`docs/si19/deps.sha256`).
2. Checks the compiler hash.
3. Builds `world-test` and the `world` driver with `quiet-build`.
4. Runs the tests.
5. Runs an uninterrupted job, then stops at events 100 and 300, resumes each in a
   new process and requires the final images and reports to be byte-identical.
6. Requires refusal of damaged and truncated images.
7. Runs six fault-injected units. Each must compile and fail the unchanged test.

Results go to `evidence/SI19/run/`. ELF files and images stay in a temporary
build directory.

`tests/si19/world-test.om` contains:

- an exact integer golden oracle for two real trainer updates (forward, loss,
  dW, db and the committed parameters);
- a forward-only consistency check;
- fixture determinism, seed sensitivity and transition-boundary checks;
- exact lineage counts and the shuffle wiring;
- held-out key and id separation;
- in-process restart at 10 cut points;
- checkpoint refusals that leave state unchanged.

## Not covered
The following remain outside this module:

- calibration and uncertainty outputs;
- curiosity and active experiments;
- consent and revocation enforcement (#20, Open Humanity);
- Cortex/X1 records;
- sealed evaluation;
- GPU or simulator execution;
- a pretrained-adviser comparison;
- energy measurement.

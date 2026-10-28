# Read-only device discovery, 2026-10-23

Scope: Architecture #303, CUDApp #36/#62 and aien-train #7. The raw observations
and exit codes are in `evidence/DISTRIBUTED7/discovery/`. No kernel, GPU workload,
service, resident model, driver, firmware or remote configuration was changed.

| Runner | Observation | Gate classification |
|---|---|---|
| Spark | Linux ARM64, NVIDIA PCI display device `10de:2e12`; this inventory contains no non-NVIDIA GPU | Host discovery only. Existing GB10 owner retains execution and measurement windows. |
| Configured MacBook SSH target | Connection succeeded; macOS 26.6.2 build 25G83; Apple M2, 10 GPU cores, 8 GiB system memory; system profiler reports Metal 4 | Physical non-NVIDIA hardware is reachable. No model/ISA/API execution was run. Whole-model #62 remains **BLOCKED_IMPLEMENTATION**, not blanket BLOCKED_HARDWARE. |
| Configured iMac SSH target | Connection timed out, exit 255 | **BLOCKED_HARDWARE** for this runner's discovery/physical gates. Its GPU model is not inferred from the CPU or an old note. |

On the MacBook, `xcrun --find metal` returned 72 with an unavailable-tool error.
This establishes that this command did not find the offline Metal compiler in
the configured developer environment. It does not establish absence of the
Metal runtime, inability to use an alternative documented API, or lack of a GPU.
No tools were installed or developer settings changed. Driver/firmware versions,
device feature queries, available memory during a workload and numerical-format
support were not measured.

The M2 is a feasible physical target for the existing Apple owner to investigate;
the shared transformer workload has not been selected or demonstrated to fit.
Eight GiB is total system memory, not a promised GPU allocation budget. Exact
model, checkpoint, tokenizer, precision, context and quality fixtures must be
frozen with Omega #661 before any two-vendor claim.

## Reused implementations and owners

- CUDApp main was `80f3a2492eef315106feebfdffcb32de39be7d95` when inspected.
  The canonical native backend and extraction/serving integration remain with
  L395/L386, CUDApp #5/#35 and Omega #638/#661/#666. No second runtime was added.
- Existing `feat/native-amd-agx-lab` was
  `268a2e9dddab02b63140c37ea937ac442c55917e`. Its `native/gpulab` source implements
  bounded AMD/Apple ISA generation and virtual execution. Its documented lab ABI
  explicitly lacks hardware queue submission. The Apple M2 ID is a supported
  simulated subset, not a physical model path. L617/L641 retain this source.
- L444 retains the Metal/Vulkan OS-driver bridge work. CUDApp #42 owns Apple
  family discovery and native integration. No paused owner was restarted.
- The seeded native-host compiler at
  `omega-mv@f0499f415571bdd7496b6de391096915e78f7556` already has native file,
  process, Unix socket and TCP-loopback facilities. Independent lane #660 reran
  its existing suite. Absence of all native networking is therefore **not** a
  valid blocker. Native GPU submission, macOS attachment and authenticated
  Fabric/collective integration remain separate requirements.

## Exact remaining physical gates

- Apple M2: a seeded source-to-device submission path through the existing
  Apple/backend owners; exact OS/API/driver/firmware and feature inventory;
  matching same-model forward outputs, independent numerical oracle and
  meaningful compiled controls; model-level timings and memory costs.
- iMac/AMD candidate: a reachable authorized runner and read-only device/driver
  discovery before choosing any AMD model/API. No hardware purchase is requested.
- Multi-machine training: accessible hosts alone do not qualify a collective.
  The existing authenticated peer transport, native tensor payload execution and
  actual trainer must be integrated before a physical topology is counted.

Host and simulator evidence do not advance these physical gates.

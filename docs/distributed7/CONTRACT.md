# Distributed7 admission v1

This bounded native component checks collective metadata agreement before an
existing transport or numerical executor may begin work. It implements no
transport, authentication, authorization, tensor allocation, sum, gradient,
checkpoint or training operation. `0` means that this metadata transition was
accepted. It never means a collective executed or that its result is correct.

Tracking: [aien-train #7](https://github.com/aien-dev/aien-train/issues/7),
[Architecture #303](https://github.com/aien-dev/aien-architecture-mv/issues/303).
Existing owners retain the compiler, Fabric, tensor engine, trainer and devices.

## Consumed interfaces

| Interface | Exact source | Use |
|---|---|---|
| M20 metadata-v1 | Omega `3a995ba8dace538c1532c747d829ca6e4a73582d`, `native/ecosystem663/metadata.om`, SHA256 `f60b6d62c1bf90de42455effae977de66adeca1268669f6144809a3327aec7a4` | Directly calls `emcheck`; consumes its `ED` and `ES` arrays, without duplicating tensor validation |
| Fabric identities and limits | Omega `c9846a44ef350bdece0a2822decec793b59cbbbc`, `src/runtime/aien_machine_id.h`, `src/fabric/fabric.h` | Full nonzero 32-byte canonical machine IDs; maximum 16 members; generation 1 through 4294967295 |
| Existing task boundary D1 | Same Omega pin, `src/fabric/fab_work.h`, SHA256 `320a9fd366bac11b04a0e6fff336bf25335e49dc237d34062ac727258ee1c0a4` | Maximum 64 MiB tensor payload and 600000 ms deadline span; no new PST1 or HTK1 wire kind |

M20 dtype IDs remain 1=F32, 2=F16, 3=BF16. Their presence here is metadata
recognition only. Floating-point numerical collectives remain unimplemented in
this component. No integer fixture is relabeled as a floating-point tensor.
M20 allows storage slot 0; only generation 0 is invalid.

The descriptor `ED` has 24 unsigned words: schema, dtype, rank, element count,
offset, storage slot, storage generation, reserved zero, eight dimensions,
eight strides in elements. `ES` is a trusted allocator snapshot, not wire data.
The exact upstream module defines both. `emcheck` validates each rank's local
descriptor against its local snapshot. Packed row-major strides are additionally
required here. Scalars are accepted; empty tensors, broadcast views and
noncontiguous views are refused. This is a bounded collective profile, not a
restriction imposed on the shared tensor engine.

## API and trust boundary

The application has one serialized writer. Global arrays are the current
seed compiler's module interface. No thread safety or hostile-process isolation
is claimed. Inputs must be staged before calls; callers must not modify private
state arrays between calls.

Public staging inputs:

- `d7_ids`: up to 16 full canonical 32-byte machine IDs, ordered by group rank.
- `d7_gens`: corresponding Fabric membership generations.
- `d7_peer`: the 32-byte identity proved by the existing authenticated channel.
- `d7_tens`: a nonzero 32-byte semantic tensor identity supplied by the existing
  authority boundary. It names the logical tensor slot and contract across
  workers, not each rank's differing gradient contents.
- `ED`, `ES`: the current locally checked descriptor and allocator snapshot.

Internal arrays are `d7_rostr`, `d7_gen`, `d7_expec`, `d7_meta`, `d7_seen` and
`d7_state`. Input identities are compared in full; no shortened hash, hardware
model name, rank number or process-local handle replaces machine identity.
This component does not derive IDs, validate AMID wire checksums, authenticate a
peer or grant capabilities. A caller-provided identity is not itself evidence
of authentication.

| Entry | Behavior |
|---|---|
| `d7_group(epoch,size)` | Copies the already authorized roster only after validating all members, distinct nonzero IDs, generations and capacity. Requires a strictly higher epoch; cannot replace an active or agreed round. |
| `d7_begin(epoch,sequence,now_ms,deadline_ms,operation)` | Opens one bounded metadata round. Sequence starts at 1; operation 1 is the local sum-agreement tag, not an HTK1 allocation. Snapshots the common descriptor and tensor identity. |
| `d7_offer(rank,epoch,sequence,generation,now_ms,operation)` | Validates one authenticated peer's metadata and records it once. All members must match the semantic tensor ID, dtype, rank, shape, element count and packed strides. Each local offset, storage slot and storage generation may differ. |
| `d7_next(epoch,sequence,now_ms)` | Advances the metadata schedule only after every member agrees. This publishes no tensor or success receipt. Sequence exhaustion retires the group instead of wrapping. |
| `d7_stop(epoch,sequence)` | Cancels an active/agreed metadata round; recovery requires a higher group epoch. |
| `d7_clock(now_ms)` | Applies the local monotonic deadline check without waiting for another offer. At the deadline, including equality, the round expires. |

`now_ms` comes from the caller's trusted local monotonic clock. A remote timestamp
must never drive it. Backwards time is refused. The transport owner must arrange
a timer or call `d7_clock` while waiting; this library owns no background task.
The 600000 ms bound is inherited from HTK1 but this component is not an HTK1 codec.

State modes: 0 empty, 1 idle, 2 collecting metadata, 3 all metadata agreed,
4 cancelled, 5 expired. A wrong request does not add a participant or advance the
sequence. A valid clock observation may expire a round even when the offered
peer metadata is invalid. An agreed round still expires if it is not advanced
before its deadline. Cancelled/expired state cannot be revived by a delayed
message or by moving the local clock backwards.

| Return | Meaning |
|---|---|
| 0 | Metadata transition accepted |
| 1 | Unsupported or mismatched local operation tag |
| 2 | Invalid state |
| 3 | Stale/wrong group epoch |
| 4 | Wrong collective sequence |
| 5 | Rank outside frozen membership |
| 6 | Invalid/stale member generation |
| 7 | Missing/duplicate/wrong full machine identity |
| 8 | Duplicate member contribution |
| 9 | Tensor metadata, identity or local handle refusal |
| 10 | Deadline reached or round already expired |
| 11 | Local monotonic clock moved backwards |
| 12 | Incomplete metadata agreement |
| 13 | Cancelled round |
| 14 | Capacity/deadline-span/sequence limit |

## Qualification and remaining gates

The native test consumer covers complete and incomplete groups, rank permutations,
all 32 bytes of peer and tensor identities, identical element counts with wrong
shapes, differing valid local handles, stale handles, wrong dtype/layout/schema,
duplicates, stale epochs/sequences, generation overflow, cancellation, exact
deadline equality and sequence exhaustion. Five compiled mutants remove peer
binding, admit a partial group, weaken order checks, weaken descriptor matching
or change deadline equality. The suite must reject all five.

This provides no physical or multi-process distributed training result. The
following work stays open under existing owners:

1. Omega #536 and protocols #27 must provide the authenticated native Fabric/PST1
   adapter and allocate an interoperable collective envelope. Current HTK1 defines
   only seeded-i64-matmul and tokenizer tasks. This component does not invent a
   third kind or send a private replacement wire format.
2. Omega #663/#664 and aien-train #6 must provide qualified floating-point
   forward/backward/update and actual tensor payload ownership. This guard does
   not substitute for the trainer's optimizer or collective implementation.
3. aien-train #7 still needs real separate-worker transport, bounded buffers,
   reduction order and numerical oracles, timeout/cancellation integration,
   sharding and failure/restart with durable checkpoints.
4. Consumer cutover, deletion of superseded implementations, independent review,
   physical execution and complete-workflow performance remain open.

No production dependency was retired by these metadata tests.

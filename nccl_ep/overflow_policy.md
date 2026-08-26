# Recv Overflow Policy

Routing is data-dependent, so a rank can be targeted by more tokens than the
`max_recv_tokens_per_rank` budget its group was created with.
`ncclEpGroupConfig_t::overflow_policy` selects what happens then.

This policy applies to **High Throughput (HT) only**; Low Latency ignores the
field.

## Where overflow is detected

Overflow is detected by the metadata scan inside `ncclEpCreateHandle` /
`ncclEpUpdateHandle` — not at dispatch time — by comparing the rank's true recv
total against the budget.

A rank's incoming tokens pass through two stages, and **each stage checks the same
budget in its own coordinate system**:

1. **FLAT staging** — one row per arriving token, deduplicated across the local
   experts it hits. A token whose FLAT slot reaches `max_recv_tokens_per_rank` is
   dropped here.
2. **Expert-major permute** (`NCCL_EP_LAYOUT_EXPERT_MAJOR` only) — one slot per
   *(token, local expert)* pair, with each expert's zone padded up to
   `dispatch_output_per_expert_alignment`. A token whose expert-major slot reaches
   the budget is dropped here.

Stage 2 is strictly more crowded than stage 1: it expands a token routed to *k*
local experts into *k* slots, then adds alignment padding on top. **A rank can
therefore overflow at stage 2 while stage 1 fit comfortably.** `NCCL_EP_LAYOUT_FLAT`
has only stage 1.

## Policies

- **`NCCL_EP_OVERFLOW_TRAP`** — the zero-init default, since `NCCL_EP_OVERFLOW_AUTO`
  resolves to it. The device executes `__trap()` on overflow, which **aborts the
  process** and destroys the CUDA context; it is not a recoverable `ncclResult_t`.
  A diagnostic naming the actual and configured counts is printed from the device
  first. Appropriate when capacity is planned ahead and an overflow means a bug.

- **`NCCL_EP_OVERFLOW_DROP`** — the overflowing tokens are discarded and the
  pipeline continues normally. Dispatch and combine both return `ncclSuccess`.

## Behavior under `DROP`

- `ncclEpLayoutInfo_t.recv_total_counter` reports the **true, pre-drop** total,
  while the recv count driving the rest of the pipeline is clamped to
  `max_recv_tokens_per_rank`. Comparing the two is how a caller detects that a
  drop occurred and how many tokens were lost.
- Expert-major per-expert counts are clamped the same way: an expert whose zone
  begins at or past capacity reports a count of `0`, and its offset clamps to
  capacity.
- Retained slots hold their tokens as usual. On the expert-major permute path
  (the default when `NCCL_EP_HT_EM_LOCAL_DUP` and `NCCL_EP_HT_EM_NVLINK_DUP` are
  off), dispatch additionally zero-fills the recv buffers up to
  `max_recv_tokens_per_rank`: `dispatch_outputs.tokens`, plus `topk_weights` on a
  forward dispatch and `scales` under `NCCL_EP_DISP_QUANT_FWD`. Tokens are dropped
  during FLAT staging, upstream of the permute, so the published per-expert counts
  can claim rows the permute kernel never wrote — neither a delivered token nor
  alignment padding. Zeroing turns those into no-op zero rows instead of stale or
  uninitialized memory. The fill is unconditional under `DROP`, so it costs one
  memset of the recv region per dispatch even when nothing overflows.
- In combine, a dropped assignment contributes **zero** to its token's reduction.
  A token whose assignments were *all* dropped is returned as zeros. Dropped
  tokens are silently absent from the result; nothing downstream reports them.
- Because the counts are clamped by zone arithmetic while delivery is decided per
  slot, `expert_counters[e]` is an **upper bound** on the rows actually written,
  not an exact count. The zero-fill above is what makes consuming that bound safe.

## Example 1: padding overflows the budget that FLAT fit

Two local experts, `max_recv_tokens_per_rank = 8`,
`dispatch_output_per_expert_alignment = 2`. Eight distinct tokens arrive; five
route to `E0`, three to `E1`.

```
Stage 1 — FLAT staging (one row per arriving token)

  slot:   0     1     2     3     4     5     6     7
        [ T0 ][ T1 ][ T2 ][ T3 ][ T4 ][ T5 ][ T6 ][ T7 ]
        `-------------- 8 <= 8: nothing dropped --------------'

Stage 2 — expert-major permute (each zone padded up to 2)

  slot:   0     1     2     3     4    5   |  6     7     8    9
        [ ------ E0: 5 tokens ------ ][pad]  [ E1: 3 tokens ][pad]
                                              `-- kept --'`-- >= 8: DROPPED --'

  true padded total = 6 + 4 = 10  >  8
```

Every token cleared stage 1 with the budget exactly met, yet `E1` still loses one
of its three tokens: padding `E0` from 5 slots to 6 pushed `E1`'s zone up, and its
third token landed at slot 8. The caller observes:

| Reported value | Value | Derivation |
|---|---|---|
| `expert_counters[0]` | `5` | `min(5, room = 8 - 0)` |
| `expert_counters[1]` | `2` | `min(3, room = 8 - 6)` — one of three tokens dropped |
| `expert_offsets[0]` | `0` | zone base |
| `expert_offsets[1]` | `6` | `min(cum = 6, capacity = 8)` |
| `recv_total_counter` | `10` | true pre-drop padded total |

`recv_total_counter (10) > max_recv_tokens_per_rank (8)` is the drop signal.

## Example 2: how stage-1 drops leave phantom rows

Two local experts, `max_recv_tokens_per_rank = 4`, alignment `1` (no padding, to
isolate the effect). Six tokens arrive.

```
Stage 1 — FLAT staging

  slot:   0     1     2     3   |   4     5
        [ T0 ][ T1 ][ T2 ][ T3 ]   [ T4 ][ T5 ]
        `-------- kept --------'   `-- >= 4: DROPPED --'

Routing:   E0 <- T0, T4          E1 <- T1, T2, T3, T5
                     ^^                              ^^
                     already gone after stage 1

Stage 2 — expert-major permute. Zone sizes come from the routing, so tokens
          dropped at stage 1 still reserve their expert-major slot.

  slot:   0     1     2     3   |   4     5
        [ T0 ][ T4 ][ T1 ][ T2 ]   [ T3 ][ T5 ]
           |     |                 `-- >= 4: dropped at stage 2 --'
           |     `-- no FLAT row exists -> never written -> PHANTOM
           `-- delivered
```

`expert_counters` reports `[2, 2]` — four valid rows, slots 0-3. But slot 1 was
never written by anyone: `T4` died at stage 1, so the copy warp has no source row
for it, while stage 2 kept the slot because `1 < 4`. It is neither a delivered
token nor alignment padding, so the pad warp skips it too. Without the zero-fill
it would hand the caller whatever the buffer held before — stale tokens from the
previous iteration, or uninitialized memory on the first.

This is why `expert_counters[e]` is an upper bound: `E0` advertises 2 rows and
delivers 1. Slot positions above are illustrative — the exact assignment depends
on scan order — but the mechanism is exactly this.

## Constraints

Two constraints are easy to trip over:

1. **`DROP` does not let you under-size the recv tensors.** They must still be
   allocated for `max_recv_tokens_per_rank` slots; dispatch returns
   `ncclInvalidArgument` otherwise. `DROP` bounds the damage from a routing spike,
   it does not shrink the buffers.
2. **`DROP` requires an explicit `max_recv_tokens_per_rank`.** It is incompatible
   with [HT eager mode](eager_mode.md) (`max_recv_tokens_per_rank = NCCL_EP_AUTO`), which relies on
   trap semantics, and `ncclEpCreateGroup` rejects the combination.

# HT Eager Mode

By default a High Throughput group commits up front to a fixed per-rank recv
budget, `ncclEpGroupConfig_t::max_recv_tokens_per_rank`, and every dispatch recv
tensor is sized to it. Because routing is data-dependent, that budget has to cover
the worst case, and most iterations leave most of it unused.

**Eager mode** trades that fixed reservation for per-iteration sizing. Create the
group with `max_recv_tokens_per_rank = NCCL_EP_AUTO` and the caller sizes dispatch
recv buffers to the actual recv count of the current routing instead.

Eager mode is **HT only**. Low Latency always sizes its buffers automatically and
ignores the field.

## The derived internal bound

`NCCL_EP_AUTO` does not remove the worst-case reservation — it moves it. At
`ncclEpCreateGroup` the library derives an internal bound:

```
nRanks * max_dispatch_tokens_per_rank * max(num_topk, 1)
```

which is the case where every rank routes every token to this one. That bound
sizes **library-internal** buffers, most importantly the intra-LSA staging buffer.
Only the *caller's* recv tensors get the per-iteration treatment.

This is the mode's main trap: a GPU OOM during `ncclEpCreateGroup` under
`NCCL_EP_AUTO` usually means the derived internal budget is too large, not that
the device is genuinely too small. The library prints the exact arithmetic it used
when that allocation fails. If your routing never approaches the bound, set
`max_recv_tokens_per_rank` explicitly to a measured peak — that is strictly
cheaper than eager mode for internal memory.

## Sizing recv buffers

Supply `ncclEpLayoutInfo_t::recv_total_counter` to `ncclEpCreateHandle` or
`ncclEpUpdateHandle`. The metadata kernel writes the rank's actual recv count
there; copy it device-to-host, synchronize, then allocate.

The counter is readable in **any** mode. What eager mode adds is permission to
*size the dispatch outputs to it* rather than to the worst case.

The two layouts then differ in how dispatch validates that buffer:

- **`NCCL_EP_LAYOUT_FLAT`** — dispatch queries the routed count itself and checks
  `dispatch_outputs.tokens` against it, failing with `ncclInvalidArgument` and a
  "recv buffer too small" diagnostic before any kernel writes caller memory. That
  query costs a device-to-host synchronization inside the dispatch call.
- **`NCCL_EP_LAYOUT_EXPERT_MAJOR`** — the kernels read the routed count from
  device memory, so the recv tensor's own row count *is* the declared capacity and
  no extra synchronization is needed. Padded per-expert zones live in the caller's
  buffer, so size it to include the alignment padding.

A rank may legitimately receive **zero** tokens under eager routing. Recv outputs
may then be empty (`data == nullptr`), but the token and weight outputs must agree
on emptiness — they describe rows of the same recv set, so one cannot be empty
while the other is not.

## Restrictions

1. **`num_topk` is required for `NCCL_EP_LAYOUT_EXPERT_MAJOR`.** Set
   `ncclEpGroupConfig_t::num_topk` to an upper bound on per-token top-k across the
   group's handles; expert-major expansion is unbounded without it.
   `ncclEpInitHandle` returns `ncclInvalidUsage` if it is missing, and validates
   each handle's `num_topk` against it. It is optional for `NCCL_EP_LAYOUT_FLAT`,
   whose deduplicated rows make the factor `1`.
2. **No CUDA Graph capture of `ncclEpDispatch`.** Sizing per routing requires the
   recv count on the host, which is unavailable mid-capture. Capturing dispatch
   under eager mode returns `ncclInvalidUsage`. Use a fixed
   `max_recv_tokens_per_rank` if you need graph capture.
3. **No `NCCL_EP_OVERFLOW_DROP`.** Eager mode relies on trap semantics, and
   `ncclEpCreateGroup` rejects the combination. See
   [Recv Overflow Policy](overflow_policy.md).

## Choosing between the modes

| | Fixed `max_recv_tokens_per_rank` | Eager (`NCCL_EP_AUTO`) |
|---|---|---|
| Caller recv tensors | Worst-case budget | Actual routed count |
| Internal buffers | The configured budget | Derived worst case |
| CUDA Graph capture | Supported | Dispatch cannot be captured |
| `NCCL_EP_OVERFLOW_DROP` | Supported | Rejected |
| Per-dispatch sync | None required | FLAT queries the count device-to-host |

Prefer a fixed budget when you can measure a realistic peak: it uses less internal
memory, captures into graphs, and supports `DROP`. Reach for eager mode when the
caller's recv tensors dominate your memory footprint and their worst case is far
above the typical routing.

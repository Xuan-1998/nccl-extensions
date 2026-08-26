# Zero-Copy Staging

By default NCCL EP moves payloads through library-owned staging buffers: dispatch
writes tokens into an internal buffer that peers read, and combine does the
mirror. When the caller's tensors are already backed by NCCL windows
(`ncclCommWindowRegister`), those copies are avoidable — peers can read and write
the caller's memory directly.

`ncclEpGroupConfig_t::zero_copy` controls that.

## The three modes

| Value | Behavior |
|---|---|
| `NCCL_EP_ZERO_COPY_AUTO` | Zero-init default. Staging stays allocated; window-backed tensors are used directly where the path supports it. |
| `NCCL_EP_ZERO_COPY_OFF` | Identical to `AUTO`. |
| `NCCL_EP_ZERO_COPY_ON` | Windows become **required** for the direct paths. Missing windows are an error, not a fallback. Token staging is not allocated. |

`AUTO` and `OFF` are not distinguished anywhere in the library — only `ON` is
tested. Both leave the pre-enum opportunistic behavior in place: a window-backed
tensor is used directly when the path allows it, and a plain device pointer
quietly stages. Treat the pair as a single "opportunistic" setting.

Note that opportunistic direct use happens **regardless of the mode**. What `ON`
adds is a hard requirement, the removal of the staging allocation, and — for
expert-major — a different algorithm (below).

## High Throughput

Under `NCCL_EP_ZERO_COPY_ON`, two tensors must be window-backed:

| Call | Tensor | On a plain pointer |
|---|---|---|
| `ncclEpDispatch` | `outputs->tokens` | `ncclInvalidArgument` |
| `ncclEpCombine` | `inputs->tokens` | `ncclInvalidArgument` |

Both are required — dispatch's receive side and combine's send side are the two
places peers touch caller memory. Under `NCCL_EP_DISP_QUANT_FWD` the scale tensors
must be windowed in the same pairing; see [Quantization](quantization.md).

Dispatch **input** windowing is a separate, per-tensor opt-in. `ON` constrains
only the output; a windowed `inputs->tokens` is honored in any mode.

### `ON` changes the expert-major algorithm

This is the least obvious consequence. The expert-major recipe is auto-selected
from `zero_copy` and the topology:

| `zero_copy` | LSA teams | Selected mode |
|---|---|---|
| not `ON` | any | `kLocalPermute` — FLAT dispatch, then permute kernels |
| `ON` | > 1 | `kNvlinkDup` — sender duplicates per-expert over NVLink |
| `ON` | 1 | `kLocalDup` — receiver-side fan-out |

So turning zero-copy on does not merely skip a copy; it selects a different
expert-major implementation with different staging, different performance, and
different memory behavior. `NCCL_EP_HT_EM_LOCAL_DUP` and
`NCCL_EP_HT_EM_NVLINK_DUP` override the choice.

### Memory

`ON` elides both token staging regions in the intra-LSA buffer — the dispatch and
combine token regions are simply not allocated.

The scale region is *not* elided, and neither are the per-expert probability
regions.

## Low Latency

`zero_copy` is a **dispatch-only** switch in LL. Combine reads its input directly
and uses windows only to translate peer receive-buffer pointers, so LL combine
stays staged in every mode.

Under `ON`, LL dispatch requires at least one eligible payload window to be
present, and fails with `ncclInvalidArgument` naming the reason otherwise. The
eligible direct path is rank-major BF16 dispatch on an NVLink-only topology with a
window-backed token output. Under `NCCL_EP_DISP_QUANT_FWD`, the token and scale
outputs can be window-backed independently.

`AUTO` / `OFF` stage through library buffers whenever a window is missing.

## Debugging

`NCCL_EP_DEBUG=1` reports, per dispatch, whether zero-copy was selected and why
not when it was not — including the recipe, the requested mode, whether the
topology is NVLink-only, the layout, and which tensors carried windows. That
diagnostic is the fastest way to find out why an `AUTO` group is still staging.

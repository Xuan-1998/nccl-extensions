# Zero-Copy

By default NCCL EP moves payloads through library-owned staging buffers: dispatch
writes tokens into an internal buffer that peers read, and combine does the
mirror. Zero-copy may remove that hop for certain algorithms as peers can read and
write the caller's own input/output buffers directly.

The enabler is an **NCCL window**. A tensor whose memory is registered as a window
is addressable by remote peers, so the library can point them at it instead of at
its own staging.

## Preparing a tensor for zero-copy

Register the buffer with `ncclCommWindowRegister`, then attach the resulting
handle to the `ncclEpTensor_t` descriptor via `win_hdl` / `win_offset` instead of
setting `data`:

```c
size_t dims[2] = { num_recv_slots, hidden };
void*  buf     = nullptr;
ncclMemAlloc(&buf, num_recv_slots * hidden * sizeof(nv_bfloat16));

ncclWindow_t win;
ncclCommWindowRegister(comm, buf, bytes, &win, NCCL_WIN_COLL_SYMMETRIC);

ncclEpTensor_t recv = NCCL_EP_TENSOR_INIT;
recv.ndim = 2;
recv.datatype = ncclBfloat16;
recv.sizes = dims;
recv.win_hdl = win;        // instead of recv.data
recv.win_offset = 0;       // byte offset into the window
```

Notes:

- Registration is **collective** — every rank in the communicator must register
  its corresponding buffer.
- `win_offset` lets several tensors share one registration; give each its byte
  offset within the window.
- Only *user-registered* windows count. The library's own internal window does not
  put a tensor on the zero-copy path.
- A descriptor with no window is an ordinary tensor and simply stages.

Rows and the window offset must satisfy the same 16-byte alignment rules as any
other EP tensor; see [Quantization](quantization.md) for the quantized cases.

## What is supported today

Direct access is used **wherever the path supports it, in any mode** — attaching
a window is enough. Support differs by algorithm.

### High Throughput

Both directions are supported:

| Call             | Tensor            |
|------------------|-------------------|
| `ncclEpDispatch` | `outputs->tokens` |
| `ncclEpCombine`  | `inputs->tokens`  |

Under `NCCL_EP_DISP_QUANT_FWD` the scale tensors participate in the same pairing.
Dispatch **inputs** may also be windowed; that is an independent per-tensor
choice, not something the group flag governs.

Expert-major dispatch on the permute path — what you get under `AUTO`/`OFF` —
stages into the recv buffer even when the tensor is windowed, because the permute
kernel, not the peers, writes the caller's tensor. Setting `ON` selects a
different expert-major mode that does write peer buffers directly; see the
algorithm switch below.

### Low Latency

Zero-copy is **dispatch-only**. LL combine reads its input directly and uses
windows only to translate peer receive-buffer pointers, so it always stages.

LL dispatch takes the direct path only when all of these hold:

- **NVLink-only topology** — `lsa_team_size == nRanks`, i.e. no RDMA leg;
- **`NCCL_EP_LAYOUT_RANK_MAJOR`**;
- recipe is `NCCL_EP_DISP_QUANT_NONE` or `NCCL_EP_DISP_QUANT_FWD`.

Then `outputs->tokens` is written directly when windowed, and under `QUANT_FWD`
`outputs->scales` independently as well — either, both, or neither.

## The group-wide `zero_copy` flag

`ncclEpGroupConfig_t::zero_copy` allows users to inform the library that the
tensors on the direct paths — `ncclEpDispatch` outputs and `ncclEpCombine`
inputs — will have a NCCL window attached. Having zero-copy guaranteed allows
NCCL EP to optimize memory consumption by allocating only the required staging
buffer space.

| Value                    | Behavior                                                                                               |
|--------------------------|--------------------------------------------------------------------------------------------------------|
| `NCCL_EP_ZERO_COPY_AUTO` | Opportunistic: windows are used where supported, missing windows stage.                                |
| `NCCL_EP_ZERO_COPY_OFF`  | Identical to `AUTO`.                                                                                   |
| `NCCL_EP_ZERO_COPY_ON`   | Windows become required. A missing window is an error, not a fallback. Token staging is not allocated. |

Under `ON`:

- **HT** rejects a plain `ncclEpDispatch` `outputs->tokens` or `ncclEpCombine`
  `inputs->tokens` with `ncclInvalidArgument`.
- **LL** requires at least one eligible payload window on dispatch and fails with
  `ncclInvalidArgument` naming the unmet condition otherwise.

### `ON` changes the expert-major algorithm

The expert-major recipe is auto-selected from `zero_copy` and the topology:

| `zero_copy` | LSA teams | Selected mode                                           |
|-------------|-----------|---------------------------------------------------------|
| not `ON`    | any       | `kLocalPermute` — FLAT dispatch, then permute kernels   |
| `ON`        | > 1       | `kNvlinkDup` — sender duplicates per-expert over NVLink |
| `ON`        | 1         | `kLocalDup` — receiver-side fan-out                     |

So `ON` is a performance decision as well as a memory one: it selects a different
implementation with different staging and different behavior.

**The switch cannot be opted out of.** `NCCL_EP_HT_EM_LOCAL_DUP` and
`NCCL_EP_HT_EM_NVLINK_DUP` force a dup mode — including under `AUTO`, where the
staging buffers are still allocated — but there is no corresponding override for
`kLocalPermute`. Permute is reachable only as the fallthrough: no override set
*and* `zero_copy != ON`. Selecting `ON` therefore commits expert-major to a dup
mode, so windowing policy and algorithm choice are not independent.

### Memory

`ON` elides both token staging regions in the intra-LSA buffer — the dispatch and
combine token regions are simply not allocated. The scale region and the
per-expert probability regions are not elided.

## Debugging

`NCCL_EP_DEBUG=1` reports, per dispatch, whether zero-copy was selected and — when
it was not — which condition failed: the recipe, the requested mode, whether the
topology is NVLink-only, the layout, and which tensors carried windows. That is
the fastest way to find out why a group is still staging.

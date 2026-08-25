# Quantization

This document defines the public tensor contracts for NCCL EP quantization
recipes. `B` is the batch size, `H` is the logical hidden dimension, and `S` is
the caller-provided number of scale elements per token.

## `NCCL_EP_DISP_QUANT_FWD`

This recipe forwards the physical bytes of two 2D inputs: tokens `[B x H]` and
scales `[B x S]`. Tokens may use FP32, FP16, BF16, FP8, or FP4x2; scales may
use FP32, FP16, BF16, FP8, or `ncclUint8` raw byte storage. `S` is taken
directly from the scale tensor; the recipe does not infer a scale-block size.
Each physical row and storage base (or window offset) must be 16-byte aligned.

FP4x2 is a packed pair of logical FP4 values. For logical hidden size `H`, use
physical token shape `[B x H/2]` (`sizes[1]` is `H/2`); the alignment rule
therefore requires `H` to be a multiple of 32. NCCL EP byte-forwards this
reserved type—NCCL collectives do not support it. Dispatch scale outputs have
the same leading layout dimensions as token outputs and `S` as their final
dimension. In LL rank-major mode, token and scale output descriptors can
independently be backed by NCCL windows.

`QUANT_FWD` applies to both HT and LL, and to both pass directions; `FWD`
means that scales are forwarded, not that the operation is a forward pass.
Both token and scale outputs are required, must match their respective input
dtype and physical row width, and `round_scales` must be zero. HT outputs are
2D and are either both window-backed or neither is; expert-major permutation
may stage before writing those windows. With `zero_copy = NCCL_EP_ZERO_COPY_ON`,
provide paired token and scale windows.

## `NCCL_EP_DISP_QUANT_DS_FP8E3M4`

This LL-only recipe performs DeepSeek FP8 E3M4 quantization internally. Supply
BF16 token inputs and no input scale tensor. Dispatch produces E4M3 token
bytes and required FP32 output scales, one per 128 token elements. The hidden
dimension must be divisible by 512.

## `NCCL_EP_DISP_QUANT_NONE` and `NCCL_EP_COMB_QUANT_NONE`

The unquantized recipes transport tokens in their declared dtype. Do not
provide scale tensors.

## `NCCL_EP_COMB_QUANT_NVFP4`

This experimental LL-only recipe transports BF16 expert outputs using NVFP4 and
requires CUDA 12.9+ with `cuda_fp4.h` and an E2M1 FP4-family GPU target. Set
`combine_config.quant_recipe` and provide `combine_inputs.scales` as a FP32 3D
tensor with the same leading dimensions as `combine_inputs.tokens` and a final
dimension of one. Each valid post-expert token needs one global scale, computed
across its activation elements as `2688 / amax(abs(token))`.

## `max_token_bytes`

`ncclEpGroupConfig_t::max_token_bytes` is the physical-byte budget for one
token row, including any attached scale data. For example, packed FP4 with `S`
scale elements requires `H/2 + S * sizeof(scale_dtype)` bytes. In HT, configure
this bound as a multiple of 16 bytes.

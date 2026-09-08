# NCCL EP HT on EFA — RL-tuned results (research/rl-ep-autotune)

All numbers measured on 2 nodes x 8 H200 (EP16) over EFA with the unordered-fabric
port in this branch (`NCCL_EP_UNORDERED_FABRIC=1`, GDA backend `NCCL_GIN_TYPE=5`),
`ep_bench --algorithm ht --hidden 7168 --top-k 8 --experts 256 --warmup 5 --iters 20`,
reading `send: rdma_bw` from the "BW based on total time" block (the NCCLOFI-1931
runbook reading rule). Every config passed `ep_test -a ht` 16/16 before and after.

## Best verified configs by token count

| tokens/rank | config (qps, chunk, subputs, cb, sms) | dispatch | combine | total GB/s |
|---|---|---|---|---|
| 8192  | 2, 256, 6, 32, 11  | 101.23 | 91.76 | **192.99** |
| 12288 | 2, 384, 8, 32, 12  | —      | —     | **198.43** |
| 16384 | 2, 512, 8, 32, 11  | 96.66  | 94.28 | **191.45** |
| 2048  | 2, 256, best-known | 59.32  | 57.48 | 116.80 |
| 1536  | 2, 128, 8, 8, 8    | 72.28  | 61.93 | 134.21 |

Reference points on the same harness: the pre-tuning GDA best was 85.8 / 77.0
(qps=2, chunk 512, subputs 16, cb 32) and the proxy-ordered ticket baseline was
43.7 / 75.0 at 8192 tokens.

Two structural findings from the search:

- The chunk law is size-dependent: c128 wins at 512-1536 tokens, c256 at 2048-8192,
  c512 at 16384 (c256 fails correctness there). Bandwidth over token count is
  non-monotone with a genuine peak at 12288 tokens.
- qps=2 is required at the fast shapes; qps=4 fails the correctness gate.

These configs were found by an RL loop (a trained model proposing configs against a
hardware-measuring judge with an ep_test correctness gate) plus targeted sweeps; the
8192-token optimum matches the best human-guided result exactly.

## Reproduce (runtime env, no rebuild needed)

Source `docs/tuning/tuned_best_8192.env` (or the 16k variant) before launching.

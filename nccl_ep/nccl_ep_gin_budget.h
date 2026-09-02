/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#ifndef NCCL_EP_GIN_BUDGET_H_
#define NCCL_EP_GIN_BUDGET_H_

// ============================================================================
// HT GIN resource model: signal namespace layout and endpoint budget.
//
// Everything the HT kernels and the group bootstrap need to agree on about
// GIN signals and contexts lives here, so the layout identities are stated
// once and checked at compile time instead of being repeated as inline
// arithmetic at every use site.
//
// Why a budget model exists at all: on the EFA GDA backend, an indexed signal
// is hardware, not software. For every GIN context the plugin creates one
// data endpoint plus one signal/counter endpoint PER indexed signal, and each
// signal/counter endpoint holds two hardware counters (FI_WRITE +
// FI_REMOTE_WRITE). Contexts spread across the GPU's EFA rails
// (rail = context_id % num_rails). The provider budget is roughly 256 QPs per
// NIC, and the hardware-counter pool binds at the same scale. A request over
// budget does not fail with a readable error: fi_enable returns ENOMEM in the
// middle of createContext. The helpers here compute the per-NIC cost up front
// so bootstrap can print what actually happened and what to change.
//
// Empirical calibration on p5en (H200, 2 EFA rails per GPU), 2 nodes x 16
// chunks (113 signals under the pre-compaction layout):
//   1 context/NIC x (1 + 2*113) = 227 counters -> fits
//   2 contexts/NIC x 227        = 454 counters -> ENOMEM
// which is why the counter pool, not the QP count, is treated as the binding
// constraint below.
//
// Signal namespace (per receiving rank). Signals are per-receiving-rank
// resources, and HT rail communicators pair ranks of equal local rank, so a
// signal id only needs to distinguish (source remote node, chunk):
//
//   [0,  E)        combine tail signals,  id = src_remote * max_chunks + chunk
//   [E,  2E)       dispatch tail signals, id = src_remote * max_chunks + chunk
//   [2E, 2E + 32)  slack for NCCL barrier sessions
//
// with E = (rdma_team_size - 1) * max_chunks_per_rank and
// src_remote = src_node < dst_node ? src_node : src_node - 1. The dimensions
// the old layout also carried ([dst] and [local_rank]) are constants at any
// given receiver and are deliberately absent.
// ============================================================================

namespace nccl_ep {
namespace gin_budget {

// ---- provider budget (EFA GDA, empirical; see header comment) --------------
static constexpr int kQpBudgetPerNic = 256;
static constexpr int kCountersPerNicBudget = 256;
static constexpr int kCountersPerScEndpoint = 2; // FI_WRITE + FI_REMOTE_WRITE

// Barrier slack kept at the tail of the signal space for NCCL-internal use.
static constexpr int kBarrierSignalSlack = 32;

// ---- signal namespace layout ------------------------------------------------
__host__ __device__ constexpr int edge_chunk_signals(int rdma_team_size, int max_chunks_per_rank) {
    return (rdma_team_size - 1) * max_chunks_per_rank;
}

__host__ __device__ constexpr int combine_signal_offset() {
    return 0;
}

__host__ __device__ constexpr int dispatch_tail_base(int rdma_team_size, int max_chunks_per_rank) {
    return edge_chunk_signals(rdma_team_size, max_chunks_per_rank);
}

__host__ __device__ constexpr int total_signals(int rdma_team_size, int max_chunks_per_rank) {
    return 2 * edge_chunk_signals(rdma_team_size, max_chunks_per_rank) + kBarrierSignalSlack;
}

// The two directions must tile the space exactly, with combine first: the
// kernels derive header-slot indices as (signal_id - combine base), so any
// gap or reordering here corrupts the header region silently.
static_assert(combine_signal_offset() == 0, "combine region must start the signal space");
static_assert(dispatch_tail_base(2, 16) == edge_chunk_signals(2, 16),
              "dispatch tail region must start where the combine region ends");
static_assert(total_signals(2, 16) == 2 * 16 + 32, "2-node x 16-chunk shape: 64 signals");
static_assert(total_signals(8, 32) == 2 * 7 * 32 + 32, "8-node x 32-chunk shape");

// ---- per-NIC cost model -----------------------------------------------------
// contexts_on_busiest_nic = ceil(contexts / rails); each context on a NIC owns
// 1 data endpoint + n_signals sc endpoints.
__host__ __device__ constexpr int contexts_on_busiest_nic(int num_contexts, int num_rails) {
    return (num_contexts + num_rails - 1) / num_rails;
}

__host__ __device__ constexpr int qps_per_nic(int num_contexts, int num_rails, int n_signals) {
    return contexts_on_busiest_nic(num_contexts, num_rails) * (1 + n_signals);
}

__host__ __device__ constexpr int counters_per_nic(int num_contexts, int num_rails, int n_signals) {
    return contexts_on_busiest_nic(num_contexts, num_rails) *
           (1 + kCountersPerScEndpoint * n_signals);
}

__host__ __device__ constexpr bool fits_gda_budget(int num_contexts, int num_rails, int n_signals) {
    return qps_per_nic(num_contexts, num_rails, n_signals) <= kQpBudgetPerNic &&
           counters_per_nic(num_contexts, num_rails, n_signals) <= kCountersPerNicBudget;
}

// Largest context count that fits the GDA budget for a given signal count.
// Counters bind before QPs whenever kCountersPerScEndpoint > 1, so this is
// effectively rails * floor(budget / (1 + 2 * n_signals)).
__host__ __device__ constexpr int max_contexts_for(int num_rails, int n_signals) {
    const int per_nic_by_counters = kCountersPerNicBudget / (1 + kCountersPerScEndpoint * n_signals);
    const int per_nic_by_qps = kQpBudgetPerNic / (1 + n_signals);
    const int per_nic = per_nic_by_counters < per_nic_by_qps ? per_nic_by_counters : per_nic_by_qps;
    return per_nic * num_rails;
}

// Calibration anchors: the shapes measured on p5en (2 rails).
static_assert(fits_gda_budget(/*ctx=*/2, /*rails=*/2, total_signals(2, 16)),
              "the validated p5en config (2 contexts, 64 signals) must fit");
static_assert(!fits_gda_budget(/*ctx=*/4, /*rails=*/2, /*n_signals=*/113),
              "the observed ENOMEM config (4 contexts, pre-compaction 113 signals) must not fit");
static_assert(max_contexts_for(/*rails=*/2, total_signals(2, 16)) >= 2,
              "at the compacted 2-node namespace at least 2 contexts must fit");

} // namespace gin_budget
} // namespace nccl_ep

#endif // NCCL_EP_GIN_BUDGET_H_

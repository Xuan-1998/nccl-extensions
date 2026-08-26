/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Regression test for interleaved (1F1B / PP) HT dispatch on two live handles.
 *
 *   Companion to test_ht_combine_pp_interleave. Each handle's dispatch must route
 *   from its own per-handle state even when a second handle is created / updated
 *   between this handle's create and its dispatch (as in a 1F1B schedule). If any
 *   routing state dispatch relies on were group-scoped and keyed only by rank, the
 *   later handle would clobber it and the earlier handle would scatter tokens to
 *   the wrong experts.
 *
 * Test recipe (expert-major, exercised under every HT EM mode):
 *   1. Create handle A (routing R_A) then handle B (routing R_B), R_A != R_B per
 *      token. A is created first, so any shared per-rank routing state ends up
 *      holding R_B by the time A's dispatch runs.
 *   2. Forward-dispatch both; verify each rank's received token-value set matches
 *      that handle's routing. B's forward is scheduled before A's backward to
 *      mirror the 1F1B interleave in the second test.
 *   3. Backward scatter (no routing input) reuses the handle's forward routing;
 *      verify the older handle A still scatters to its own experts after the newer
 *      handle B ran.
 *
 * The received-value SET is layout/mode independent (dup modes duplicate slots but
 * not distinct values), so the same expectations hold under every EM mode.
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"
#include <set>

static float bf16_val(nv_bfloat16 v) {
    return __bfloat162float(v);
}

class HtDispatchPpInterleaveTest : public ::testing::Test {
protected:
    // Distinct per-handle routing (function of source rank + token) so a crossover
    // scatters to the wrong experts and changes the received set.
    static int64_t route_a(int src_rank, int token) { return (src_rank * kNumTokens + token) % kNumExperts; }
    static int64_t route_b(int src_rank, int token) { return (src_rank * kNumTokens + token + 1) % kNumExperts; }

    // Local experts hosted by a rank: [rank * epr, rank * epr + epr).
    static int experts_per_rank() { return kNumExperts / g_nranks; }
    static int host_rank_of_expert(int64_t expert) { return static_cast<int>(expert) / experts_per_rank(); }

    // Token value shipped by source rank s, token i: a unique integer, exactly
    // representable in bf16 (< 256), keyed by a per-pass base.
    static float token_value(float base, int src_rank, int token) {
        return base + static_cast<float>(src_rank * kNumTokens + token);
    }

    // Distinct token values this rank should receive under a given routing.
    std::set<float> expected_recv_set(float base, int64_t (*route)(int, int)) {
        std::set<float> s;
        for (int src = 0; src < g_nranks; ++src)
            for (int i = 0; i < kNumTokens; ++i)
                if (host_rank_of_expert(route(src, i)) == g_rank) s.insert(token_value(base, src, i));
        return s;
    }

    struct Disp {
        int64_t* d_topk = nullptr;
        nv_bfloat16* d_tok = nullptr;
        nv_bfloat16* d_recv = nullptr;
        float* d_weights = nullptr;
        float* d_recv_w = nullptr;
        ncclEpTensor_t* t_topk = nullptr;
        ncclEpTensor_t* t_tok = nullptr;
        ncclEpTensor_t* t_recv = nullptr;
        ncclEpTensor_t* t_w = nullptr;
        ncclEpTensor_t* t_recv_w = nullptr;
        ncclEpHandle_t handle = nullptr;

        void free_io() {
            if (t_recv_w) ncclEpTensorDestroy(t_recv_w);
            if (t_w) ncclEpTensorDestroy(t_w);
            if (t_recv) ncclEpTensorDestroy(t_recv);
            if (t_tok) ncclEpTensorDestroy(t_tok);
            t_recv_w = t_w = t_recv = t_tok = nullptr;
            if (d_recv_w) cudaFree(d_recv_w);
            if (d_weights) cudaFree(d_weights);
            if (d_recv) cudaFree(d_recv);
            if (d_tok) cudaFree(d_tok);
            d_recv_w = d_weights = nullptr;
            d_recv = d_tok = nullptr;
        }
        void free_all() {
            free_io();
            if (t_topk) ncclEpTensorDestroy(t_topk);
            if (d_topk) cudaFree(d_topk);
            if (handle) ncclEpHandleDestroy(handle);
        }
    };

    // Create an expert-major handle with the given routing.
    void create_handle(Disp& st, int64_t (*route)(int, int)) {
        int64_t h_topk[kNumTokens];
        for (int i = 0; i < kNumTokens; ++i) h_topk[i] = route(g_rank, i);
        CUDA_ASSERT(cudaMalloc(&st.d_topk, kNumTokens * kTopK * sizeof(int64_t)));
        CUDA_ASSERT(cudaMemcpy(st.d_topk, h_topk, sizeof(h_topk), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&st.t_topk, 2, ncclInt64, st.d_topk, kNumTokens, kTopK));
        NCCL_ASSERT(ncclEpCreateHandle(
            &st.handle, g_ep_group_em, NCCL_EP_LAYOUT_EXPERT_MAJOR, st.t_topk, nullptr, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        ASSERT_NE(st.handle, nullptr);
    }

    // Forward dispatch with tokens carrying token_value(base, ...). Retains recv.
    void forward_dispatch(Disp& st, float base) {
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = token_value(base, g_rank, i);
            for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }
        std::vector<float> h_w(kNumTokens * kTopK, 1.0f);

        CUDA_ASSERT(cudaMalloc(&st.d_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&st.d_recv, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMemset(st.d_recv, 0, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&st.d_weights, kNumTokens * kTopK * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&st.d_recv_w, kMaxRecvSlots * sizeof(float)));
        CUDA_ASSERT(cudaMemset(st.d_recv_w, 0, kMaxRecvSlots * sizeof(float)));
        CUDA_ASSERT(cudaMemcpy(st.d_tok, h_tok.data(), kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(st.d_weights, h_w.data(), kNumTokens * kTopK * sizeof(float), cudaMemcpyHostToDevice));

        NCCL_ASSERT(epTensorCreate(&st.t_tok, 2, ncclBfloat16, st.d_tok, kNumTokens, kHidden));
        NCCL_ASSERT(epTensorCreate(&st.t_recv, 2, ncclBfloat16, st.d_recv, kMaxRecvSlots, kHidden));
        NCCL_ASSERT(epTensorCreate(&st.t_w, 2, ncclFloat32, st.d_weights, kNumTokens, kTopK));
        NCCL_ASSERT(epTensorCreate(&st.t_recv_w, 1, ncclFloat32, st.d_recv_w, kMaxRecvSlots));

        ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in.tokens = st.t_tok;
        d_in.topk_weights = st.t_w;
        d_out.tokens = st.t_recv;
        d_out.topk_weights = st.t_recv_w;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        NCCL_ASSERT(ncclEpDispatch(st.handle, &d_in, &d_out, nullptr, &dcfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(st.handle, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    }

    // Backward scatter (no routing input): re-ship new token values under the
    // handle's forward routing. Retains recv.
    void backward_scatter(Disp& st, float base) {
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = token_value(base, g_rank, i);
            for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }
        CUDA_ASSERT(cudaMalloc(&st.d_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&st.d_recv, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMemset(st.d_recv, 0, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMemcpy(st.d_tok, h_tok.data(), kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

        NCCL_ASSERT(epTensorCreate(&st.t_tok, 2, ncclBfloat16, st.d_tok, kNumTokens, kHidden));
        NCCL_ASSERT(epTensorCreate(&st.t_recv, 2, ncclBfloat16, st.d_recv, kMaxRecvSlots, kHidden));

        ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in.tokens = st.t_tok;
        d_out.tokens = st.t_recv;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        dcfg.pass_direction = NCCL_EP_BWD_PASS;
        NCCL_ASSERT(ncclEpDispatch(st.handle, &d_in, &d_out, nullptr, &dcfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(st.handle, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    }

    // Read the distinct received token-value set for this rank.
    std::set<float> recv_set(Disp& st) {
        unsigned int num_recv = 0;
        EXPECT_EQ(ncclEpHandle_test_getNumRecvTokens(st.handle, &num_recv), ncclSuccess);
        std::vector<nv_bfloat16> h_recv(static_cast<size_t>(num_recv) * kHidden);
        if (num_recv > 0)
            EXPECT_EQ(
                cudaMemcpy(
                    h_recv.data(), st.d_recv, static_cast<size_t>(num_recv) * kHidden * sizeof(nv_bfloat16),
                    cudaMemcpyDeviceToHost),
                cudaSuccess);
        std::set<float> got;
        for (unsigned s = 0; s < num_recv; ++s) got.insert(bf16_val(h_recv[static_cast<size_t>(s) * kHidden]));
        return got;
    }
};

// Two live handles, two forward dispatches back to back. A (created first) is the
// victim: a shared per-rank routing buffer would hold B's routing by now.
TEST_F(HtDispatchPpInterleaveTest, TwoForwardDispatches) {
    constexpr float kBaseA = 1.0f, kBaseB = 100.0f;
    Disp a, b;
    create_handle(a, &route_a);
    create_handle(b, &route_b);
    forward_dispatch(a, kBaseA);
    forward_dispatch(b, kBaseB);

    EXPECT_EQ(recv_set(a), expected_recv_set(kBaseA, &route_a))
        << "handle A rank " << g_rank << " forward dispatch routed with the other handle's routing";
    EXPECT_EQ(recv_set(b), expected_recv_set(kBaseB, &route_b))
        << "handle B rank " << g_rank << " forward dispatch routed with the other handle's routing";

    b.free_all();
    a.free_all();
}

// 1F1B interleave: forward-dispatch the newer handle B, then backward-scatter the
// older handle A. A's scatter must reuse A's own forward routing, not B's.
TEST_F(HtDispatchPpInterleaveTest, ForwardThenBackwardDispatch) {
    constexpr float kBaseA = 1.0f, kBaseB = 100.0f, kBaseAbwd = 200.0f;
    Disp a, b;
    create_handle(a, &route_a);
    forward_dispatch(a, kBaseA);  // prime A's routing
    create_handle(b, &route_b);
    forward_dispatch(b, kBaseB);

    EXPECT_EQ(recv_set(b), expected_recv_set(kBaseB, &route_b))
        << "handle B rank " << g_rank << " forward dispatch routed with the other handle's routing";

    a.free_io();  // release A's forward recv/input tensors; keep handle + routing
    backward_scatter(a, kBaseAbwd);
    EXPECT_EQ(recv_set(a), expected_recv_set(kBaseAbwd, &route_a))
        << "handle A rank " << g_rank << " backward scatter routed with the other handle's routing";

    b.free_all();
    a.free_all();
}

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_ht_dispatch_pp_interleave_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}

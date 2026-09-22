// Compare Q4_0 MUL_MAT on the CPU_NEUTRON extra buffer type against the plain CPU path.
// usage: test-neutron-mm [KxNxM ...]   e.g. 1024x1024x1 1024x3072x512
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static ggml_backend_buffer_type_t find_buft(ggml_backend_t be, const char * name) {
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto get = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (!get) return nullptr;
    for (ggml_backend_buffer_type_t * b = get(dev); b && *b; b++)
        if (strcmp(ggml_backend_buft_name(*b), name) == 0) return *b;
    return nullptr;
}

// returns median seconds per run and fills out
static double run(ggml_backend_t be, ggml_backend_buffer_type_t wbuft, const std::vector<uint8_t> & wq, int K, int N, int M,
                  const std::vector<float> & a, std::vector<float> & out, int iters) {
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_0, K, N); ggml_set_name(W, "W");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, wbuft);
    if (!wbuf) { fprintf(stderr, "alloc W failed\n"); exit(1); }
    ggml_backend_tensor_set(W, wq.data(), 0, wq.size());

    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M); ggml_set_name(A, "A");
    ggml_tensor * Y = ggml_mul_mat(ctx, W, A); ggml_set_name(Y, "Y");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(A, a.data(), 0, a.size() * 4);
    ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, Y);

    ggml_backend_graph_compute(be, gf);  // warmup (first call packs nothing; W packed at set_tensor)
    std::vector<double> ts;
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(be, gf);
        ts.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    out.resize((size_t)N * M); ggml_backend_tensor_get(Y, out.data(), 0, out.size() * 4);
    ggml_backend_buffer_free(buf); ggml_backend_buffer_free(wbuf); ggml_free(ctx); ggml_free(wctx);
    return ts[ts.size() / 2];
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_name("CPU", nullptr);
    ggml_backend_cpu_set_n_threads(be, 6);
    ggml_backend_buffer_type_t npu = find_buft(be, "CPU_NEUTRON");
    if (!npu) { fprintf(stderr, "CPU_NEUTRON buffer type not available\n"); return 1; }
    std::vector<std::string> shapes = { "1024x1024x1", "1024x3072x1", "3072x1024x1", "1024x1024x512", "3072x1024x512", "5120x5120x1", "5120x13824x1", "5120x13824x512" };
    if (argc > 1) shapes.assign(argv + 1, argv + argc);
    int fails = 0;
    for (auto & s : shapes) {
        int K, N, M; if (sscanf(s.c_str(), "%dx%dx%d", &K, &N, &M) != 3) { fprintf(stderr, "bad shape %s\n", s.c_str()); return 1; }
        std::mt19937 rng(0); std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> wf((size_t)K * N); for (auto & v : wf) v = nd(rng);
        std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q4_0, K) * N);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, wf.data(), wq.data(), 0, N, K, nullptr);
        std::vector<float> a((size_t)K * M); for (auto & v : a) v = nd(rng);
        std::vector<float> yc, yn;
        const int iters = M > 64 ? 5 : 20;
        double tc = run(be, ggml_backend_cpu_buffer_type(), wq, K, N, M, a, yc, iters);
        double tn = run(be, npu, wq, K, N, M, a, yn, iters);
        double num = 0, den = 0;
        for (size_t i = 0; i < yc.size(); i++) { double d = yn[i] - yc[i]; num += d * d; den += (double)yc[i] * yc[i]; }
        double rel = std::sqrt(num / (den + 1e-30));
        bool ok = rel < 0.03;
        fails += !ok;
        printf("%-18s cpu %8.2f ms  neutron %8.2f ms  x%5.2f  relRMS %.4f  %s\n", s.c_str(), tc * 1e3, tn * 1e3, tc / tn, rel, ok ? "OK" : "FAIL");
        fflush(stdout);
    }
    ggml_backend_free(be);
    return fails ? 1 : 0;
}

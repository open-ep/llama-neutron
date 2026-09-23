// neutron-pack: pre-pack a GGUF's Q4_0 weights into the Neutron .npk cache WITHOUT touching the NPU or CMA.
// Packing is deterministic, so the cache produced here is byte-identical to what the ggml backend writes at
// load time -- but with no 10 GB CMA pinned and no model mmap fighting for RAM, and with -t N it runs in
// parallel (one tensor per thread).
//   neutron-pack [-t N] [--cache-dir DIR] [--dry-run] model.gguf
#include "ggml.h"
#include "gguf.h"
#include "neutron_pack.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

struct job { std::string name; int N, K; size_t off, nbytes; std::vector<int> kc; };

static int noop_clean(const void *, int) { return 0; }

int main(int argc, char ** argv) {
    int nthreads = 1; bool dry = false; const char * model = nullptr;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-t" && i + 1 < argc) nthreads = atoi(argv[++i]);
        else if (a == "--cache-dir" && i + 1 < argc) setenv("NEUTRON_CACHE_DIR", argv[++i], 1);
        else if (a == "--dry-run") dry = true;
        else if (a[0] != '-') model = argv[i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (!model) { fprintf(stderr, "usage: neutron-pack [-t N] [--cache-dir DIR] [--dry-run] model.gguf\n"); return 1; }
    neutron_clean_cache = noop_clean;   // host memory only: nothing to flush to a device

    ggml_context * ctx = nullptr;
    gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ &ctx };
    gguf_context * g = gguf_init_from_file(model, ip);
    if (!g) { fprintf(stderr, "cannot open %s\n", model); return 1; }
    const size_t data_off = gguf_get_data_offset(g);

    std::vector<job> jobs; size_t skipped_type = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t || t->type != GGML_TYPE_Q4_0 || ggml_n_dims(t) != 2) { skipped_type++; continue; }
        const int K = (int)t->ne[0], N = (int)t->ne[1];
        if (N % 128 || K % 32) { skipped_type++; continue; }
        const std::vector<int> & kc = neutron_k_chunks(K);
        if (kc.empty()) { skipped_type++; continue; }
        jobs.push_back({name, N, K, data_off + gguf_get_tensor_offset(g, i), ggml_nbytes(t), kc});
    }
    double total_params = 0; for (auto & j : jobs) total_params += (double)j.N * j.K;
    printf("%s: %zu Q4_0 tensors for the NPU (%.2fB params), %zu others stay on CPU; threads=%d%s\n",
           model, jobs.size(), total_params / 1e9, skipped_type, nthreads, dry ? " (dry run)" : "");
    if (dry) { for (auto & j : jobs) printf("  %-32s N=%6d K=%6d chunks=%zu\n", j.name.c_str(), j.N, j.K, j.kc.size()); return 0; }

    std::atomic<size_t> next{0}, packed{0}, cached{0}, failed{0}; std::atomic<double> bytes{0};
    std::mutex pm; auto t0 = std::chrono::steady_clock::now();
    auto worker = [&]() {
        FILE * f = fopen(model, "rb");
        if (!f) { failed++; return; }
        for (size_t i; (i = next++) < jobs.size();) {
            const job & j = jobs[i];
            std::vector<uint8_t> raw(j.nbytes);
            if (fseeko(f, (off_t)j.off, SEEK_SET) != 0 || fread(raw.data(), 1, j.nbytes, f) != j.nbytes) { failed++; continue; }
            const std::string cpath = neutron_cache_path(raw.data(), j.nbytes, j.N, j.K);
            if (cpath.empty()) { fprintf(stderr, "NEUTRON_CACHE_DIR is empty: nowhere to write\n"); failed++; continue; }
            struct stat st;
            if (stat(cpath.c_str(), &st) == 0 && st.st_size > (off_t)sizeof(npk_hdr)) { cached++; continue; }
            size_t cap = 0; for (int k : j.kc) cap += GGML_PAD(neutron_pack_bound(j.N, k), 64);
            std::vector<uint8_t> out(cap);
            std::vector<neutron_layout> L;
            auto ts = std::chrono::steady_clock::now();
            if (neutron_pack_q4_0(raw.data(), j.N, j.K, j.kc, out.data(), cap, L) != 0 || !neutron_npk_write(cpath, j.kc, L)) {
                std::lock_guard<std::mutex> lk(pm); fprintf(stderr, "FAILED %s\n", j.name.c_str()); failed++; continue;
            }
            size_t tot = 0; for (auto & l : L) tot += l.total;
            bytes = bytes + (double)tot; packed++;
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
            std::lock_guard<std::mutex> lk(pm);
            printf("[%3zu/%zu] %-32s N=%6d K=%6d -> %6.1f MB  %5.1fs\n", packed + cached, jobs.size(), j.name.c_str(), j.N, j.K, tot / 1e6, dt);
            fflush(stdout);
        }
        fclose(f);
    };
    std::vector<std::thread> th;
    for (int i = 0; i < std::max(1, nthreads); i++) th.emplace_back(worker);
    for (auto & t : th) t.join();
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("done: packed %zu, already cached %zu, failed %zu, %.2f GB written, %.0f s\n",
           packed.load(), cached.load(), failed.load(), bytes.load() / 1e9, el);
    gguf_free(g); ggml_free(ctx);
    return failed ? 1 : 0;
}

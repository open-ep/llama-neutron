// ggml backend for the Renesas DRP-AI (RZ/V2H) with run-time weight hot-swap.
//
// GGML_DRPAI_TEMPLATES  dir of shape templates t_<K>x<N>_m<M>/ (deploy.{so,json,params} + tpl.txt), one per matrix shape
//                       and row count, compiled once with the DRP-AI Translator and reused by every model.
// GGML_DRPAI_MODEL      model package: one dir per GGUF weight (or <name>.c<i> chunks) holding
//                         weight.bin   packed int8 weights + per-channel requant parameters (DRP-AI weight.bin format)
//                         scales.bin   static input / output scale (float32 x2)
//                         sidecar.bin  CPU-side pieces: outlier input cols / output rows, output equalization
// All packed weights live in one pool at the top of the DRP-AI area.  Before a Run the template's AI-MAC parameter
// commands are repointed at the weight's pool slot and its drp_param scales are rewritten (a few hundred bytes).
// Per token the DRP sees two int8 rows: x_hi = q(x) and x_lo = (x - x_hi) * g; y = y_hi + y_lo / g.
#include "ggml-drpai.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "MeraDrpRuntimeWrapper.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <cstdarg>
#include <poll.h>
#include <dirent.h>
#include <set>
#include <linux/drpai.h>

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <map>
#include <chrono>
#include <atomic>
#include <thread>
#include <functional>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

// The runtime's Run() issues INADDR + START with arguments that are constant per object; we record them during the
// one warm-up Run() and replay them ourselves afterwards (same ioctls, but no per-KB output reads).
struct drpai_runargs { bool have = false; drpai_inout_t inaddr{}; drpai_data_t proc[DRPAI_INDEX_NUM]{}; };
static std::map<int, drpai_runargs> & drpai_recorded() { static std::map<int, drpai_runargs> m; return m; }
extern "C" __attribute__((visibility("default"))) int ioctl(int fd, unsigned long req, ...) {
    static int (*real)(int, unsigned long, ...) = nullptr;
    if (!real) real = (int (*)(int, unsigned long, ...)) dlsym(RTLD_NEXT, "ioctl");
    va_list ap; va_start(ap, req); void * arg = va_arg(ap, void *); va_end(ap);
    int r = real(fd, req, arg);
    if (r == 0 && arg) {
        if (req == DRPAI_PREPOST_INADDR) { auto & a = drpai_recorded()[fd]; memcpy(&a.inaddr, arg, sizeof(a.inaddr)); }
        else if (req == DRPAI_START) { auto & a = drpai_recorded()[fd]; memcpy(a.proc, arg, sizeof(a.proc)); a.have = true; }
    }
    return r;
}

static constexpr uint64_t DRPAI_ALIGN = 16ull << 20;          // runtime requires 16 MiB aligned object start
static constexpr uint32_t AIMAC_PARAM_OP = 0x50000600;       // command word followed by a weight / PPU pointer

struct drpai_tpl {
    int K = 0, N = 0, M = 0;
    uint64_t w_off = 0, w_len = 0, apc_off = 0, apc_len = 0, dp_off = 0, dp_len = 0, in_off = 0, out_off = 0;
    uint64_t base = 0;
    std::unique_ptr<MeraDrpRuntimeWrapper> rt;
    int fd = -1;                                              // the runtime's own fd for this object (holds the driver state)
    drpai_runargs args;                                        // recorded INADDR + START arguments
    uint64_t dout_off = 0, dout_len = 0;
    bool raw = false;
    std::vector<uint32_t> apc0;
    std::vector<uint8_t> dp0;
    const void * cur = nullptr;                               // image currently written into this template
};
struct drpai_image { drpai_tpl * t = nullptr; std::vector<uint32_t> apc; std::vector<uint8_t> dp; };
struct drpai_chunk {
    int K = 0, N = 0, ki = 0, ko = 0, row0 = 0;
    float g = 8.f, a_s = 1.f, out_s = 1.f;
    std::vector<int32_t> in_idx, out_idx;
    std::vector<float> W_in, W_out, r;                       // (N,ki) (ko,K) (N)
    std::vector<float> inv_r;                                // 1/r
    std::vector<ggml_fp16_t> W_inT16, W_out16;               // (ki,N) and (ko,K) in fp16: these stream from DRAM every token
    uint64_t pool = 0;
    drpai_image dec, pf;                                     // M=2 (one token) and M=PF (PF/2 tokens)
};
struct drpai_weight {
    std::vector<drpai_chunk> chunks; int K = 0, N = 0;
    bool fused = false;                                      // several GGUF tensors share this object (rows [row0, row0+n) each)
    const void * last_src = nullptr; int last_M = 0; long last_gen = -1;   // result cache for the fused members
    std::vector<float> last_out;                             // (M, N)
};
struct drpai_view { drpai_weight * w; int row0, nrows; };    // tensor->extra

struct drpai_ctx {
    std::string pkg, tpl_root;
    int fd = -1;
    uint64_t area = 0, area_size = 0, next_tpl = 0, pool_top = 0;
    std::map<std::tuple<int, int, int>, std::unique_ptr<drpai_tpl>> tpls;
    std::map<std::string, std::unique_ptr<drpai_weight>> weights;
    std::map<std::string, std::unique_ptr<drpai_view>> views;      // by GGUF tensor name
    long gen = 0;
    std::vector<float> xin, yout;
    std::vector<ggml_fp16_t> yraw;
    int pf = 256;
};
static drpai_ctx & drpai_global() { static drpai_ctx c; return c; }

// workers sleep on a condition variable between jobs; the caller runs slice 0 itself
#include <mutex>
#include <condition_variable>
struct drpai_pool {
    int n = 1;
    std::vector<std::thread> th;
    std::mutex mu; std::condition_variable cv;
    int gen = 0, done = 0; bool quit = false;
    std::function<void(int, int)> job;
    void start(int nthreads) {
        n = std::max(1, nthreads);
        for (int i = 1; i < n; i++) th.emplace_back([this, i] {
            int seen = 0;
            for (;;) {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen; lk.unlock();
                job(i, n);
                lk.lock(); if (++done == n - 1) cv.notify_all();
            }
        });
    }
    void run(const std::function<void(int, int)> & f, bool parallel) {
        if (n == 1 || !parallel) { f(0, 1); return; }
        { std::lock_guard<std::mutex> lk(mu); job = f; done = 0; gen++; }
        cv.notify_all();
        f(0, n);
        std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return done == n - 1; });
    }
    ~drpai_pool() { { std::lock_guard<std::mutex> lk(mu); quit = true; } cv.notify_all(); for (auto & t : th) t.join(); }
};
static drpai_pool & drpai_threads() { static drpai_pool p; return p; }

// GGML_DRPAI_PROF=1: per-phase time totals printed at exit
enum { P_SPLIT, P_SWITCH, P_SETIN, P_RUN, P_GETOUT, P_FINISH, P_GRAPH, P_N };
static const char * prof_name[P_N] = { "split", "switch", "set_input", "run", "get_output", "finish", "graph_total" };
static double prof_ms[P_N]; static long prof_runs, prof_graphs; static bool prof_on;
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void prof_dump() {
    if (!prof_on) return;
    fprintf(stderr, "drpai prof: %ld graphs, %ld DRP runs\n", prof_graphs, prof_runs);
    for (int i = 0; i < P_N; i++) fprintf(stderr, "  %-12s %10.1f ms total %8.3f ms/run\n", prof_name[i], prof_ms[i], prof_runs ? prof_ms[i] / prof_runs : 0);
}

static bool is_dir(const std::string & p) { struct stat st; return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode); }
static std::vector<char> read_file(const std::string & p) { std::ifstream s(p, std::ios::binary); return {std::istreambuf_iterator<char>(s), {}}; }
static bool dmem_write(drpai_ctx & c, uint64_t addr, const void * p, size_t n) {
    drpai_data_t d{addr, n};
    return ioctl(c.fd, DRPAI_ASSIGN, &d) == 0 && write(c.fd, p, n) == (ssize_t) n;
}
static bool dmem_read(drpai_ctx & c, uint64_t addr, void * p, size_t n) {
    drpai_data_t d{addr, n};
    return ioctl(c.fd, DRPAI_ASSIGN, &d) == 0 && read(c.fd, p, n) == (ssize_t) n;
}

static std::set<int> drpai_open_fds() {
    std::set<int> r; DIR * d = opendir("/proc/self/fd"); if (!d) return r;
    dirent * e; char buf[64];
    while ((e = readdir(d))) {
        int n = atoi(e->d_name); if (n < 3) continue;
        std::string p = std::string("/proc/self/fd/") + e->d_name;
        ssize_t k = readlink(p.c_str(), buf, 63);
        if (k > 0) { buf[k] = 0; if (!strcmp(buf, "/dev/drpai0")) r.insert(n); }
    }
    closedir(d); return r;
}

static drpai_tpl * drpai_get_tpl(drpai_ctx & ctx, int K, int N, int M) {
    auto key = std::make_tuple(K, N, M);
    auto it = ctx.tpls.find(key);
    if (it != ctx.tpls.end()) return it->second.get();
    std::string d = ctx.tpl_root + "/t_" + std::to_string(K) + "x" + std::to_string(N) + "_m" + std::to_string(M) + "/";
    auto t = std::make_unique<drpai_tpl>();
    FILE * f = fopen((d + "tpl.txt").c_str(), "r");
    if (!f) { GGML_LOG_ERROR("drpai: no template %s\n", d.c_str()); return nullptr; }
    unsigned long long v[31];
    int n = 0; while (n < 31 && fscanf(f, "%llu", &v[n]) == 1) n++;
    fclose(f);
    if (n < 11) { GGML_LOG_ERROR("drpai: bad %stpl.txt\n", d.c_str()); return nullptr; }
    t->K = v[0]; t->N = v[1]; t->M = v[2]; t->w_off = v[3]; t->w_len = v[4]; t->apc_off = v[5]; t->apc_len = v[6];
    t->dp_off = v[7]; t->dp_len = v[8]; t->in_off = v[9]; t->out_off = v[10];
    t->base = ctx.next_tpl;
    auto fds_before = drpai_open_fds();
    t->rt.reset(new MeraDrpRuntimeWrapper());
    if (!t->rt->LoadModel(d, t->base)) { GGML_LOG_ERROR("drpai: LoadModel %s failed\n", d.c_str()); return nullptr; }
    for (int f : drpai_open_fds()) if (!fds_before.count(f)) t->fd = f;
    if (n == 31 && t->fd >= 0 && !getenv("GGML_DRPAI_NO_RAW") && (M > 2 || getenv("GGML_DRPAI_RAW_DEC"))) {
        t->dout_off = v[11 + 2 * DRPAI_INDEX_OUTPUT]; t->dout_len = v[12 + 2 * DRPAI_INDEX_OUTPUT];
        std::vector<float> zero((size_t) M * K, 0.f);
        drpai_recorded().erase(t->fd);
        t->rt->SetInput(0, zero.data()); t->rt->Run();                 // warm-up: records this object's run arguments
        auto it = drpai_recorded().find(t->fd);
        if (it != drpai_recorded().end() && it->second.have) { t->args = it->second; t->raw = true; }
    }
    ctx.next_tpl = (t->rt->GetLastAddress() + DRPAI_ALIGN - 1) & ~(DRPAI_ALIGN - 1);
    if (ctx.next_tpl > ctx.pool_top) { GGML_LOG_ERROR("drpai: DRP-AI area full (templates)\n"); return nullptr; }
    t->apc0.resize(t->apc_len / 4); t->dp0.resize(t->dp_len);
    if (!dmem_read(ctx, t->base + t->apc_off, t->apc0.data(), t->apc_len) || !dmem_read(ctx, t->base + t->dp_off, t->dp0.data(), t->dp_len)) {
        GGML_LOG_ERROR("drpai: cannot read back template %s\n", d.c_str()); return nullptr;
    }
    GGML_LOG_INFO("drpai: template %dx%d m%d at 0x%llx fd %d%s\n", K, N, M, (unsigned long long) t->base, t->fd, t->raw ? " (raw run)" : "");
    drpai_tpl * p = t.get();
    ctx.tpls[key] = std::move(t);
    return p;
}

static bool drpai_make_image(drpai_ctx & ctx, drpai_chunk & c, int M, drpai_image & im) {
    im.t = drpai_get_tpl(ctx, c.K, c.N, M);
    if (!im.t) return false;
    drpai_tpl & t = *im.t;
    im.apc = t.apc0;
    int n = 0;
    for (size_t i = 0; i + 1 < im.apc.size(); i++)
        if (t.apc0[i] == AIMAC_PARAM_OP && t.apc0[i + 1] >= t.w_off && t.apc0[i + 1] < t.w_off + t.w_len) {
            im.apc[i + 1] = (uint32_t) (t.apc0[i + 1] - t.w_off + (c.pool - t.base)); n++;
        }
    if (!n) { GGML_LOG_ERROR("drpai: no weight pointers in template %dx%d m%d\n", c.K, c.N, M); return false; }
    im.dp = t.dp0;
    ggml_fp16_t a = ggml_fp32_to_fp16(c.a_s), o = ggml_fp32_to_fp16(c.out_s);
    memcpy(im.dp.data() + t.in_off, &a, 2); memcpy(im.dp.data() + t.out_off, &o, 2);
    return true;
}

static bool load_sidecar(drpai_chunk & c, const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4]; uint32_t v, K, N, ki, ko;
    bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "DRPL", 4) == 0 && fread(&v, 4, 1, f) && fread(&K, 4, 1, f) && fread(&N, 4, 1, f)
           && fread(&ki, 4, 1, f) && fread(&ko, 4, 1, f) && fread(&c.g, 4, 1, f) && fread(&c.a_s, 4, 1, f);
    if (ok) {
        c.K = K; c.N = N; c.ki = ki; c.ko = ko;
        c.in_idx.resize(ki); c.out_idx.resize(ko); c.W_in.resize((size_t) N * ki); c.W_out.resize((size_t) ko * K); c.r.resize(N);
        ok = fread(c.in_idx.data(), 4, ki, f) == ki && fread(c.out_idx.data(), 4, ko, f) == ko && fread(c.W_in.data(), 4, c.W_in.size(), f) == c.W_in.size()
          && fread(c.W_out.data(), 4, c.W_out.size(), f) == c.W_out.size() && fread(c.r.data(), 4, N, f) == N;
    }
    fclose(f);
    if (ok) {
        c.W_inT16.resize(c.W_in.size()); c.W_out16.resize(c.W_out.size()); c.inv_r.resize(N);
        for (uint32_t n = 0; n < N; n++) { c.inv_r[n] = 1.f / c.r[n]; for (uint32_t i = 0; i < ki; i++) c.W_inT16[(size_t) i * N + n] = ggml_fp32_to_fp16(c.W_in[(size_t) n * ki + i]); }
        for (size_t i = 0; i < c.W_out.size(); i++) c.W_out16[i] = ggml_fp32_to_fp16(c.W_out[i]);
        std::vector<float>().swap(c.W_in); std::vector<float>().swap(c.W_out);
    }
    return ok;
}

static std::vector<std::string> find_dirs(const std::string & pkg, const std::string & name) {
    std::vector<std::string> d;
    if (is_dir(pkg + "/" + name)) { d.push_back(pkg + "/" + name); return d; }
    for (int i = 0; ; i++) {
        std::string p = pkg + "/" + name + ".c" + std::to_string(i);
        if (!is_dir(p)) break;
        d.push_back(p);
    }
    return d;
}
static std::string pkg_name_for(const char * tname) {
    std::string n = tname;
    if (n == "token_embd.weight") return "output.weight";                  // tied lm_head
    return n;
}
// blk.<i>.attn_q.weight -> blk.<i>.attn_qkv.weight rows [0,2048) etc. when the package has the fused object
static bool fused_lookup(const std::string & pkg, const std::string & name, std::string & fname, int & row0, int & nrows) {
    static const struct { const char * part; const char * fused; } tab[] = { {"attn_q", "attn_qkv"}, {"attn_k", "attn_qkv"}, {"attn_v", "attn_qkv"}, {"ffn_gate", "ffn_gateup"}, {"ffn_up", "ffn_gateup"} };
    size_t a = name.find('.', 4); if (name.compare(0, 4, "blk.") != 0 || a == std::string::npos) return false;
    std::string part = name.substr(a + 1); if (part.size() < 8 || part.compare(part.size() - 7, 7, ".weight") != 0) return false;
    part = part.substr(0, part.size() - 7);
    for (auto & e : tab) if (part == e.part) {
        fname = name.substr(0, a + 1) + e.fused + ".weight";
        FILE * f = fopen((pkg + "/" + fname + "/parts.txt").c_str(), "r"); if (!f) return false;
        char pn[64]; int n, off = 0; bool found = false;
        while (fscanf(f, "%63s %d", pn, &n) == 2) { if (part == pn) { row0 = off; nrows = n; found = true; break; } off += n; }
        fclose(f); return found;
    }
    return false;
}

static bool drpai_has_weight(const struct ggml_tensor * t) {
    drpai_ctx & ctx = drpai_global();
    if (ctx.pkg.empty()) return false;
    static const char * skip = getenv("GGML_DRPAI_SKIP");              // comma-separated substrings of tensor names left on the CPU
    if (skip) { std::string sk = skip, nm = t->name; size_t a = 0; while (a <= sk.size()) { size_t b = sk.find(',', a); if (b == std::string::npos) b = sk.size(); if (b > a && nm.find(sk.substr(a, b - a)) != std::string::npos) return false; a = b + 1; } }
    static std::map<std::string, bool> known;
    std::string key = pkg_name_for(t->name);
    auto it = known.find(key);
    if (it != known.end()) return it->second;
    bool ok = false;
    std::string fname; int frow0 = 0, fnrows = 0;
    if (fused_lookup(ctx.pkg, key, fname, frow0, fnrows)) {
        FILE * f = fopen((ctx.pkg + "/" + fname + "/sidecar.bin").c_str(), "rb"); uint32_t h[6] = {0};
        if (f) { if (fread(h, 4, 6, f) == 6) ok = (long long) h[2] == t->ne[0] && fnrows == t->ne[1]; fclose(f); }
        known[key] = ok; return ok;
    }
    auto dirs = find_dirs(ctx.pkg, key);
    if (!dirs.empty()) {
        long long N = 0, K = 0;
        for (auto & d : dirs) {
            FILE * f = fopen((d + "/sidecar.bin").c_str(), "rb");
            if (!f) { N = -1; break; }
            uint32_t h[6] = {0};
            if (fread(h, 4, 6, f) == 6) { K = h[2]; N += h[3]; }
            fclose(f);
        }
        ok = K == t->ne[0] && N == t->ne[1];
        if (!ok && N > 0) GGML_LOG_WARN("drpai: %s shape mismatch pkg %lldx%lld vs tensor %lldx%lld\n", t->name, K, N, (long long) t->ne[0], (long long) t->ne[1]);
    }
    known[key] = ok;
    return ok;
}

static drpai_weight * drpai_load_weight(drpai_ctx & ctx, const std::string & key, bool fused) {
    auto it = ctx.weights.find(key);
    if (it != ctx.weights.end()) return it->second.get();
    auto w = std::make_unique<drpai_weight>(); w->fused = fused;
    int row0 = 0;
    for (auto & d : find_dirs(ctx.pkg, key)) {
        drpai_chunk c;
        if (!load_sidecar(c, d + "/sidecar.bin")) { GGML_LOG_ERROR("drpai: bad sidecar in %s\n", d.c_str()); return nullptr; }
        auto sc = read_file(d + "/scales.bin");
        auto wb = read_file(d + "/weight.bin");
        if (sc.size() != 8 || wb.empty()) { GGML_LOG_ERROR("drpai: missing scales.bin / weight.bin in %s\n", d.c_str()); return nullptr; }
        memcpy(&c.a_s, sc.data(), 4); memcpy(&c.out_s, sc.data() + 4, 4);
        uint64_t slot = (wb.size() + 4095) & ~4095ull;
        if (ctx.pool_top - slot < ctx.next_tpl) { GGML_LOG_ERROR("drpai: DRP-AI area full (weights)\n"); return nullptr; }
        ctx.pool_top -= slot;
        c.pool = ctx.pool_top;
        if (!dmem_write(ctx, c.pool, wb.data(), wb.size())) { GGML_LOG_ERROR("drpai: pool write failed for %s\n", d.c_str()); return nullptr; }
        if (!drpai_make_image(ctx, c, 2, c.dec)) return nullptr;
        if (!drpai_make_image(ctx, c, ctx.pf, c.pf)) c.pf.t = nullptr;   // decode-only if no prefill template
        c.row0 = row0; row0 += c.N;
        w->K = c.K; w->N = row0;
        w->chunks.push_back(std::move(c));
    }
    GGML_LOG_INFO("drpai: loaded %s (%d chunks, K=%d N=%d) pool 0x%llx\n", key.c_str(), (int) w->chunks.size(), w->K, w->N,
                  (unsigned long long) ctx.pool_top);
    drpai_weight * p = w.get();
    ctx.weights[key] = std::move(w);
    return p;
}
static drpai_view * drpai_get_weight(const struct ggml_tensor * t) {
    drpai_ctx & ctx = drpai_global();
    std::string key = pkg_name_for(t->name);
    auto it = ctx.views.find(key);
    if (it != ctx.views.end()) return it->second.get();
    std::string fname; int row0 = 0, nrows = 0; bool fused = fused_lookup(ctx.pkg, key, fname, row0, nrows);
    drpai_weight * w = drpai_load_weight(ctx, fused ? fname : key, fused);
    if (!w) return nullptr;
    auto v = std::make_unique<drpai_view>(); v->w = w; v->row0 = fused ? row0 : 0; v->nrows = fused ? nrows : w->N;
    drpai_view * p = v.get(); ctx.views[key] = std::move(v); return p;
}

static const float * drpai_run(drpai_ctx & ctx, drpai_image & im, const float * in, int rows, int N) {
    drpai_tpl & t = *im.t;
    double t0 = now_ms();
    if (t.cur != &im) {
        dmem_write(ctx, t.base + t.apc_off, im.apc.data(), t.apc_len);
        dmem_write(ctx, t.base + t.dp_off, im.dp.data(), t.dp_len);
        t.cur = &im;
    }
    double t1 = now_ms();
    t.rt->SetInput(0, in);
    double t2 = now_ms();
    int64_t n = (int64_t) rows * N;
    ctx.yout.resize(n);
    const ggml_fp16_t * p = nullptr; const void * p32 = nullptr;
    if (t.raw) {
        if (ioctl(t.fd, DRPAI_PREPOST_INADDR, &t.args.inaddr) < 0) { perror("drpai: INADDR"); GGML_ABORT("drpai INADDR failed"); }
        static bool want_freq = getenv("GGML_DRPAI_RAW_FREQ") != nullptr;
        if (want_freq) { uint32_t f = 1; ioctl(t.fd, DRPAI_SET_DRPAI_FREQ, &f); }
        if (ioctl(t.fd, DRPAI_START, t.args.proc) < 0) { perror("drpai: START"); GGML_ABORT("drpai START failed"); }
        pollfd pf{t.fd, POLLIN, 0};
        if (poll(&pf, 1, 10000) <= 0) GGML_ABORT("drpai: run timeout");
        drpai_status_t st{};
        if (ioctl(t.fd, DRPAI_GET_STATUS, &st) < 0 || st.err) { GGML_LOG_ERROR("drpai: status err %d\n", st.err); GGML_ABORT("drpai run error"); }
        GGML_ASSERT(t.dout_len >= (uint64_t) n * 2);
        ctx.yraw.resize(n);
        static bool own_fd = getenv("GGML_DRPAI_RAW_OWNFD") != nullptr;
        drpai_data_t od{t.base + t.dout_off, (uint64_t) n * 2};
        int rfd = own_fd ? t.fd : ctx.fd;
        if (ioctl(rfd, DRPAI_ASSIGN, &od) != 0 || read(rfd, ctx.yraw.data(), n * 2) != (ssize_t) (n * 2)) GGML_ABORT("drpai: output read failed");
        p = ctx.yraw.data();
    } else {
        t.rt->Run();
        auto o = t.rt->GetOutput(0);
        GGML_ASSERT(std::get<2>(o) == n);
        if (std::get<0>(o) == InOutDataType::FLOAT32) p32 = std::get<1>(o); else p = (const ggml_fp16_t *) std::get<1>(o);
    }
    double t3 = now_ms();
    if (p32) memcpy(ctx.yout.data(), p32, n * 4);
    else {
        float * q = ctx.yout.data(); int64_t j = 0;
#ifdef __aarch64__
        for (; j + 8 <= n; j += 8) {
            float16x8_t h = vreinterpretq_f16_u16(vld1q_u16((const uint16_t *) p + j));
            vst1q_f32(q + j, vcvt_f32_f16(vget_low_f16(h))); vst1q_f32(q + j + 4, vcvt_f32_f16(vget_high_f16(h)));
        }
#endif
        for (; j < n; j++) q[j] = ggml_fp16_to_fp32(p[j]);
    }
    if (prof_on) { double t4 = now_ms(); prof_ms[P_SWITCH] += t1 - t0; prof_ms[P_SETIN] += t2 - t1; prof_ms[P_RUN] += t3 - t2; prof_ms[P_GETOUT] += t4 - t3; prof_runs++; }
    return ctx.yout.data();
}

static void split_token(const drpai_chunk & c, const float * x, float * hi, float * lo) {
    const float inv = 1.f / c.a_s, a = c.a_s, g = c.g;
    int k = 0;
#ifdef __aarch64__
    const float32x4_t vinv = vdupq_n_f32(inv), va = vdupq_n_f32(a), vg = vdupq_n_f32(g), lim = vdupq_n_f32(127.f), nlim = vdupq_n_f32(-127.f);
    for (; k + 4 <= c.K; k += 4) {
        float32x4_t v = vld1q_f32(x + k);
        float32x4_t h = vmulq_f32(vminq_f32(lim, vmaxq_f32(nlim, vrndnq_f32(vmulq_f32(v, vinv)))), va);
        float32x4_t l = vmulq_f32(vminq_f32(lim, vmaxq_f32(nlim, vrndnq_f32(vmulq_f32(vmulq_f32(vsubq_f32(v, h), vg), vinv)))), va);
        vst1q_f32(hi + k, h); vst1q_f32(lo + k, l);
    }
#endif
    for (; k < c.K; k++) {
        float v = x[k];
        float h = std::fmin(127.f, std::fmax(-127.f, std::nearbyint(v * inv))) * a;
        float l = std::fmin(127.f, std::fmax(-127.f, std::nearbyint((v - h) * g * inv))) * a;
        hi[k] = h; lo[k] = l;
    }
    for (int i = 0; i < c.ki; i++) { hi[c.in_idx[i]] = 0; lo[c.in_idx[i]] = 0; }
}
static inline void axpy16(float * y, const ggml_fp16_t * x, float a, int n) {
    int i = 0;
#ifdef __aarch64__
    float32x4_t va = vdupq_n_f32(a);
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vreinterpretq_f16_u16(vld1q_u16((const uint16_t *) x + i));
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), vcvt_f32_f16(vget_low_f16(h)), va));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), vcvt_f32_f16(vget_high_f16(h)), va));
    }
#endif
    for (; i < n; i++) y[i] += a * ggml_fp16_to_fp32(x[i]);
}
static inline float dot16(const float * a, const ggml_fp16_t * b, int n) {
    int i = 0; float s = 0;
#ifdef __aarch64__
    float32x4_t acc = vdupq_n_f32(0), acc2 = vdupq_n_f32(0);
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vreinterpretq_f16_u16(vld1q_u16((const uint16_t *) b + i));
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vcvt_f32_f16(vget_low_f16(h)));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 4), vcvt_f32_f16(vget_high_f16(h)));
    }
    s = vaddvq_f32(vaddq_f32(acc, acc2));
#endif
    for (; i < n; i++) s += a[i] * ggml_fp16_to_fp32(b[i]);
    return s;
}
// finish `rows` tokens at once: xs[t] = token input (stride xs_nb bytes), y = DRP rows (2*t = hi, 2*t+1 = lo), out[t] (stride out_nb)
static void finish_slice(const drpai_chunk & c, int rows, const char * xs, size_t xs_nb, const float * y, char * outb, size_t out_nb, int n0, int n1, int t0, int t1) {
    const float ig = 1.f / c.g;
    const int N = c.N, K = c.K, BLK = 256;
    for (int t = t0; t < t1; t++) {
        const float * yh = y + (size_t) (2 * t) * N, * yl = yh + N; float * out = (float *) (outb + (size_t) t * out_nb);
        int n = n0;
#ifdef __aarch64__
        float32x4_t vig = vdupq_n_f32(ig);
        for (; n + 4 <= n1; n += 4) vst1q_f32(out + n, vmulq_f32(vfmaq_f32(vld1q_f32(yh + n), vld1q_f32(yl + n), vig), vld1q_f32(c.inv_r.data() + n)));
#endif
        for (; n < n1; n++) out[n] = (yh[n] + yl[n] * ig) * c.inv_r[n];
    }
    for (int nb = n0; nb < n1; nb += BLK) {
        int len = std::min(BLK, n1 - nb);
        for (int t = t0; t < t1; t++) {
            const float * x = (const float *) (xs + (size_t) t * xs_nb); float * out = (float *) (outb + (size_t) t * out_nb);
            for (int i = 0; i < c.ki; i++) { float xv = x[c.in_idx[i]]; if (xv != 0) axpy16(out + nb, c.W_inT16.data() + (size_t) i * N + nb, xv, len); }
        }
    }
    for (int j = 0; j < c.ko; j++) {
        int oi = c.out_idx[j];
        if (oi < n0 || oi >= n1) continue;
        const ggml_fp16_t * w = c.W_out16.data() + (size_t) j * K;
        for (int t = t0; t < t1; t++) ((float *) (outb + (size_t) t * out_nb))[oi] = dot16((const float *) (xs + (size_t) t * xs_nb), w, K);
    }
}
static void finish_tokens(const drpai_chunk & c, int rows, const char * xs, size_t xs_nb, const float * y, char * outb, size_t out_nb) {
    drpai_threads().run([&](int id, int nt) {
        if (rows >= nt) { int t0 = rows * id / nt, t1 = rows * (id + 1) / nt; finish_slice(c, rows, xs, xs_nb, y, outb, out_nb, 0, c.N, t0, t1); }
        else { int n0 = (c.N / 8 * id / nt) * 8, n1 = (c.N / 8 * (id + 1) / nt) * 8; if (id == nt - 1) n1 = c.N; finish_slice(c, rows, xs, xs_nb, y, outb, out_nb, n0, n1, 0, rows); }
    }, rows >= 8 || c.N >= 8192);
}
static void drpai_mul_mat_to(drpai_ctx & ctx, drpai_weight & w, const struct ggml_tensor * src1, char * outb, size_t out_nb);
static void drpai_mul_mat(drpai_ctx & ctx, drpai_view & v, struct ggml_tensor * dst) {
    const struct ggml_tensor * src1 = dst->src[1];
    drpai_weight & w = *v.w;
    const int M = (int) src1->ne[1];
    if (!w.fused) { drpai_mul_mat_to(ctx, w, src1, (char *) dst->data, dst->nb[1]); return; }
    if (w.last_src != src1->data || w.last_M != M || w.last_gen != ctx.gen) {       // first member of the group computes for all
        w.last_out.resize((size_t) M * w.N);
        drpai_mul_mat_to(ctx, w, src1, (char *) w.last_out.data(), (size_t) w.N * sizeof(float));
        w.last_src = src1->data; w.last_M = M; w.last_gen = ctx.gen;
    }
    for (int t = 0; t < M; t++) memcpy((char *) dst->data + (size_t) t * dst->nb[1], w.last_out.data() + (size_t) t * w.N + v.row0, (size_t) v.nrows * sizeof(float));
}
static void drpai_mul_mat_to(drpai_ctx & ctx, drpai_weight & w, const struct ggml_tensor * src1, char * outb, size_t out_nb) {
    const int K = w.K, M = (int) src1->ne[1];
    for (auto & c : w.chunks) {
        int t = 0;
        while (t < M) {
            bool use_pf = c.pf.t && M - t >= 8;
            int rows_tok = use_pf ? std::min(ctx.pf / 2, M - t) : 1;
            drpai_image & im = use_pf ? c.pf : c.dec;
            int rows = use_pf ? ctx.pf : 2;
            double ts = now_ms();
            ctx.xin.resize((size_t) rows * K);
            if (rows > 2 * rows_tok) memset(ctx.xin.data() + (size_t) 2 * rows_tok * K, 0, (size_t) (rows - 2 * rows_tok) * K * sizeof(float));
            drpai_threads().run([&](int id, int nt) {
                for (int i = id; i < rows_tok; i += nt) {
                    const float * x = (const float *) ((const char *) src1->data + (size_t) (t + i) * src1->nb[1]);
                    split_token(c, x, ctx.xin.data() + (size_t) (2 * i) * K, ctx.xin.data() + (size_t) (2 * i + 1) * K);
                }
            }, rows_tok >= 8);
            if (prof_on) prof_ms[P_SPLIT] += now_ms() - ts;
            const float * y = drpai_run(ctx, im, ctx.xin.data(), rows, c.N);
            double tf = now_ms();
            finish_tokens(c, rows_tok, (const char *) src1->data + (size_t) t * src1->nb[1], src1->nb[1], y,
                          outb + (size_t) t * out_nb + (size_t) c.row0 * sizeof(float), out_nb);
            if (prof_on) prof_ms[P_FINISH] += now_ms() - tf;
            t += rows_tok;
        }
    }
}

static void drpai_get_rows(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const auto * tt = ggml_get_type_traits(src0->type);
    const int64_t K = src0->ne[0];
    for (int64_t i = 0; i < ggml_nelements(src1); i++) {
        int32_t row = *(const int32_t *) ((const char *) src1->data + i * src1->nb[0]);
        const void * s = (const char *) src0->data + row * src0->nb[1];
        float * d = (float *) ((char *) dst->data + i * dst->nb[1]);
        if (src0->type == GGML_TYPE_F32) memcpy(d, s, K * 4); else tt->to_float(s, d, K);
    }
}

// ---- buffer (plain host memory, so the CPU backend can also read it) ----
static void drpai_buffer_free(ggml_backend_buffer_t buffer) { free(buffer->context); }
static void * drpai_buffer_get_base(ggml_backend_buffer_t buffer) { return buffer->context; }
static enum ggml_status drpai_buffer_init_tensor(ggml_backend_buffer_t, struct ggml_tensor * tensor) {
    tensor->extra = drpai_has_weight(tensor) ? drpai_get_weight(tensor) : nullptr;
    return GGML_STATUS_SUCCESS;
}
static void drpai_buffer_memset_tensor(ggml_backend_buffer_t, struct ggml_tensor * t, uint8_t v, size_t off, size_t n) { memset((char *) t->data + off, v, n); }
static void drpai_buffer_set_tensor(ggml_backend_buffer_t, struct ggml_tensor * t, const void * d, size_t off, size_t n) { memcpy((char *) t->data + off, d, n); }
static void drpai_buffer_get_tensor(ggml_backend_buffer_t, const struct ggml_tensor * t, void * d, size_t off, size_t n) { memcpy(d, (const char *) t->data + off, n); }
static bool drpai_buffer_cpy_tensor(ggml_backend_buffer_t, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) { memcpy(dst->data, src->data, ggml_nbytes(src)); return true; }
    return false;
}
static void drpai_buffer_clear(ggml_backend_buffer_t buffer, uint8_t v) { memset(buffer->context, v, buffer->size); }
static const struct ggml_backend_buffer_i drpai_buffer_i = {
    /* .free_buffer     = */ drpai_buffer_free,
    /* .get_base        = */ drpai_buffer_get_base,
    /* .init_tensor     = */ drpai_buffer_init_tensor,
    /* .memset_tensor   = */ drpai_buffer_memset_tensor,
    /* .set_tensor      = */ drpai_buffer_set_tensor,
    /* .get_tensor      = */ drpai_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ drpai_buffer_cpy_tensor,
    /* .clear           = */ drpai_buffer_clear,
    /* .reset           = */ NULL,
};
static const char * drpai_buft_get_name(ggml_backend_buffer_type_t) { return "DRPAI"; }
static ggml_backend_buffer_t drpai_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * p = nullptr;
    if (posix_memalign(&p, 64, size ? size : 64) != 0) return nullptr;
    return ggml_backend_buffer_init(buft, drpai_buffer_i, p, size);
}
static size_t drpai_buft_get_alignment(ggml_backend_buffer_type_t) { return 64; }
static bool drpai_buft_is_host(ggml_backend_buffer_type_t) { return true; }

ggml_backend_buffer_type_t ggml_backend_drpai_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name         = */ drpai_buft_get_name,
            /* .alloc_buffer     = */ drpai_buft_alloc_buffer,
            /* .get_alignment    = */ drpai_buft_get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL,
            /* .is_host          = */ drpai_buft_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_drpai_reg(), 0),
        /* .context = */ NULL,
    };
    return &buft;
}

// ---- backend ----
static const char * drpai_backend_get_name(ggml_backend_t) { return "DRPAI"; }
static void drpai_backend_free(ggml_backend_t backend) { delete backend; }
static enum ggml_status drpai_backend_graph_compute(ggml_backend_t, struct ggml_cgraph * cgraph) {
    drpai_ctx & ctx = drpai_global();
    ctx.gen++;
    double tg = now_ms();
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                drpai_view * v = (drpai_view *) node->src[0]->extra;
                if (!v) v = drpai_get_weight(node->src[0]);
                GGML_ASSERT(v && "drpai: no object for this weight");
                drpai_mul_mat(ctx, *v, node);
            } break;
            case GGML_OP_GET_ROWS: drpai_get_rows(node); break;
            case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: break;
            default: GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }
    if (prof_on) { prof_ms[P_GRAPH] += now_ms() - tg; prof_graphs++; }
    return GGML_STATUS_SUCCESS;
}
static struct ggml_backend_i drpai_backend_i = {
    /* .get_name                = */ drpai_backend_get_name,
    /* .free                    = */ drpai_backend_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ drpai_backend_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};
static ggml_guid_t drpai_guid(void) {
    static ggml_guid guid = { 0xd7, 0x9a, 0x11, 0x52, 0xc3, 0x0e, 0x4b, 0x8f, 0xa2, 0x61, 0x7d, 0x33, 0x90, 0x1e, 0x55, 0x06 };
    return &guid;
}
ggml_backend_t ggml_backend_drpai_init(void) {
    return new ggml_backend { drpai_guid(), drpai_backend_i, ggml_backend_reg_dev_get(ggml_backend_drpai_reg(), 0), nullptr };
}
bool ggml_backend_is_drpai(ggml_backend_t b) { return b && ggml_guid_matches(b->guid, drpai_guid()); }

// ---- device ----
static const char * drpai_dev_get_name(ggml_backend_dev_t) { return "DRPAI"; }
static const char * drpai_dev_get_description(ggml_backend_dev_t) { return "Renesas DRP-AI3 (RZ/V2H)"; }
static void drpai_dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = 0; *total = 0; }
static enum ggml_backend_dev_type drpai_dev_get_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }
static void drpai_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = drpai_dev_get_name(dev); props->description = drpai_dev_get_description(dev); props->type = drpai_dev_get_type(dev);
    drpai_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { /* async */ false, /* host_buffer */ false, /* buffer_from_host_ptr */ false, /* events */ false, /* mmap_support */ false };
}
static ggml_backend_t drpai_dev_init_backend(ggml_backend_dev_t, const char *) { return ggml_backend_drpai_init(); }
static ggml_backend_buffer_type_t drpai_dev_get_buffer_type(ggml_backend_dev_t) { return ggml_backend_drpai_buffer_type(); }
static bool drpai_dev_supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: return true;
        case GGML_OP_GET_ROWS:
            return op->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 && src0->ne[2] == 1 && drpai_has_weight(src0);
        case GGML_OP_MUL_MAT:
            return src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && src0->ne[2] == 1 && src0->ne[3] == 1
                && src1->ne[2] == 1 && src1->ne[3] == 1 && drpai_has_weight(src0);
        default: return false;
    }
}
static bool drpai_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return ggml_backend_buft_is_host(buft); }
static const struct ggml_backend_device_i drpai_device_i = {
    /* .get_name             = */ drpai_dev_get_name,
    /* .get_description      = */ drpai_dev_get_description,
    /* .get_memory           = */ drpai_dev_get_memory,
    /* .get_type             = */ drpai_dev_get_type,
    /* .get_props            = */ drpai_dev_get_props,
    /* .init_backend         = */ drpai_dev_init_backend,
    /* .get_buffer_type      = */ drpai_dev_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ drpai_dev_supports_op,
    /* .supports_buft        = */ drpai_dev_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---- reg ----
static const char * drpai_reg_get_name(ggml_backend_reg_t) { return "DRPAI"; }
static size_t drpai_reg_get_device_count(ggml_backend_reg_t) { return drpai_global().pkg.empty() ? 0 : 1; }
static ggml_backend_dev_t drpai_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = { drpai_device_i, reg, nullptr };
    return &dev;
}
static const struct ggml_backend_reg_i drpai_reg_i = { drpai_reg_get_name, drpai_reg_get_device_count, drpai_reg_get_device, NULL };
ggml_backend_reg_t ggml_backend_drpai_reg(void) {
    static bool init = false;
    static ggml_backend_reg reg = { GGML_BACKEND_API_VERSION, drpai_reg_i, nullptr };
    if (!init) {
        init = true;
        drpai_ctx & ctx = drpai_global();
        const char * p = getenv("GGML_DRPAI_MODEL");
        const char * t = getenv("GGML_DRPAI_TEMPLATES");
        prof_on = getenv("GGML_DRPAI_PROF") != nullptr;
        drpai_threads().start(getenv("GGML_DRPAI_THREADS") ? atoi(getenv("GGML_DRPAI_THREADS")) : 4);
        if (prof_on) atexit(prof_dump);
        if (p && t && is_dir(p) && is_dir(t)) {
            ctx.fd = open("/dev/drpai0", O_RDWR);
            drpai_data_t a{};
            if (ctx.fd < 0) perror("drpai: open /dev/drpai0");
            else if (ioctl(ctx.fd, DRPAI_GET_DRPAI_AREA, &a) < 0) perror("drpai: DRPAI_GET_DRPAI_AREA");
            else {
                ctx.area = a.address; ctx.area_size = a.size; ctx.next_tpl = a.address; ctx.pool_top = a.address + a.size;
                ctx.pkg = p; ctx.tpl_root = t;
            }
            fprintf(stderr, "drpai: package %s, templates %s, DRP-AI area 0x%llx + 0x%llx%s\n", p, t, (unsigned long long) ctx.area,
                    (unsigned long long) ctx.area_size, ctx.pkg.empty() ? " (unusable, backend disabled)" : "");
        } else if (p || t) {
            fprintf(stderr, "drpai: need both GGML_DRPAI_MODEL and GGML_DRPAI_TEMPLATES directories, backend disabled\n");
        }
    }
    return &reg;
}
GGML_BACKEND_DL_IMPL(ggml_backend_drpai_reg)

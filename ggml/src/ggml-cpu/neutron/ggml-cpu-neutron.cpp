// NXP Neutron NPU offload for ggml-cpu: Q4_0 weights live in Neutron CMA, MUL_MAT runs on the NPU.
// Two transports: "lib" (libNeutronDriver allocateBuffer()/matmul(), one buffer <= 3.5GB, 512MB slots) and
// "direct" (NEUTRON_DIRECT=1: our own dmabuf regions via the driver uapi, unlimited total, for 14B-class models).
// Modeled on ggml-cpu/repack.cpp's extra buffer type.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "traits.h"
#include "ggml-cpu-neutron.h"
#include "neutron_pack.h"
#include "neutron_uapi.h"
#include "neutron/NeutronDriver.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <poll.h>
#include <set>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace ggml::cpu::neutron {

static int env_int(const char * k, int d) { const char * v = getenv(k); return v ? atoi(v) : d; }
static bool verbose() { static int v = env_int("NEUTRON_VERBOSE", 0); return v; }

// ---- NEUTRON_PROF=1: accumulate per-phase time, print at exit ----
struct prof_t {
    bool on = env_int("NEUTRON_PROF", 0) != 0;
    double gather = 0, quant = 0, issue = 0, wait = 0, dequant = 0; long jobs = 0, calls = 0; double bytes = 0;
    ~prof_t() {
        if (!on || !jobs) return;
        fprintf(stderr, "ggml-neutron prof: %ld mul_mat calls, %ld jobs, %.1f MB weights/job avg\n", calls, jobs, bytes / jobs / 1e6);
        fprintf(stderr, "  gather %.1f ms  quant %.1f ms  issue %.1f ms  wait %.1f ms  dequant %.1f ms  (totals; per job: issue %.0f us, wait %.0f us)\n",
                gather * 1e3, quant * 1e3, issue * 1e3, wait * 1e3, dequant * 1e3, issue / jobs * 1e6, wait / jobs * 1e6);
    }
};
static prof_t & P() { static prof_t p; return p; }
static inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- packed-weight cache: packing is deterministic and slow (~0.45 s/MB on A55), so keep the result on disk ----
static uint64_t fnv1a64(const void * data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * p = (const uint8_t *)data;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, p + i, 8); h = (h ^ w) * 1099511628211ull; }
    for (; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
static std::string cache_path(const void * data, size_t size, int N, int K) {
    const char * dir = getenv("NEUTRON_CACHE_DIR");
    if (dir && !*dir) return "";                       // NEUTRON_CACHE_DIR="" disables the cache
    std::string d = dir ? dir : std::string(getenv("HOME") ? getenv("HOME") : "/tmp") + "/.cache/ggml-neutron";
    char buf[128];
    snprintf(buf, sizeof buf, "/%016llx_%dx%d.npk", (unsigned long long)fnv1a64(data, size), N, K);
    return d + buf;
}
struct npk_hdr { uint32_t magic, nchunks; };            // followed by nchunks x npk_chunk, then the blobs (64-aligned)
struct npk_chunk { uint32_t k, weight_len, compress_num, pad; uint64_t total; };
static constexpr uint32_t NPK_MAGIC = 0x314b504e; // "NPK1"

// ---- NPU memory: regions of contiguous CMA; weights bump-allocated from the bottom, A/Y scratch at the top ----
struct region {
    uint8_t * cpu = nullptr; uint64_t dma = 0; int fd = -1;   // fd < 0: slot of the libNeutronDriver buffer
    size_t size = 0, used = 0, scratch_off = 0;
    int lib_slot = 0;
};
struct dev {
    static constexpr size_t LIB_SLOT = 512ull << 20;
    bool direct = false;
    int dfd = -1;                 // our /dev/neutron0 handle (direct mode)
    size_t reserved = 0;          // scratch bytes per region
    size_t region_bytes = 0;      // direct mode: dmabuf size per region
    std::vector<region> regions;
    std::mutex mu;
    bool ready = false;

    bool init() {
        std::lock_guard<std::mutex> g(mu);
        if (ready) return true;
        direct = env_int("NEUTRON_DIRECT", 0) != 0;
        reserved = (size_t)env_int("NEUTRON_SCRATCH_MB", direct ? 256 : 192) << 20;
        if (!direct) {
            int n = std::max(1, std::min(env_int("NEUTRON_SLOTS", 4), 7));   // 8*512MB overflows the driver's u32 size
            void * p = nullptr;
            if (allocateBuffer((uint64_t)n * LIB_SLOT, &p, true) != ENONE || !p) {
                fprintf(stderr, "ggml-neutron: allocateBuffer(%d x 512MB) failed, NPU disabled\n", n); return false;
            }
            for (int i = 0; i < n; i++) {
                region r; r.cpu = (uint8_t *)p + i * LIB_SLOT; r.size = LIB_SLOT; r.scratch_off = LIB_SLOT - reserved; r.lib_slot = i;
                regions.push_back(r);
            }
            fprintf(stderr, "ggml-neutron: lib mode, CMA %d x 512MB at %p, scratch %zu MB/slot\n", n, p, reserved >> 20);
        } else {
            // libNeutronDriver still boots the firmware (FIRMWARE_LOAD needs a staging buffer of its own)
            void * p = nullptr;
            if (allocateBuffer(1ull << 20, &p, true) != ENONE || !p) { fprintf(stderr, "ggml-neutron: firmware init via allocateBuffer failed\n"); return false; }
            dfd = open("/dev/neutron0", O_RDWR | O_CLOEXEC);
            if (dfd < 0) { perror("ggml-neutron: open /dev/neutron0"); return false; }
            region_bytes = (size_t)std::min(env_int("NEUTRON_BUF_MB", 2048), 2048) << 20;   // firmware header offsets are signed 32-bit: keep every region under 2 GiB
            fprintf(stderr, "ggml-neutron: direct mode, regions of %zu MB, scratch %zu MB/region\n", region_bytes >> 20, reserved >> 20);
        }
        ready = true;
        return true;
    }
    bool new_region(size_t bytes) {   // direct mode only, mu held
        struct neutron_uapi_buffer_create bc = {}; bc.size = (uint32_t)bytes;
        int fd = ioctl(dfd, NEUTRON_IOCTL_BUFFER_CREATE, &bc);
        if (fd < 0) { fprintf(stderr, "ggml-neutron: BUFFER_CREATE %zu MB failed: %s\n", bytes >> 20, strerror(errno)); return false; }
        void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { perror("ggml-neutron: mmap dmabuf"); close(fd); return false; }
        region r; r.cpu = (uint8_t *)p; r.dma = bc.addr; r.fd = fd; r.size = bytes; r.scratch_off = bytes - reserved;
        regions.push_back(r);
        if (verbose()) fprintf(stderr, "ggml-neutron: region %zu: %zu MB dma %#llx fd %d\n", regions.size() - 1, bytes >> 20, (unsigned long long)r.dma, fd);
        return true;
    }
    uint8_t * carve(size_t size, int & ri) {
        std::lock_guard<std::mutex> g(mu);
        size = GGML_PAD(size, 64);
        for (size_t i = 0; i < regions.size(); i++) {
            if (regions[i].scratch_off - regions[i].used >= size) { uint8_t * p = regions[i].cpu + regions[i].used; regions[i].used += size; ri = (int)i; return p; }
        }
        if (direct) {
            // CMA may be fragmented / nearly full: try smaller regions before giving up
            for (size_t bytes = region_bytes; bytes >= size + reserved && bytes >= (256u << 20); bytes /= 2) {
                if (new_region(bytes)) { region & r = regions.back(); r.used = size; ri = (int)regions.size() - 1; return r.cpu; }
            }
        }
        return nullptr;
    }
    int region_of(const void * p) const {
        for (size_t i = 0; i < regions.size(); i++)
            if ((const uint8_t *)p >= regions[i].cpu && (const uint8_t *)p < regions[i].cpu + regions[i].size) return (int)i;
        return -1;
    }
    void sync(int ri, size_t off, size_t size, bool from_device) {
        const region & r = regions[ri];
        if (!direct) { clean_cache(r.cpu + off, (int)size); return; }
        struct neutron_uapi_cache_sync cs; cs.fd = r.fd; cs.offset = (uint32_t)off; cs.size = (uint32_t)size; cs.direction = from_device;
        if (ioctl(dfd, NEUTRON_IOCTL_CACHE_SYNC, &cs) < 0) perror("ggml-neutron: CACHE_SYNC");
    }
    // one matmul job. header/a/y are offsets inside region ri; header holds the 16 x u32 job descriptor
    bool job(int ri, size_t header, size_t a, size_t a_size, size_t y, size_t y_size) {
        const region & r = regions[ri];
        if (!direct) return matmul(r.cpu + header, 64, r.cpu + a, (int)a_size, r.cpu + y, (int)y_size, r.lib_slot) == ENONE;
        sync(ri, header, 64, false);
        // observed from libNeutronDriver 3.1.1: base_ddr = dma + 0x1ff00, job type KERNEL_LOAD, driver syncs in/out ranges itself
        static const uint64_t BASE_BIAS = (uint64_t)env_int("NEUTRON_BASE_BIAS", 0x1ff00);
        const uint64_t base = r.dma + BASE_BIAS;
        struct neutron_uapi_inference_args ia = {};
        ia.kernel_offset = (uint32_t)header;
        ia.base_ddr_l = (uint32_t)base; ia.base_ddr_h = (uint32_t)(base >> 32);
        ia.firmw_id = 0; ia.buf_id = r.fd;
        ia.input_offset = (uint32_t)a; ia.input_size = (uint32_t)a_size;
        ia.output_offset = (uint32_t)y; ia.output_size = (uint32_t)y_size;
        double t0 = P().on ? now_s() : 0;
        int jfd = ioctl(dfd, NEUTRON_IOCTL_KERNEL_LOAD, &ia);
        if (jfd < 0) { perror("ggml-neutron: KERNEL_LOAD"); return false; }
        double t1 = P().on ? now_s() : 0;
        struct pollfd pf = { jfd, POLLIN, 0 };
        int pr = poll(&pf, 1, 10000);
        close(jfd);
        if (P().on) { P().issue += t1 - t0; P().wait += now_s() - t1; }
        if (pr <= 0) { fprintf(stderr, "ggml-neutron: job poll %s\n", pr == 0 ? "timeout" : "error"); return false; }
        return true;
    }
};
static dev & D() { static dev d; return d; }
static int hook_clean_cache(const void * p, int n) {
    int ri = D().region_of(p);
    if (ri < 0) return clean_cache(p, n);
    D().sync(ri, (const uint8_t *)p - D().regions[ri].cpu, n, false);
    return 0;
}

// K values verified correct on firmware 3.1.1 (M=1..2048, deterministic). Other K are split into these chunks
// and the partial products summed on the CPU (firmware tiling bugs above 10752 and at several odd sizes).
static const std::set<int> & k_allowed() {
    static std::set<int> allow = [] {
        std::set<int> s = {896, 1024, 2048, 2560, 4096, 4864, 5120, 8192, 9216, 9728, 10240, 10752};
        if (const char * e = getenv("NEUTRON_K_ALLOW")) {
            s.clear();
            for (const char * p = e; *p; ) { s.insert(atoi(p)); while (*p && *p != ',') p++; if (*p) p++; }
        }
        return s;
    }();
    return allow;
}
static bool split_rec(int K, std::vector<int> & out, size_t depth, size_t limit) {
    if (K == 0) return true;
    if (depth == limit) return false;
    for (auto it = k_allowed().rbegin(); it != k_allowed().rend(); ++it) {
        if (*it > K) continue;
        out.push_back(*it);
        if (split_rec(K - *it, out, depth + 1, limit)) return true;
        out.pop_back();
    }
    return false;
}
static const std::vector<int> & k_chunks(int K) {   // minimal-count partition of K into allowed chunks; empty if impossible
    static std::map<int, std::vector<int>> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find(K);
    if (it != cache.end()) return it->second;
    std::vector<int> out;
    for (size_t limit = 1; limit <= 6 && !split_rec(K, out, 0, limit); limit++) out.clear();
    return cache[K] = out;
}

// ---- per-tensor state (tensor->extra) ----
class tensor_traits : public ggml::cpu::tensor_traits {
  public:
    std::vector<neutron_layout> L;   // one packed sub-weight per K chunk
    std::vector<int> kc;             // K chunk sizes
    int N = 0, K = 0, ri = 0;
    bool packed = false;

    bool work_size(int, const struct ggml_tensor *, size_t & size) override { size = 0; return true; }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) return false;
        if (params->ith != 0) return true;     // ponytail: NPU path is serial; other threads idle at the barrier
        run(op);
        return true;
    }

    void run(struct ggml_tensor * op) {
        const ggml_tensor * A = op->src[1];
        const int64_t M = A->ne[1];
        GGML_ASSERT(packed && A->ne[0] == K && op->ne[0] == N && op->ne[1] == M);
        if (P().on) P().calls++;
        const int kmax = *std::max_element(kc.begin(), kc.end());
        const region & R = D().regions[ri];
        const size_t sc_off = R.scratch_off;
        uint8_t * sc = R.cpu + sc_off;
        const size_t budget = D().reserved - 4096;
        int64_t max_m = std::min<int64_t>((int64_t)(budget / ((size_t)kmax + (size_t)N * 4)), 2048);  // 2048 = firmware M tile limit
        GGML_ASSERT(max_m >= 1);
        // ponytail: compute is serialized on thread 0, so one set of host scratch buffers is reused across calls
        static std::vector<float> in_scales, a_tmp, y_tmp;
        const int64_t mm = std::min<int64_t>(M, max_m);
        if (in_scales.size() < (size_t)mm) in_scales.resize(mm);
        if (kc.size() > 1) {
            if (a_tmp.size() < (size_t)mm * kmax) a_tmp.resize((size_t)mm * kmax);
            if (y_tmp.size() < (size_t)mm * N)    y_tmp.resize((size_t)mm * N);
        }

        for (int64_t m0 = 0; m0 < M; m0 += max_m) {
            const int64_t m = std::min(max_m, M - m0);
            uint8_t * a_npu = sc;
            const size_t y_off = sc_off + GGML_PAD((size_t)m * kmax, 64);
            int32_t * y_npu = (int32_t *)(R.cpu + y_off);
            const float * a_in = (const float *)((const uint8_t *)A->data + m0 * A->nb[1]);
            float * y_out = (float *)((uint8_t *)op->data + m0 * op->nb[1]);

            int k0 = 0;
            for (size_t c = 0; c < kc.size(); c++) {
                const int k = kc[c];
                const float * a_src = a_in;
                double tg = P().on ? now_s() : 0;
                if (kc.size() > 1) {   // gather the K slice into a contiguous m x k block
                    for (int64_t r = 0; r < m; r++) memcpy(&a_tmp[(size_t)r * k], a_in + (size_t)r * K + k0, (size_t)k * 4);
                    a_src = a_tmp.data();
                }
                const size_t a_size = (size_t)m * k, y_size = (size_t)m * N * 4;
                double tq = P().on ? now_s() : 0;
                QuantizeInput(a_src, a_npu, in_scales.data(), (uint32_t)m, (uint32_t)k);
                if (!D().direct) D().sync(ri, sc_off, a_size, false);   // direct mode: driver syncs the input range itself
                if (P().on) { P().gather += tq - tg; P().quant += now_s() - tq; P().jobs++; P().bytes += L[c].total; }

                neutron_layout & l = L[c];
                uint32_t * h = l.header;
                auto off = [&](const void * p) { return (uint32_t)((const uint8_t *)p - (const uint8_t *)h); };
                h[0]  = off(l.compress_len);
                h[1]  = off(l.decode_bias);
                h[2]  = (uint32_t)m;
                h[3]  = (uint32_t)k;
                h[4]  = (uint32_t)N | (1u << 18);
                h[5]  = off(a_npu);
                h[6]  = off(l.b_neutron);
                h[7]  = off(l.b_bias);
                h[8]  = off(l.b_factors);
                h[9]  = off(y_npu);
                h[10] = (uint32_t)GetMatmulTypeFlag(true, false);
                h[11] = 4;   // result bytes
                h[12] = 4;   // weight bits
                h[13] = 32;  // group size
                h[14] = off(l.decode_scale);
                h[15] = off(l.decode_input);

                bool ok = D().job(ri, (uint8_t *)h - R.cpu, sc_off, a_size, y_off, y_size);
                GGML_ASSERT(ok && "neutron matmul failed");
                static const bool ysync = env_int("NEUTRON_YSYNC", 0) != 0;   // driver syncs the output range itself
                if (!D().direct || ysync) D().sync(ri, y_off, y_size, true);
                double td = P().on ? now_s() : 0;
                if (kc.size() == 1) {
                    DequantizeOutput(y_npu, y_out, in_scales.data(), 1, (uint32_t)m, (uint32_t)N);
                } else {
                    DequantizeOutput(y_npu, y_tmp.data(), in_scales.data(), 1, (uint32_t)m, (uint32_t)N);
                    const size_t n = (size_t)m * N;
                    if (c == 0) memcpy(y_out, y_tmp.data(), n * 4);
                    else for (size_t i = 0; i < n; i++) y_out[i] += y_tmp[i];
                }
                if (P().on) P().dequant += now_s() - td;
                k0 += k;
            }
        }
    }
};

// ---- buffer ----
struct buffer_ctx { uint8_t * base; size_t size; int ri; };

static void buffer_free(ggml_backend_buffer_t buffer) { delete (buffer_ctx *)buffer->context; }  // ponytail: CMA carve is never returned
static void * buffer_get_base(ggml_backend_buffer_t buffer) { return ((buffer_ctx *)buffer->context)->base; }

static enum ggml_status buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    auto * t = new tensor_traits();   // ponytail: leaked with the model, same as repack's static traits
    t->K = (int)tensor->ne[0]; t->N = (int)tensor->ne[1];
    t->kc = k_chunks(t->K);
    t->ri = ((buffer_ctx *)buffer->context)->ri;
    tensor->extra = t;
    return GGML_STATUS_SUCCESS;
}

static void buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor) && tensor->type == GGML_TYPE_Q4_0);
    auto * t = (tensor_traits *)tensor->extra;
    const int K = t->K, N = t->N;
    GGML_ASSERT(!t->kc.empty());
    const size_t row = ggml_row_size(GGML_TYPE_Q4_0, K);
    uint8_t * out = (uint8_t *)tensor->data;
    size_t cap = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);
    t->L.resize(t->kc.size());
    const std::string cpath = cache_path(data, size, N, K);
    if (!cpath.empty()) {
        if (FILE * f = fopen(cpath.c_str(), "rb")) {
            npk_hdr h; std::vector<npk_chunk> ch;
            bool ok = fread(&h, sizeof h, 1, f) == 1 && h.magic == NPK_MAGIC && h.nchunks == t->kc.size();
            if (ok) { ch.resize(h.nchunks); ok = fread(ch.data(), sizeof(npk_chunk), h.nchunks, f) == h.nchunks; }
            uint8_t * o = out; size_t rem = cap;
            for (size_t c = 0; ok && c < ch.size(); c++) {
                ok = ch[c].k == (uint32_t)t->kc[c] && ch[c].total <= rem && fread(o, 1, ch[c].total, f) == ch[c].total;
                if (ok) { neutron_layout_from_blob(o, N, ch[c].k, ch[c].weight_len, ch[c].compress_num, &t->L[c]); GGML_ASSERT(t->L[c].total == ch[c].total);
                          const size_t used = GGML_PAD(ch[c].total, 64); o += used; rem -= used; }
            }
            fclose(f);
            if (ok) {
                t->packed = true;
                if (verbose()) fprintf(stderr, "ggml-neutron: loaded %s from cache %s\n", tensor->name, cpath.c_str());
                return;
            }
            fprintf(stderr, "ggml-neutron: bad cache file %s, repacking\n", cpath.c_str());
        }
    }
    int b0 = 0;
    for (size_t c = 0; c < t->kc.size(); c++) {
        const int k = t->kc[c], bc = k / QK4_0;
        // Q4_0 columns [b0*32, (b0+bc)*32) -> MatMulNBits nibble layout + fp32 scales
        std::vector<uint8_t> nib((size_t)N * bc * 16);
        std::vector<float> scales((size_t)N * bc);
        for (int n = 0; n < N; n++) {
            const block_q4_0 * blk = (const block_q4_0 *)((const uint8_t *)data + n * row) + b0;
            for (int b = 0; b < bc; b++) {
                scales[(size_t)n * bc + b] = GGML_FP16_TO_FP32(blk[b].d);
                uint8_t * o = &nib[((size_t)n * bc + b) * 16];
                for (int j = 0; j < 16; j++) {
                    auto el = [&](int e) -> uint8_t { return e < 16 ? (blk[b].qs[e] & 0x0F) : (blk[b].qs[e - 16] >> 4); };
                    o[j] = (uint8_t)(el(2 * j) | (el(2 * j + 1) << 4));
                }
            }
        }
        int rc = neutron_pack_nbits(nib.data(), scales.data(), N, k, out, cap, &t->L[c]);
        if (rc != 0) {
            fprintf(stderr, "ggml-neutron: packed size %zu exceeds remaining %zu for %s (%dx%d chunk %d)\n", t->L[c].total, cap, tensor->name, N, K, k);
            GGML_ABORT("neutron pack overflow");
        }
        const size_t used = GGML_PAD(t->L[c].total, 64);
        out += used; cap -= used; b0 += bc;
    }
    t->packed = true;
    if (!cpath.empty()) {
        std::string dir = cpath.substr(0, cpath.rfind('/'));
        for (size_t i = 1; i < dir.size(); i++) if (dir[i] == '/') mkdir(dir.substr(0, i).c_str(), 0755);   // mkdir -p
        mkdir(dir.c_str(), 0755);
        std::string tmp = cpath + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) fprintf(stderr, "ggml-neutron: cannot write cache %s\n", tmp.c_str());
        else {
            npk_hdr h = { NPK_MAGIC, (uint32_t)t->kc.size() };
            fwrite(&h, sizeof h, 1, f);
            for (size_t c = 0; c < t->kc.size(); c++) { npk_chunk ch = { (uint32_t)t->kc[c], t->L[c].weight_len, (uint32_t)t->L[c].compress_num, 0, t->L[c].total }; fwrite(&ch, sizeof ch, 1, f); }
            for (size_t c = 0; c < t->kc.size(); c++) fwrite(t->L[c].header, 1, t->L[c].total, f);
            fclose(f);
            rename(tmp.c_str(), cpath.c_str());
        }
    }
    if (verbose()) {
        size_t tot = 0; for (auto & l : t->L) tot += l.total;
        fprintf(stderr, "ggml-neutron: packed %s N=%d K=%d (%zu chunk%s) -> %zu bytes (%.3f B/param) region %d\n",
                tensor->name, N, K, t->kc.size(), t->kc.size() > 1 ? "s" : "", tot, (double)tot / ((double)N * K), t->ri);
    }
}

static void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * c = (buffer_ctx *)buffer->context; memset(c->base, value, c->size);
}

static const struct ggml_backend_buffer_i buffer_iface = {
    /* .free_buffer     = */ buffer_free,
    /* .get_base        = */ buffer_get_base,
    /* .init_tensor     = */ buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ buffer_clear,
    /* .reset           = */ nullptr,
};

// ---- buffer type ----
static const char * buft_get_name(ggml_backend_buffer_type_t) { return "CPU_NEUTRON"; }

static ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    int ri = -1;
    uint8_t * p = D().carve(size, ri);
    if (!p) { fprintf(stderr, "ggml-neutron: out of NPU memory for %zu MB buffer\n", size >> 20); return nullptr; }
    auto * ctx = new buffer_ctx{p, size, ri};
    return ggml_backend_buffer_init(buft, buffer_iface, ctx, size);
}
static size_t buft_get_alignment(ggml_backend_buffer_type_t) { return 64; }
static size_t buft_get_max_size(ggml_backend_buffer_type_t) {
    return (D().direct ? D().region_bytes : dev::LIB_SLOT) - D().reserved - (1 << 20);
}
static size_t buft_get_alloc_size(ggml_backend_buffer_type_t, const struct ggml_tensor * t) {
    if (t->type == GGML_TYPE_Q4_0 && ggml_n_dims(t) == 2) {
        size_t s = 0;
        for (int k : k_chunks((int)t->ne[0])) s += GGML_PAD(neutron_pack_bound((int)t->ne[1], k), 64);
        if (s) return s;
    }
    return ggml_nbytes(t);
}

class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) return false;
        const ggml_tensor * w = op->src[0], * a = op->src[1];
        if (!w->buffer || w->buffer->buft != ggml_backend_cpu_neutron_buffer_type()) return false;
        if (w->type != GGML_TYPE_Q4_0 || ggml_n_dims(w) != 2) return false;
        if (a->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || a->ne[2] != 1 || a->ne[3] != 1) return false;
        if (a->buffer && !ggml_backend_buft_is_host(a->buffer->buft)) return false;
        const int64_t K = w->ne[0], N = w->ne[1];
        return (N % 128 == 0) && (K % 32 == 0) && !k_chunks((int)K).empty();
    }
    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_neutron_buffer_type())
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        return nullptr;
    }
};

}  // namespace ggml::cpu::neutron

ggml_backend_buffer_type_t ggml_backend_cpu_neutron_buffer_type(void) {
    static ggml_backend_buffer_type_t buft = [] () -> ggml_backend_buffer_type_t {
        if (getenv("NEUTRON_DISABLE") || !ggml::cpu::neutron::D().init()) return nullptr;
        neutron_clean_cache = ggml::cpu::neutron::hook_clean_cache;
        static struct ggml_backend_buffer_type t = {
            /* .iface    = */ {
                /* .get_name         = */ ggml::cpu::neutron::buft_get_name,
                /* .alloc_buffer     = */ ggml::cpu::neutron::buft_alloc_buffer,
                /* .get_alignment    = */ ggml::cpu::neutron::buft_get_alignment,
                /* .get_max_size     = */ ggml::cpu::neutron::buft_get_max_size,
                /* .get_alloc_size   = */ ggml::cpu::neutron::buft_get_alloc_size,
                /* .is_host          = */ nullptr,
            },
            /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
            /* .context = */ new ggml::cpu::neutron::extra_buffer_type(),
        };
        return &t;
    }();
    return buft;
}

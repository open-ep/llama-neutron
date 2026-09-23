// laya-score: run a Laya typed decision (choice / score / noul) on an embedded board.
//
// Laya = ModernBERT encoder + a typed decision head. The encoder is ~93% of the compute and runs
// through llama.cpp (so the Neutron NPU accelerates its Q4_0 matmuls); the head is small and runs
// here as a ggml graph. Reference: laya/common.py build_sequence() and DecisionModel.forward().
//
//   sequence : [CLS] "<type> question: <instructions>" [SEP] [MASK]opt0 [MASK]opt1 ... [SEP] state [SEP]
//   forward  : h = encoder(ids); h += type_emb[qtype]
//              h = head_layer(h) x N          (pre-norm TransformerEncoderLayer, nhead = d/64, ff = 4d, ReLU)
//              logits = scorer(h[markers])    (LayerNorm -> Linear -> GELU -> Linear(->1))
//              p = softmax(logits / temperature[bucket])
#include "llama.h"
#include "common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <iostream>
#include <string>
#include <vector>

// llama.cpp/ggml are chatty while loading; keep stderr clean unless -v (errors always pass through).
static void quiet_log(ggml_log_level level, const char * text, void * user) {
    if (*(const bool *) user || level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

struct head_w {
    struct ggml_context * ctx = nullptr;
    gguf_context * gguf = nullptr;
    std::map<std::string, ggml_tensor *> t;
    int n_layers = 2;
    int max_len = 1024, head_max_len = 256;
    std::vector<float> temp_by_type{1, 1, 1};
    std::map<std::string, float> temp_bucket;

    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) { fprintf(stderr, "laya: missing head tensor %s\n", n.c_str()); exit(1); }
        return it->second;
    }
    bool load(const char * path) {
        gguf_init_params p = { /*no_alloc*/ false, /*ctx*/ &ctx };
        gguf = gguf_init_from_file(path, p);
        if (!gguf) { fprintf(stderr, "laya: cannot read %s\n", path); return false; }
        for (int64_t i = 0; i < gguf_get_n_tensors(gguf); i++) {
            const char * n = gguf_get_tensor_name(gguf, i);
            t[n] = ggml_get_tensor(ctx, n);
        }
        auto key = [&](const char * k) { return gguf_find_key(gguf, k); };
        if (key("laya.head_layers")   >= 0) n_layers     = gguf_get_val_u32(gguf, key("laya.head_layers"));
        if (key("laya.max_len")       >= 0) max_len      = gguf_get_val_u32(gguf, key("laya.max_len"));
        if (key("laya.head_max_len")  >= 0) head_max_len = gguf_get_val_u32(gguf, key("laya.head_max_len"));
        if (int64_t k = key("laya.temperature"); k >= 0) {
            const float * a = (const float *) gguf_get_arr_data(gguf, k);
            temp_by_type.assign(a, a + gguf_get_arr_n(gguf, k));
        }
        int64_t kk = key("laya.temp_bucket_keys"), kv = key("laya.temp_bucket_vals");
        if (kk >= 0 && kv >= 0) {
            const float * v = (const float *) gguf_get_arr_data(gguf, kv);
            for (size_t i = 0; i < gguf_get_arr_n(gguf, kk); i++) temp_bucket[gguf_get_arr_str(gguf, kk, i)] = v[i];
        }
        return true;
    }
};

// laya/common.py temp_bucket()
static std::string temp_bucket_name(int qtype, int k) {
    const char * n = qtype == 0 ? "choice" : qtype == 1 ? "score" : "noul";
    if (qtype == 2) return std::string(n) + ":2";
    if (k <= 2)  return std::string(n) + ":2";
    if (k <= 5)  return std::string(n) + ":3-5";
    if (k <= 10) return std::string(n) + ":6-10";
    return std::string(n) + ":11+";
}

// one pre-norm nn.TransformerEncoderLayer: x = x + attn(norm1(x)); x = x + ff(norm2(x)), ff = W2(relu(W1 x))
static ggml_tensor * head_layer(ggml_context * c, ggml_cgraph * gf, const head_w & H, int li,
                                ggml_tensor * x, int n_embd, int n_head, int T) {
    const std::string p = "head.layers." + std::to_string(li) + ".";
    const int hd = n_embd / n_head;
    ggml_tensor * n1 = ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), H.get(p + "norm1.weight")), H.get(p + "norm1.bias"));
    // fused qkv: in_proj_weight is (3*d, d) laid out [q; k; v]
    ggml_tensor * qkv = ggml_add(c, ggml_mul_mat(c, H.get(p + "self_attn.in_proj_weight"), n1), H.get(p + "self_attn.in_proj_bias"));
    ggml_tensor * q = ggml_cont(c, ggml_view_2d(c, qkv, n_embd, T, qkv->nb[1], 0));
    ggml_tensor * k = ggml_cont(c, ggml_view_2d(c, qkv, n_embd, T, qkv->nb[1], 1 * n_embd * sizeof(float)));
    ggml_tensor * v = ggml_cont(c, ggml_view_2d(c, qkv, n_embd, T, qkv->nb[1], 2 * n_embd * sizeof(float)));
    q = ggml_permute(c, ggml_reshape_3d(c, q, hd, n_head, T), 0, 2, 1, 3);   // (hd, T, nh)
    k = ggml_permute(c, ggml_reshape_3d(c, k, hd, n_head, T), 0, 2, 1, 3);
    v = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, hd, n_head, T), 1, 2, 0, 3));  // (T, hd, nh)
    ggml_tensor * kq = ggml_mul_mat(c, ggml_cont(c, k), ggml_cont(c, q));    // (T, T, nh)
    kq = ggml_soft_max_ext(c, kq, nullptr, 1.0f / sqrtf((float) hd), 0.0f);
    ggml_tensor * kqv = ggml_mul_mat(c, v, kq);                              // (hd, T, nh)
    ggml_tensor * a = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), n_embd, T);
    a = ggml_add(c, ggml_mul_mat(c, H.get(p + "self_attn.out_proj.weight"), a), H.get(p + "self_attn.out_proj.bias"));
    x = ggml_add(c, x, a);
    ggml_tensor * n2 = ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), H.get(p + "norm2.weight")), H.get(p + "norm2.bias"));
    ggml_tensor * f = ggml_add(c, ggml_mul_mat(c, H.get(p + "linear1.weight"), n2), H.get(p + "linear1.bias"));
    f = ggml_relu(c, f);                                                     // PyTorch default activation
    f = ggml_add(c, ggml_mul_mat(c, H.get(p + "linear2.weight"), f), H.get(p + "linear2.bias"));
    return ggml_add(c, x, f);
    GGML_UNUSED(gf);
}

int main(int argc, char ** argv) {
    std::string model, headp, state, instructions;
    std::vector<std::string> opts;
    std::string qtype_s = "choice";
    std::string tokens_csv, markers_csv;
    bool serve = false;
    int nthreads = 4;
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (a == "-m"     && i + 1 < argc) model        = nx();
        else if (a == "--head" && i + 1 < argc) headp        = nx();
        else if (a == "-s"     && i + 1 < argc) state        = nx();
        else if (a == "-i"     && i + 1 < argc) instructions = nx();
        else if (a == "--opt"  && i + 1 < argc) opts.push_back(nx());
        else if (a == "--type" && i + 1 < argc) qtype_s      = nx();
        else if (a == "-t"     && i + 1 < argc) nthreads     = atoi(nx().c_str());
        else if (a == "--tokens"  && i + 1 < argc) tokens_csv  = nx();
        else if (a == "--markers" && i + 1 < argc) markers_csv = nx();
        else if (a == "--serve")                  serve       = true;
        else if (a == "-v")                     verbose      = true;
        else { fprintf(stderr,
            "usage: laya-score -m encoder.gguf --head laya-head.gguf -s \"<state>\" -i \"<instructions>\"\n"
            "                  --opt A --opt B [--type choice|score|noul] [-t N] [-v]\n"
            "   pre-tokenized: --tokens id,id,... --markers i,j,...  (for models whose GGUF vocab\n"
            "                  cannot tokenize, e.g. the Metaspace-BPE multilingual encoder)\n"
            "   --serve      load once, then read \"<qtype> <markers csv> <tokens csv>\" per line on\n"
            "                stdin and print one result JSON per line (model load is ~25 s, so a UI\n"
            "                must keep this resident instead of spawning it per request)\n"); return 1; }
    }
    if (model.empty() || headp.empty()) { fprintf(stderr, "missing -m/--head\n"); return 1; }
    const bool pretok = !tokens_csv.empty() || serve;
    if (!pretok && instructions.empty()) { fprintf(stderr, "need -i (or --tokens/--markers)\n"); return 1; }
    if (pretok && !serve && markers_csv.empty()) { fprintf(stderr, "--tokens requires --markers\n"); return 1; }
    int qtype = qtype_s == "choice" ? 0 : qtype_s == "score" ? 1 : 2;
    if (qtype == 2 && opts.empty()) opts = { "false: no, the statement does not hold", "true: yes, the statement holds" };
    if (!pretok && opts.empty()) { fprintf(stderr, "need --opt\n"); return 1; }

    llama_log_set(quiet_log, &verbose);
    ggml_log_set(quiet_log, &verbose);

    head_w H;
    if (!H.load(headp.c_str())) return 1;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * lm = llama_model_load_from_file(model.c_str(), mp);
    if (!lm) { fprintf(stderr, "laya: cannot load %s\n", model.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(lm);
    const int n_embd = llama_model_n_embd(lm);
    const int n_head = n_embd / 64;

    std::vector<llama_token> ids;
    std::vector<int> markers;
    std::string request;
    auto parse_csv = [](const std::string & s, auto & out) {
        for (size_t i = 0; i < s.size(); ) {
            size_t j = s.find(',', i);
            out.push_back(atoi(s.substr(i, j == std::string::npos ? j : j - i).c_str()));
            if (j == std::string::npos) break;
            i = j + 1;
        }
    };
    if (serve) {
        printf("{\"ready\": true}\n"); fflush(stdout);
        if (!std::getline(std::cin, request)) return 0;
        size_t a1 = request.find(' '), a2 = request.find(' ', a1 + 1);
        if (a1 == std::string::npos || a2 == std::string::npos) { fprintf(stderr, "bad request\n"); return 1; }
        qtype_s = request.substr(0, a1);
        markers_csv = request.substr(a1 + 1, a2 - a1 - 1);
        tokens_csv = request.substr(a2 + 1);
    }
    if (pretok) {
        ids.clear(); markers.clear();
        parse_csv(tokens_csv, ids);
        parse_csv(markers_csv, markers);
        if (opts.empty()) for (size_t i = 0; i < markers.size(); i++) opts.push_back("option" + std::to_string(i));
        goto have_sequence;
    }

    // ---- build_sequence(): [CLS] "<type> question: <ins>" [SEP] [MASK]opt.. [SEP] state [SEP]
    {
    auto tok = [&](const std::string & s) { return common_tokenize(vocab, s, false, false); };
    const llama_token CLS = llama_vocab_bos(vocab), SEP = llama_vocab_eos(vocab), MASK = llama_vocab_mask(vocab);
    std::vector<llama_token> head_ids = tok(qtype_s + " question: " + instructions);
    std::vector<std::vector<llama_token>> opt_ids;
    int used, budget, room;
    for (auto & o : opts) {
        std::vector<llama_token> v{ MASK };
        auto x = tok(" " + o);
        if (x.size() > 48) x.resize(48);
        v.insert(v.end(), x.begin(), x.end());
        opt_ids.push_back(v);
    }
    used = 0; for (auto & o : opt_ids) used += o.size();
    budget = H.head_max_len - used;
    if (budget < 16) {
        int per = std::max(4, (H.head_max_len - 16) / std::max<int>(1, opt_ids.size()));
        for (auto & o : opt_ids) if ((int) o.size() > per) o.resize(per);
        used = 0; for (auto & o : opt_ids) used += o.size();
        budget = H.head_max_len - used;
    }
    if ((int) head_ids.size() > std::max(8, budget)) head_ids.resize(std::max(8, budget));

    ids.assign(1, CLS);
    ids.insert(ids.end(), head_ids.begin(), head_ids.end());
    ids.push_back(SEP);
    for (auto & o : opt_ids) { markers.push_back(ids.size()); ids.insert(ids.end(), o.begin(), o.end()); }
    ids.push_back(SEP);
    room = std::max(0, H.max_len - (int) ids.size() - 1);
    auto st = tok(state);
    if ((int) st.size() > room) st.resize(room);
    ids.insert(ids.end(), st.begin(), st.end());
    ids.push_back(SEP);
    if ((int) ids.size() > H.max_len) ids.resize(H.max_len);
    }
have_sequence:
    // The encoder context is sized once for the model's max_len so that --serve can reuse it;
    // each request only decodes its own T tokens.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = H.max_len;
    cp.n_threads = cp.n_threads_batch = nthreads;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;     // we need every token, not a pooled vector
    llama_context * lc = llama_init_from_model(lm, cp);
    if (!lc) { fprintf(stderr, "laya: cannot create context\n"); return 1; }

    for (;;) {
        const int T = ids.size();
        for (int m : markers) if (m >= T) { fprintf(stderr, "laya: options do not fit in head_max_len\n"); return 1; }
        if (verbose) { fprintf(stderr, "laya: %d tokens, %zu options, markers at", T, markers.size());
                       for (int m : markers) fprintf(stderr, " %d", m); fprintf(stderr, "\n"); }

        // ---- encoder (this is what the NPU accelerates) ----
        llama_memory_clear(llama_get_memory(lc), true);
        llama_batch batch = llama_batch_init(T, 0, 1);
        for (int i = 0; i < T; i++) {
            batch.token[i] = ids[i]; batch.pos[i] = i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = true;
        }
        batch.n_tokens = T;
        if (llama_decode(lc, batch) != 0) { fprintf(stderr, "laya: encoder decode failed\n"); return 1; }

        // ---- head as a ggml graph ----
        ggml_init_params ip = { (size_t) 512 * 1024 * 1024, nullptr, false };
        ggml_context * c = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(c, 8192, false);
        ggml_tensor * h = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_embd, T);
        ggml_set_input(h);
        ggml_tensor * te = ggml_view_1d(c, H.get("type_emb.weight"), n_embd, (size_t) qtype * n_embd * sizeof(float));
        ggml_tensor * x = ggml_add(c, h, ggml_repeat(c, te, h));
        for (int l = 0; l < H.n_layers; l++) x = head_layer(c, gf, H, l, x, n_embd, n_head, T);
        ggml_tensor * mk = ggml_new_tensor_1d(c, GGML_TYPE_I32, markers.size());
        ggml_set_input(mk);
        ggml_tensor * m = ggml_get_rows(c, x, mk);                                   // (d, K)
        m = ggml_add(c, ggml_mul(c, ggml_norm(c, m, 1e-5f), H.get("scorer.0.weight")), H.get("scorer.0.bias"));
        m = ggml_add(c, ggml_mul_mat(c, H.get("scorer.1.weight"), m), H.get("scorer.1.bias"));
        m = ggml_gelu(c, m);
        ggml_tensor * logits = ggml_add(c, ggml_mul_mat(c, H.get("scorer.3.weight"), m), H.get("scorer.3.bias"));
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);

        memcpy(h->data, llama_get_embeddings(lc), (size_t) n_embd * T * sizeof(float));
        for (size_t i = 0; i < markers.size(); i++) ((int32_t *) mk->data)[i] = markers[i];
        ggml_graph_compute_with_ctx(c, gf, nthreads);

        // ---- temperature + softmax ----
        const int K = markers.size();
        std::string bucket = temp_bucket_name(qtype, K);
        float Temp = H.temp_bucket.count(bucket) ? H.temp_bucket[bucket]
                   : (qtype < (int) H.temp_by_type.size() ? H.temp_by_type[qtype] : 1.0f);
        std::vector<float> z(K), p(K);
        for (int i = 0; i < K; i++) z[i] = ((float *) logits->data)[i] / Temp;
        float mx = z[0]; for (float v : z) mx = std::max(mx, v);
        float sum = 0; for (int i = 0; i < K; i++) { p[i] = expf(z[i] - mx); sum += p[i]; }
        int best = 0; for (int i = 0; i < K; i++) { p[i] /= sum; if (p[i] > p[best]) best = i; }

        if (serve) {
            printf("{\"tokens\": %d, \"temperature\": %.6f, \"choice_index\": %d, \"probabilities\": [", T, Temp, best);
            for (int i = 0; i < K; i++) printf("%s%.6f", i ? ", " : "", p[i]);
            printf("]}\n"); fflush(stdout);
        } else {
            printf("{\n  \"type\": \"%s\",\n  \"tokens\": %d,\n  \"temperature_bucket\": \"%s\",\n  \"temperature\": %.6f,\n",
                   qtype_s.c_str(), T, bucket.c_str(), Temp);
            printf("  \"choice\": \"%s\",\n  \"logits\": [", opts[best].c_str());
            for (int i = 0; i < K; i++) printf("%s%.6f", i ? ", " : "", ((float *) logits->data)[i]);
            printf("],\n  \"probabilities\": [");
            for (int i = 0; i < K; i++) printf("%s%.6f", i ? ", " : "", p[i]);
            printf("],\n  \"options\": [");
            for (int i = 0; i < K; i++) printf("%s\"%s\"", i ? ", " : "", opts[i].c_str());
            printf("]\n}\n");
        }
        ggml_free(c); llama_batch_free(batch);

        if (!serve) break;
        if (!std::getline(std::cin, request)) break;
        size_t a1 = request.find(' '), a2 = request.find(' ', a1 + 1);
        if (a1 == std::string::npos || a2 == std::string::npos) { fprintf(stderr, "laya: bad request\n"); continue; }
        qtype_s = request.substr(0, a1);
        qtype   = qtype_s == "choice" ? 0 : qtype_s == "score" ? 1 : 2;
        markers_csv = request.substr(a1 + 1, a2 - a1 - 1);
        tokens_csv  = request.substr(a2 + 1);
        ids.clear(); markers.clear();
        parse_csv(tokens_csv, ids);
        parse_csv(markers_csv, markers);
    }

    llama_free(lc); llama_model_free(lm); llama_backend_free();
    return 0;
}

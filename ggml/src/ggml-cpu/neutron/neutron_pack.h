// Extracted from NXP ORT Neutron EP (onnxruntime/core/providers/neutron/ops/{common,matmul_nbits}.{h,cc}), MIT license, Copyright NXP
#pragma once
// Copyright 2026 NXP



typedef struct {
    int* len;
    int* code;
    int* data_len;
    int size;
    int capacity;
} CodeTable;

typedef struct {
    uint8_t* compressed;
    int compressed_capacity;
    CodeTable code_table;
    size_t length;
} CoreCompResult;

typedef struct {
    int channelDensity;
    int lineDensity;
    int numNeutrons;
    bool bPingPong;
    int divisions;
} TilingResult;

typedef struct {
    int8_t* data;
    size_t  size;
    size_t  cap;
} Dyn8;
typedef struct {
    int32_t* data;
    size_t   size;
    size_t   cap;
} Dyn32;

typedef struct {
    Dyn8* Bpacked;
    Dyn32* lengths;
    TilingResult tilingInfo;
} PrepackOut;

typedef struct {
    bool rearrange;
    bool miniWeights;
    int weightBits;
    int groupSize;
    bool useDecodeBias;
    bool compress;
    int numMacs;
    int numNeutrons;
    int tcmSize;
    int numBanks;
} PrepackCfg;

#define IDX(i,j,cols) ( ((size_t)(i) * (size_t)(cols)) + (size_t)(j) )

#ifdef __cplusplus
extern "C" {
#endif

void dyn8_append(Dyn8* b, const int8_t* src, size_t n);
void dyn8_free(Dyn8* b);
void dyn32_push(Dyn32* b, int32_t v);
void dyn32_free(Dyn32* b);

void init_code_table(CodeTable* ct, int capacity);
void free_code_table(CodeTable* ct);
void copy_code_table(CodeTable* dest, const CodeTable* src);
int8_t find_min_int8(const int8_t* arr, int size);
void free_core_comp_result(CoreCompResult* result);
void extract_patterned_rows(const int8_t* matrix,
                        size_t rows,
                        size_t cols,
                        size_t start,
                        size_t step,
                        size_t length,
                        size_t count,
                        int8_t **out_ptr,
                        size_t *out_len);
void fetch_unp_organize(const int8_t* packed, size_t packedN, int rowsB, int colsB,
                        int channelDensity, int MACs, int weightBits, int numNeutrons,
                        int8_t** outOrganized, size_t* outOrganizedN);
void weight_packer(const int8_t* B, int rowsB, int colsB,
                   int channelDensity, int MACs, int weightBits,
                   int8_t** outPacked, size_t* outPackedN);
void fetch_unp_organize(const int8_t* packed, size_t packedN, int rowsB, int colsB,
                        int channelDensity, int MACs, int weightBits, int numNeutrons,
                        int8_t** outOrganized, size_t* outOrganizedN);
TilingResult tiling_solver(int numTokens, int embeddings_in, int embeddings_out,
              int resNumBytes, int MACS, int neutrons,
              int tcm_size, int tcm_banks, int weightBits,
              bool decodeWeights, bool useDecodeBias, int groupSize);
CoreCompResult compress_weight_tensor(
    uintptr_t data_ptr,
    int data_size,
    CodeTable code_table,
    int num_decomp,
    int word_size,
    int buffer_size,
    int packet_size);
int* CompressWeightTensorGrouped(
    int8_t* data,
    int data_size,
    int channelDensity,
    int numColsA,
    int weightBits,
    int num_decomp,
    int word_size,
    int buffer_size,
    int packet_size,
    bool compress,
    int8_t** outCompressed,
    size_t* outCompressedN,
    int* outListSize);

void PrePackWeight(const int8_t* B, int rb, int cb, PrepackCfg* pckCfg, PrepackOut* pckOut);

#ifdef __cplusplus
}
#endif

// helpers from matmul_nbits.cc / common.cc (host side, plain C++)
uint32_t ScaleToNeutron(float scale_data);
int32_t GetMatmulTypeFlag(bool packed, bool signedData);
void CalculateDecodeData(const uint8_t *B, const float *scalesData, int16_t *decodeScales, int8_t * decodeBias, uint32_t *bFactors, int32_t *bRowSum, int rowsB, int blocksPerCol, int blockSize);
void QuantizeInput(const float *in, uint8_t* out, float *scales, uint32_t a_rows, uint32_t a_cols);
void DequantizeOutput(const int32_t *in, float* out, float* scales, uint32_t a_batch, uint32_t a_rows, uint32_t b_rows);

// ---- ggml-side pack API (added) ----
#ifndef ALIGN16_SIZE
#define ALIGN16_SIZE(x) ((((size_t)(x)) + 15) & ~((size_t)15))
#endif
struct neutron_layout {
    uint32_t* header; int8_t* decode_bias; int16_t* decode_scale; int32_t* b_bias; uint32_t* b_factors;
    int8_t* b_neutron; int32_t* compress_len; uint8_t* decode_input;
    size_t total; uint32_t weight_len; int32_t compress_num;
};
// upper bound of packed bytes for an N x K 4-bit block-32 weight (used for buffer allocation before packing)
size_t neutron_pack_bound(int N, int K);
// B_nibbles: MatMulNBits layout [N][K/32][16] (elem 2j low nibble, 2j+1 high), scales fp32 [N*K/32]
// packs into out (CMA memory, all sub-buffers cache-cleaned). returns 0 ok, -1 if out_cap too small (L->total = needed)
int neutron_pack_nbits(const uint8_t* B_nibbles, const float* scales, int N, int K, uint8_t* out, size_t out_cap, neutron_layout* L);
// rebuild layout pointers over an already-packed blob (e.g. loaded from a cache file) and cache-clean it
void neutron_layout_from_blob(uint8_t* out, int N, int K, uint32_t weight_len, int32_t compress_num, neutron_layout* L);

// cache-clean hook used by the pack functions (default: libNeutronDriver clean_cache)
extern int (*neutron_clean_cache)(const void *, int);

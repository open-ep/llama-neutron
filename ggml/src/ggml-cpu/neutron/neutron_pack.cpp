// Extracted from NXP ORT Neutron EP, MIT license, Copyright 2025-2026 NXP. Host-side pack/quant helpers; no ORT dependency.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <arm_neon.h>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "neutron_pack.h"
#include "neutron/NeutronDriver.h"
#ifdef __cplusplus
extern "C" {
#endif
/* compress_weight_tensor_grouped function */
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
    int* outListSize) {

    const int max_output_size = data_size * 2;
    int8_t* compressed_stream = (int8_t*)malloc(max_output_size);
    *outCompressed = compressed_stream;

    int unit_size_outer = channelDensity * numColsA * weightBits / 8;
    int unit_size = unit_size_outer;
    int splits_per_channelC = 1;

    if (compress) {
        int splits_per_channelC_tmp = (int)ceil(unit_size_outer / (256.0 * 1024));
        splits_per_channelC = 1;
        while (splits_per_channelC < splits_per_channelC_tmp) {
            splits_per_channelC *= 2;
        }
        unit_size = unit_size_outer / splits_per_channelC;
    }

    int cycles = unit_size_outer / unit_size;
    bool compressed_one = false;
    int iters = data_size / unit_size;
    int compressed_offset = 0;

    int* len_compressed_stream_list = (int*)malloc(iters * sizeof(int));
    *outListSize = iters;

    CodeTable current_code_table;
    init_code_table(&current_code_table, 0);

    for (int iter = 0; iter < iters; iter++) {
        if (iter > 0 && iter % cycles == 0) {
            if (!compressed_one) {
                /* replace_last_m_with_sum */
                if (cycles > 0 && cycles <= iter) {
                    int sum = 0;
                    for (int i = iter - cycles; i < iter; i++) {
                        sum += len_compressed_stream_list[i];
                    }
                    len_compressed_stream_list[iter - 1] = sum;
                }
            }
            compressed_one = false;
        }

        const int8_t* current_data_ptr = data + iter * unit_size;

        /* Find minimum */
        int8_t min_val = find_min_int8(current_data_ptr, unit_size);

        if (min_val > -128 && compress) {
            compressed_one = true;

            uintptr_t current_data_addr = (uintptr_t)current_data_ptr;
            CoreCompResult result = compress_weight_tensor(current_data_addr, unit_size, current_code_table,
                                                 num_decomp, word_size, buffer_size, packet_size);

            // NOTE: compress_weight_tensor takes CodeTable by value (shallow copy).
            // When num_decomp > 1, it internally frees the code_table's pointers via
            // free_code_table(&code_table). Since those pointers are shared with
            // current_code_table, we must NOT call free_code_table(&current_code_table)
            // here — that would be a double-free. Instead, reset pointers to NULL.
            current_code_table.code = NULL;
            current_code_table.len = NULL;
            current_code_table.data_len = NULL;
            current_code_table.size = 0;
            current_code_table.capacity = 0;
            copy_code_table(&current_code_table, &result.code_table);

            int8_t* current_compressed = (int8_t*)malloc(unit_size * 2);
            int current_compressed_offset = 0;

            /* Process code words */
            for (int i = 0; i < current_code_table.size; i++) {
                int code = current_code_table.code[i];
                int len = current_code_table.len[i];
                current_compressed[current_compressed_offset++] = (int8_t)(code << (8 - len));
            }

            /* Process code lengths */
            for (int i = 0; i < current_code_table.size; i++) {
                int len = current_code_table.len[i];
                current_compressed[current_compressed_offset++] = (int8_t)(((1 << len) - 1) << (8 - len));
            }

            /* Add compressed data */
            for (size_t i = 0; i < result.length; i++) {
                uint8_t val = result.compressed[i];
                current_compressed[current_compressed_offset++] = val > 127 ?
                    (int8_t)(val - 256) : (int8_t)val;
            }

            /* Add padding */
            int padding_needed = (16 - current_compressed_offset % 16) % 16;
            for (int i = 0; i < padding_needed; i++) {
                current_compressed[current_compressed_offset++] = 0;
            }

            /* Copy to main stream */
            memcpy(compressed_stream + compressed_offset, current_compressed, current_compressed_offset);
            compressed_offset += current_compressed_offset;

            len_compressed_stream_list[iter] = current_compressed_offset;

            free(current_compressed);
            free_core_comp_result(&result);

        } else {
            /* No compression - just copy data */
            memcpy(compressed_stream + compressed_offset, current_data_ptr, unit_size);
            compressed_offset += unit_size;

            free_code_table(&current_code_table);
            init_code_table(&current_code_table, 0);
            len_compressed_stream_list[iter] = unit_size;
        }
    }

    free_code_table(&current_code_table);
    *outCompressedN = compressed_offset;

    return len_compressed_stream_list;
}

// Prepack
void PrePackWeight(const int8_t* B, int rb, int cb, PrepackCfg* pckCfg, PrepackOut* pckOut)
{
    int ca = cb;
    int weightBits = pckCfg->weightBits;
    int groupSize = pckCfg->groupSize;
    bool useDecodeBias = pckCfg->useDecodeBias;
    bool rearrange = pckCfg->rearrange;
    bool miniWeights = pckCfg->miniWeights;
    bool compress = pckCfg->compress;
    int numMacs = pckCfg->numMacs;
    int numNeutrons = pckCfg->numNeutrons;
    int tcmSize = pckCfg->tcmSize;
    int numBanks = pckCfg->numBanks;
    TilingResult t = tiling_solver(1, ca, rb, 4, numMacs, numNeutrons, tcmSize, numBanks, weightBits, (groupSize>0), useDecodeBias, groupSize);
    pckOut->tilingInfo = t;
    int MACs = numMacs;

    int cd = t.channelDensity, nn = t.numNeutrons, divisions = t.divisions;
    Dyn8* B_stream = pckOut->Bpacked;
    Dyn32* lengths_all = pckOut->lengths;

    if (rearrange) {
        // REARRANGE
        for (int rows = 0; rows < rb; rows += cd*nn) {
            for (int cols = 0; cols < ca; cols += ca/divisions) {
                int rowsB = (cd*nn < rb - rows) ? cd*nn : rb - rows;
                int colsB = (ca/divisions < ca - cols) ? ca/divisions : ca - cols;
                // slice
                size_t sliceN = (size_t)rowsB * colsB;
                int8_t* slice = (int8_t*)malloc(sliceN);
                for (int r = 0; r < rowsB; ++r)
                    for (int c = 0; c < colsB; ++c)
                        slice[IDX(r,c,colsB)] = B[IDX(rows+r, cols+c, ca)];

                // pack -> organize -> compress
                int8_t *packed=NULL, *organized=NULL, *compressed=NULL;
                size_t packedN=0, organizedN=0, compressedN=0;

                // [VICTOR] Converted to C
                weight_packer(slice, rowsB, colsB, cd, MACs, weightBits, &packed, &packedN);

                // [VICTOR] Converted to C
                fetch_unp_organize(packed, packedN, rowsB, colsB, cd, MACs, weightBits, nn, &organized, &organizedN);
                int outListSize;
                int32_t* lenList = CompressWeightTensorGrouped(organized, organizedN, cd*nn, colsB,
                                               weightBits,
                                               16,8,96,32,
                                               compress, &compressed, &compressedN, &outListSize);


                // [VICTOR] Converted to C
                dyn8_append(B_stream, compressed, compressedN);
                for (int i = 0; i < outListSize; ++i) dyn32_push(lengths_all, lenList[i]);

                free(slice); free(packed); free(organized); free(compressed);
            }
        }
    } else {
        // NO REARRANGE
        if (miniWeights) {
            fprintf(stderr, "error. Cannot have miniweights without rearrange\n");
            return;
        }
        for (int rows = 0; rows < rb; rows += cd*nn) {
                const int8_t* slice = B + rows * ca;
                for (int div_count = 0; div_count < divisions; div_count ++) {
                    // pack -> organize -> compress
                    int8_t *packed=NULL, *organized=NULL, *compressed=NULL;
                    size_t packedN=0, organizedN=0, compressedN=0;

                    extract_patterned_rows(slice, rb, ca, div_count*cd/divisions,
                        cd, cd/divisions, nn, &packed, &packedN);
                    fetch_unp_organize(packed, packedN, cd*nn/divisions, ca, cd/divisions, MACs, weightBits, nn, &organized, &organizedN);
                    int outListSize;
                    int32_t* lenList = CompressWeightTensorGrouped(organized, organizedN, cd*nn/divisions, ca,
                                weightBits,
                            16,8,96,32,
                                compress, &compressed, &compressedN, &outListSize);

                    dyn8_append(B_stream, compressed, compressedN);
                    for (int i = 0; i < outListSize; ++i) dyn32_push(lengths_all, lenList[i]);

                    free(packed); free(organized); free(compressed);
                }
        }
    }
}
#ifdef __cplusplus
}
#endif

uint32_t ScaleToNeutron(float scale_data) {
  float *scale_ptr = &scale_data;
  uint32_t u32 = *(uint32_t*) (scale_ptr);
  uint32_t scaler = (u32 >>8) & 0x7fff ; // extract mantissa (15bits)
  int8_t exp_tmp = (u32 >> 23) & 0xff; // extract exponent

  // Add hidden bit or zero out (if zero or subnormal)
  scaler = (exp_tmp==0) ? 0 :  scaler | 0x8000;
  // We subtract FP32 offset as well as 16bit growth of our scaler
  // (126 is power of -1 so mantissa is in range 0.5 to 1, 126 + 16=142,
  // where 16 is the factor we multiply by in scaler)
  exp_tmp = -(exp_tmp -142);
  // Ensure that we don't exceed available shift bits
  // (note that this step could, in theory be skipped if this never happens.
  // Not sure if we can take the chance)
  int8_t exp = (exp_tmp>63) ? 63 : exp_tmp;
  // Merge scaler and downshift factor into the Neutron 32bit scaler format
  // (16bit scaler in LSB and then 6bits of downshift)
  scaler = (exp<<16) | scaler;

  return scaler;
}

int32_t
GetMatmulTypeFlag(bool packed, bool signedData) {
    int32_t type = 0;
    if (packed && signedData) {
        type = 2;
    } else if (packed && !signedData) {
        type = 1;
    } else if (!packed && signedData) {
        type = -2;
    } else {
        type = -1;
    }
    return type;
}

double DecimalToFixedPoint(double number, int integer_bits = 10, int fraction_bits = 6) {
  bool sign = false;
  if (number < 0) {
    sign = true;
    number = -number;
  }

  if (number > (1 << integer_bits) - 1) {
    number = (1 << integer_bits) - 1;
  }

  // Split integer and fractional parts
  int integer_part = static_cast<int>(number);
  double fractional_part = number - integer_part;

  bool first_bit_obtained = (integer_part > 0);
  int bits_obtained = first_bit_obtained ? static_cast<int>(log2(integer_part)) + 1 : 0;
  int bits_left = integer_bits - bits_obtained;
  int shift = 0;
  int fixed_point_scale = integer_part;

  if (first_bit_obtained) {
    fraction_bits = bits_left;
  } else {
    while (fractional_part < 0.5 && shift < (1 << fraction_bits) - 1) {
      fractional_part *= 2;
      shift++;
    }
 }

  // Convert fractional part to binary
  for (int i = 0; i < fraction_bits; i++) {
    if (shift >= (1 << fraction_bits) - 1) {
      break;
    }
    fractional_part *= 2;
    int bit = static_cast<int>(fractional_part);
    fixed_point_scale = (fixed_point_scale * 2) | bit;
    shift++;
    fractional_part -= bit;
    if (fractional_part == 0) {
      break;
    }
  }
  int scale_10bit = fixed_point_scale & 0b1111111111;
  int shift_6bit = shift & 0b111111;
  return (scale_10bit * pow(2, -shift_6bit)) * (sign ? -1 : 1);
}

// Number must be non-negative
int16_t DecimalToNeutron(double number, int integer_bits = 10, int fraction_bits = 6) {
  if (number > (1 << integer_bits) - 1) {
    number = (1 << integer_bits) - 1;
  }

  int integer_part = static_cast<int>(number);
  double fractional_part = number - integer_part;

  bool first_bit_obtained = (integer_part > 0);
  int bits_obtained = (integer_part > 0) ? static_cast<int>(log2(integer_part)) + 1 : 0;
  int bits_left = integer_bits - bits_obtained;

  int shift = 0;
  int fixed_point_scale = integer_part & 0b1111111111;

  if (first_bit_obtained) {
    fraction_bits = bits_left;
  } else {
    while (fractional_part > 0 && fractional_part < 0.5 && shift < (1 << fraction_bits) - 1) {
      fractional_part *= 2;
      shift++;
    }
  }

  for (int i = 0; i < fraction_bits; i++) {
    if (shift >= (1 << fraction_bits) - 1) {
      break;
    }
    fractional_part *= 2;
    int bit = static_cast<int>(fractional_part);
    fixed_point_scale = (fixed_point_scale << 1) | bit;
    shift++;
    fractional_part -= bit;
    if (fractional_part == 0) {
      break;
    }
  }

  int scale_10bit = fixed_point_scale & 0b1111111111;
  int shift_6bit = shift & 0b111111;

  int16_t result = static_cast<int16_t>((shift_6bit << 10) | scale_10bit);
  return static_cast<int16_t>(result - 65536 * (shift_6bit >= 32));
}

template <typename T>
void OrganizeDecodeData(const T* decodeData, T* output, void* tempBuffer,
                        int rowsB, int colsB, int channelDensity, int numNeutrons,
                        int groupSize, int divisions, int weightBits = 4, int MACs = 16) {
    T* reorganized = static_cast<T*>(tempBuffer);
    int counter = 0;

    for (int i = 0; i < rowsB; i += channelDensity * numNeutrons) {
        for (int division = 0; division < divisions; ++division) {
            for (int neutron = 0; neutron < numNeutrons; ++neutron) {
                for (int j = 0; j < (colsB / groupSize / divisions); ++j) {
                    for (int row = 0; row < channelDensity; ++row) {
                        int row_index = i + row + neutron * channelDensity;
                        int col_index = j + division * (colsB / groupSize / divisions);
                        int flat_index = row_index * (colsB / groupSize) + col_index;

                        reorganized[counter] = decodeData[flat_index];
                        ++counter;
                    }
                }
            }
        }
    }

    int da = 0;
    int sa = 0;

    int dstStride = channelDensity * colsB / groupSize / divisions;
    int inner_cnt = std::min(dstStride, (int)(8 * 1024 / sizeof(T)));

    int iters = dstStride / inner_cnt;
    int stride = dstStride - inner_cnt;
    int repeats = rowsB / channelDensity / numNeutrons;

    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int division = 0; division < divisions; ++division) {
            for (int iter = 0; iter < iters; ++iter) {
                int da_save = da;
                for (int idx = 0; idx < numNeutrons; ++idx) {
                    for (int jdx = 0; jdx < inner_cnt; ++jdx) {
                        output[sa++] = reorganized[da++];
                    }
                    da += stride;
                }
                da = da_save + inner_cnt;
            }
            da = da - inner_cnt * iters;
            da += channelDensity * numNeutrons * colsB / groupSize / divisions;
        }
    }
}

void CalculateDecodeData(const uint8_t *B, const float *scalesData, int16_t *decodeScales,
                         int8_t * decodeBias, uint32_t *bFactors, int32_t *bRowSum, int rowsB,
                         int blocksPerCol,int blockSize) {
  for (int i = 0; i < rowsB; i ++) {
      float maxScale = std::abs(scalesData[i * blocksPerCol]);
      for (int m = 1; m < blocksPerCol; m ++) {
        float data = scalesData[i * blocksPerCol + m];
        if (std::abs(data) > maxScale) {
          maxScale = std::abs(data);
        }
      }

      float channelScale = maxScale / 128 * 8;
      float sum = 0;
      for (int j = 0; j < blocksPerCol; j ++) {
        float groupScale = scalesData[i * blocksPerCol + j];
        float scale = groupScale / channelScale;
        if (scale < 0) {
          decodeBias[i * blocksPerCol + j] = 1;
        } else {
          decodeBias[i * blocksPerCol + j] = 0;
        }
        auto fixedPointScale = DecimalToNeutron(DecimalToFixedPoint(std::abs(scale)));
        decodeScales[i * blocksPerCol + j] = fixedPointScale;

        //Caculate weights row sum
        for (int m = 0; m < blockSize; m ++) {
          int idx = i * blocksPerCol * blockSize + j * blockSize + m;
          uint8_t value = B[idx / 2];
          if (idx % 2 == 0) {
            value = (value & 0x0F);
          } else {
            value = ((value >> 4) & 0x0F);
          }

          float temp = ((float)value - 8) * DecimalToFixedPoint(scale);
          sum = sum + (int8_t)std::clamp((int)std::floor(temp + 0.5), -128, 127);
        }
      }

      bRowSum[i] = (int32_t)sum;
      bFactors[i] = ScaleToNeutron(channelScale);
  }
}

//Only 4 bit support
void PackWeight(const uint8_t *B, const float *scalesData, int8_t *packedWeights,
                int rowsB, int colsB, int blocksPerCol, int blockSize, int channelDensity,
                int row_stride, int weightBits = 4, int MACs = 16) {
  int cell_pointer = 0;
  int bit_pointer = 0;

  for (int j = 0; j < rowsB; j += channelDensity) {
    cell_pointer = std::ceil(channelDensity * colsB * weightBits / 8.0) * (j / channelDensity);
    bit_pointer = 0;

    for (int i = 0; i < colsB; i += MACs) {
      for (int m = 0; m < channelDensity; ++m) {
        for (int k = 0; k < MACs; ++k) {
          uint8_t value = B[(j + m) * row_stride / 2  + (i + k) / 2];
          if ((i + k) % 2 == 0) {
            value = (value & 0x0F);
          } else {
            value = ((value >> 4) & 0x0F);
          }
          if (scalesData[(j + m) * blocksPerCol + (i + k) / blockSize] < 0) {
            value = 7 - value;
          } else {
            value = value - 8;
          }
          uint8_t extracted_bits = value & ((1 << weightBits) - 1);
          if (8 - bit_pointer >= weightBits) {
            packedWeights[cell_pointer] |= extracted_bits << bit_pointer;
            bit_pointer += weightBits;
            cell_pointer += bit_pointer / 8;
            bit_pointer %= 8;
          } else {
            int fitting_bits = 8 - bit_pointer;
            int remaining_bits = weightBits - fitting_bits;
            uint8_t rem_extracted_bits = (value >> fitting_bits) & ((1 << remaining_bits) - 1);

            packedWeights[cell_pointer] |= extracted_bits << bit_pointer;
            cell_pointer ++;
            bit_pointer = 0;
            packedWeights[cell_pointer] |= rem_extracted_bits;
            bit_pointer += remaining_bits;
          }
        }
      }
    }
  }

  return;
}

void CMatMul(const float* A, const float* B, float* C, int M, int K, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

void ReQuantizeWeight(const uint8_t *in, int8_t *out, const float *scalesData, uint32_t *bFactors, int rowsB,
                int blocksPerCol, int blockSize) {
  for (int i = 0; i < rowsB; i ++) {
    float maxScale = scalesData[i * blocksPerCol];
    for (int m = 1; m < blocksPerCol; m ++) {
      float data = scalesData[i * blocksPerCol + m];
      if (std::abs(data) > std::abs(maxScale)) {
        maxScale = data;
      }
    }

    float channelScale;
    if (maxScale > 0) {
      channelScale = maxScale / 128 * 8;
    } else {
      channelScale = maxScale / 128 * 8 * -1.0;
    }

    for (int j = 0; j < blocksPerCol; j ++) {
      float groupScale = scalesData[i * blocksPerCol + j];
      for (int m = 0; m < blockSize; m ++) {
        int idx = i * blocksPerCol * blockSize + j * blockSize + m;
        uint8_t value = in[idx / 2];
        if (idx % 2 == 0) {
          value = (value & 0x0F);
        } else {
          value = ((value >> 4) & 0x0F);
        }

        float temp = ((float)value - 8) * groupScale / channelScale;
        out[idx] = (int8_t)std::clamp((int)std::round(temp), -128, 127);
      }
    }

    bFactors[i] = ScaleToNeutron(channelScale);
  }
}

void DequantizeWeight(const uint8_t *in, float *out, const float *scalesData,
                      int rowsB, int blocksPerCol, int blockSize) {
  for (int i = 0; i < rowsB; i ++) {
    for (int j = 0; j < blocksPerCol; j ++) {
      for (int m = 0; m < blockSize; m ++) {
        int idx = i * blocksPerCol * blockSize + j * blockSize + m;

        uint8_t value = in[idx / 2];
        if (idx % 2 == 0) {
          value = (value & 0x0F);
        } else {
          value = ((value >> 4) & 0x0F);
        }

        int outidx = (j * blockSize + m) * rowsB + i;
        out[outidx] = ((float)value - 8) * scalesData[i * blocksPerCol + j];
      }
    }
  }
}

static inline int32x4_t roundq_s32_f32(float32x4_t x) {
    float32x4_t half = vdupq_n_f32(0.5f);

    uint32x4_t is_positive = vcgeq_f32(x, vdupq_n_f32(0.0f));

    float32x4_t offset = vbslq_f32(is_positive, half, vnegq_f32(half));
    float32x4_t adjusted = vaddq_f32(x, offset);

    return vcvtq_s32_f32(adjusted);
}

// Function to perform per-tensor quantization with zero-point = 0
void  QuantizeInput(const float *in, uint8_t* out, float *scales,
                    uint32_t a_rows, uint32_t a_cols) {
  for (uint32_t i = 0; i < a_rows; i ++) {
    float max_abs = std::abs(in[i * a_cols]);
    // Loop through the array of pointers and calculate min/max
    for (uint32_t j = 0; j < a_cols; j ++) {
      if (max_abs < std::abs(in[i * a_cols + j])) {
        max_abs = std::abs(in[i * a_cols + j]);
      }
    }

    scales[i] = max_abs / 127;

    static constexpr int32_t min_val = std::numeric_limits<uint8_t>::min();
    static constexpr int32_t max_val = std::numeric_limits<uint8_t>::max();
    const int32_t zero_point = 128;

    uint32_t j = 0;
    const float32x4_t reverse_scale_dup = vdupq_n_f32(1.0f / scales[i]);
    const int32x4_t zero_point_dup = vdupq_n_s32(zero_point);
    const int32x4_t min_val_dup = vdupq_n_s32(min_val);
    const int32x4_t max_val_dup = vdupq_n_s32(max_val);

    for (; j <= a_cols - 8; j += 8) {
      const float* src_data_ptr = in + i * a_cols + j;
      float32x4_t input_val_0 = vld1q_f32(src_data_ptr);
      float32x4_t input_val_1 = vld1q_f32(src_data_ptr + 4);

      input_val_0 = vmulq_f32(input_val_0, reverse_scale_dup);
      input_val_1 = vmulq_f32(input_val_1, reverse_scale_dup);

      int32x4_t casted_val_0 = roundq_s32_f32(input_val_0);
      int32x4_t casted_val_1 = roundq_s32_f32(input_val_1);

      casted_val_0 = vaddq_s32(casted_val_0, zero_point_dup);
      casted_val_1 = vaddq_s32(casted_val_1, zero_point_dup);

      // Clamp the values to fit the target type's range.
      casted_val_0 = vmaxq_s32(casted_val_0, min_val_dup);
      casted_val_1 = vmaxq_s32(casted_val_1, min_val_dup);
      casted_val_0 = vminq_s32(casted_val_0, max_val_dup);
      casted_val_1 = vminq_s32(casted_val_1, max_val_dup);

      const uint16x4_t narrowed_val_0 = vqmovun_s32(casted_val_0);
      const uint16x4_t narrowed_val_1 = vqmovun_s32(casted_val_1);
      const uint16x8_t combined_val = vcombine_u16(narrowed_val_0, narrowed_val_1);
      const uint8x8_t combined_val_narrowed = vmovn_u16(combined_val);
      vst1_u8(out + i * a_cols + j, combined_val_narrowed);
    }
  }

  return;
}

void DequantizeOutput(const int32_t *in, float* out, float* scales,
                      uint32_t a_batch, uint32_t a_rows, uint32_t b_rows) {
  for (uint32_t b = 0; b < a_batch; b ++) {
    for (uint32_t i = 0; i < a_rows; i ++) {
      for (uint32_t j = 0; j <= b_rows - 4; j += 4) {
        uint32_t idx = b * a_rows * b_rows + i * b_rows + j;
        int32x4_t vq = vld1q_s32(in + idx);

        float32x4_t vf = vcvtq_f32_s32(vq);
        float32x4_t vscale = vdupq_n_f32(scales[i]);
        float32x4_t vres = vmulq_f32(vf, vscale);

        vst1q_f32(out + idx, vres);
      }
    }
  }
}


int (*neutron_clean_cache)(const void *, int) = clean_cache;

// ---------------- ggml-side pack (mirrors the EP inline prepack path in matmul_nbits.cc) ----------------
size_t neutron_pack_bound(int N, int K) {
    size_t bpc = K / 32;
    size_t raw = (size_t)N * K / 2;
    // measured packed size is 0.595-0.602 B/param (compression ~= raw int4 + 12% metadata); a fat bound
    // costs whole CMA regions on 14B-class models, so keep ~8% headroom: raw*1.10 + fixed slack
    return ALIGN16_SIZE(64) + ALIGN16_SIZE(N * bpc) + ALIGN16_SIZE(N * bpc * 2) + ALIGN16_SIZE(N * 4) * 2
         + ALIGN16_SIZE(raw * 11 / 10 + 65536) + ALIGN16_SIZE(N * bpc * 4 / 8 + 4096) + 64;
}

int neutron_pack_nbits(const uint8_t* B, const float* scalesData, int N, int K, uint8_t* out, size_t out_cap, neutron_layout* L) {
    const int block_size = 32, nbits = 4;
    const int blocks_per_col = K / block_size;
    const int decode_bias_len  = N * blocks_per_col * (int)sizeof(int8_t);
    const int decode_scale_len = N * blocks_per_col * (int)sizeof(int16_t);
    const int bias_len   = N * (int)sizeof(int32_t);
    const int factor_len = N * (int)sizeof(int32_t);
    const int ilength    = 16;

    int16_t*  decodeScales = (int16_t*)malloc(decode_scale_len);
    int8_t*   decodeBias   = (int8_t*)malloc(decode_bias_len);
    uint32_t* bFactorsTmp  = (uint32_t*)malloc(factor_len);
    int32_t*  bRowSum      = (int32_t*)malloc(bias_len);
    CalculateDecodeData(B, scalesData, decodeScales, decodeBias, bFactorsTmp, bRowSum, N, blocks_per_col, block_size);

    int8_t* B_int8 = (int8_t*)malloc((size_t)N * K);
    for (size_t i = 0; i < (size_t)N * K; i++) {
        uint8_t byte_val = B[i / 2];
        uint8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        B_int8[i] = (int8_t)nibble - 8;
    }
    for (size_t i = 0; i < (size_t)N; i++)
        for (size_t j = 0; j < (size_t)K; j++)
            if (decodeBias[i * blocks_per_col + j / block_size] == 1)
                B_int8[i * K + j] = -B_int8[i * K + j] - 1;

    PrepackCfg cfg;
    cfg.rearrange = true; cfg.miniWeights = true; cfg.weightBits = nbits; cfg.groupSize = block_size;
    cfg.useDecodeBias = true; cfg.compress = getenv("NEUTRON_NOCOMPRESS") == nullptr; cfg.numMacs = 16; cfg.numNeutrons = 4;
    cfg.tcmSize = 1024 * 1024; cfg.numBanks = 16;
    Dyn8 dyn8 = {}; Dyn32 dyn32 = {};
    PrepackOut pckOut; pckOut.Bpacked = &dyn8; pckOut.lengths = &dyn32;
    PrePackWeight(B_int8, N, K, &cfg, &pckOut);
    const int cd = pckOut.tilingInfo.channelDensity, nn = pckOut.tilingInfo.numNeutrons, divisions = pckOut.tilingInfo.divisions;
    const size_t weight_len = pckOut.Bpacked->size;
    const int32_t compress_num = (int32_t)pckOut.lengths->size;

    int16_t* organized_scales = (int16_t*)malloc(decode_scale_len); void* tmp_s = malloc(decode_scale_len);
    OrganizeDecodeData<int16_t>(decodeScales, organized_scales, tmp_s, N, K, cd, nn, block_size, divisions, nbits, 16);
    int8_t* organized_biases = (int8_t*)malloc(decode_bias_len); void* tmp_b = malloc(decode_bias_len);
    OrganizeDecodeData<int8_t>(decodeBias, organized_biases, tmp_b, N, K, cd, nn, block_size, divisions, nbits, 16);

    size_t total = ALIGN16_SIZE(16 * sizeof(uint32_t)) + ALIGN16_SIZE(decode_bias_len) + ALIGN16_SIZE(decode_scale_len)
                 + ALIGN16_SIZE(bias_len) + ALIGN16_SIZE(factor_len) + ALIGN16_SIZE(weight_len)
                 + ALIGN16_SIZE(compress_num * 4) + ALIGN16_SIZE(ilength);
    int rc = 0;
    L->total = total; L->weight_len = (uint32_t)weight_len; L->compress_num = compress_num;
    if (total > out_cap) { rc = -1; goto done; }

    L->header       = (uint32_t*)out;
    L->decode_bias  = (int8_t*)(out + ALIGN16_SIZE(16 * sizeof(uint32_t)));
    L->decode_scale = (int16_t*)((uint8_t*)L->decode_bias  + ALIGN16_SIZE(decode_bias_len));
    L->b_bias       = (int32_t*)((uint8_t*)L->decode_scale + ALIGN16_SIZE(decode_scale_len));
    L->b_factors    = (uint32_t*)((uint8_t*)L->b_bias      + ALIGN16_SIZE(bias_len));
    L->b_neutron    = (int8_t*)((uint8_t*)L->b_factors     + ALIGN16_SIZE(factor_len));
    L->compress_len = (int32_t*)((uint8_t*)L->b_neutron    + ALIGN16_SIZE(weight_len));
    L->decode_input = (uint8_t*)((uint8_t*)L->compress_len + ALIGN16_SIZE(compress_num * 4));

    memset(L->header, 0, 16 * sizeof(uint32_t));
    for (int i = 0; i < N; i++) L->b_bias[i] = -bRowSum[i] * 128;
    memcpy(L->decode_bias,  organized_biases, decode_bias_len);
    memcpy(L->decode_scale, organized_scales, decode_scale_len);
    memcpy(L->b_factors,    bFactorsTmp, factor_len);
    memcpy(L->b_neutron,    pckOut.Bpacked->data, weight_len);
    memcpy(L->compress_len, pckOut.lengths->data, compress_num * 4);
    memset(L->decode_input, 1, ilength);
    neutron_clean_cache(L->header, 16 * sizeof(uint32_t));
    neutron_clean_cache(L->b_neutron, weight_len);
    neutron_clean_cache(L->decode_bias, decode_bias_len);
    neutron_clean_cache(L->decode_scale, decode_scale_len);
    neutron_clean_cache(L->b_factors, factor_len);
    neutron_clean_cache(L->b_bias, bias_len);
    neutron_clean_cache(L->compress_len, compress_num * 4);
    neutron_clean_cache(L->decode_input, ilength);
done:
    free(organized_scales); free(tmp_s); free(organized_biases); free(tmp_b);
    free(decodeScales); free(decodeBias); free(bFactorsTmp); free(bRowSum); free(B_int8);
    dyn8_free(pckOut.Bpacked); dyn32_free(pckOut.lengths);
    return rc;
}

void neutron_layout_from_blob(uint8_t* out, int N, int K, uint32_t weight_len, int32_t compress_num, neutron_layout* L) {
    const int bpc = K / 32;
    const int decode_bias_len = N * bpc, decode_scale_len = N * bpc * 2, bias_len = N * 4, factor_len = N * 4, ilength = 16;
    L->header       = (uint32_t*)out;
    L->decode_bias  = (int8_t*)(out + ALIGN16_SIZE(16 * sizeof(uint32_t)));
    L->decode_scale = (int16_t*)((uint8_t*)L->decode_bias  + ALIGN16_SIZE(decode_bias_len));
    L->b_bias       = (int32_t*)((uint8_t*)L->decode_scale + ALIGN16_SIZE(decode_scale_len));
    L->b_factors    = (uint32_t*)((uint8_t*)L->b_bias      + ALIGN16_SIZE(bias_len));
    L->b_neutron    = (int8_t*)((uint8_t*)L->b_factors     + ALIGN16_SIZE(factor_len));
    L->compress_len = (int32_t*)((uint8_t*)L->b_neutron    + ALIGN16_SIZE(weight_len));
    L->decode_input = (uint8_t*)((uint8_t*)L->compress_len + ALIGN16_SIZE(compress_num * 4));
    L->weight_len = weight_len; L->compress_num = compress_num;
    L->total = ((uint8_t*)L->decode_input + ALIGN16_SIZE(ilength)) - out;
    memset(L->header, 0, 16 * sizeof(uint32_t));
    memset(L->decode_input, 1, ilength);
    neutron_clean_cache(out, (int)L->total);
}


// ======================= shared helpers (backend + neutron-pack tool) =======================
static const std::set<int> & k_allowed() {
    static std::set<int> allow = [] {
        // Verified on Neutron firmware 3.1.1: correct and deterministic across repeated runs at M=1..2048.
        // Values NOT listed here (notably 1152, 5632, 6912, 7168, 11008, 13824) hit firmware tiling bugs
        // and are decomposed into these chunks instead -- e.g. 1152 = 1024 + 128.
        std::set<int> s = {128, 256, 384, 512, 640, 768, 896, 1024, 2048, 2560, 2624,
                           4096, 4864, 5120, 8192, 9216, 9728, 10240, 10752};
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
const std::vector<int> & neutron_k_chunks(int K) {
    static std::map<int, std::vector<int>> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find(K);
    if (it != cache.end()) return it->second;
    std::vector<int> out;
    for (size_t limit = 1; limit <= 6 && !split_rec(K, out, 0, limit); limit++) out.clear();
    return cache[K] = out;
}

uint64_t neutron_fnv1a64(const void * data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * p = (const uint8_t *)data;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, p + i, 8); h = (h ^ w) * 1099511628211ull; }
    for (; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

std::string neutron_cache_path(const void * data, size_t size, int N, int K) {
    const char * dir = getenv("NEUTRON_CACHE_DIR");
    if (dir && !*dir) return "";
    std::string d = dir ? dir : std::string(getenv("HOME") ? getenv("HOME") : "/tmp") + "/.cache/ggml-neutron";
    char buf[128];
    snprintf(buf, sizeof buf, "/%016llx_%dx%d.npk", (unsigned long long)neutron_fnv1a64(data, size), N, K);
    return d + buf;
}

// GGUF Q4_0 block: fp16 scale + 32 nibbles (first 16 elements in low nibbles, next 16 in high nibbles)
struct q4_0_block { uint16_t d; uint8_t qs[16]; };
static inline float fp16_to_fp32(uint16_t h) { _Float16 f; memcpy(&f, &h, 2); return (float)f; }

int neutron_pack_q4_0(const void * q4_data, int N, int K, const std::vector<int> & kc, uint8_t * out, size_t cap, std::vector<neutron_layout> & L) {
    const int bpc = K / 32;
    const size_t row = (size_t)bpc * sizeof(q4_0_block);
    L.assign(kc.size(), neutron_layout{});
    int b0 = 0;
    for (size_t c = 0; c < kc.size(); c++) {
        const int k = kc[c], bc = k / 32;
        // Q4_0 columns [b0*32, (b0+bc)*32) -> MatMulNBits nibble layout (elem 2j low, 2j+1 high) + fp32 scales
        std::vector<uint8_t> nib((size_t)N * bc * 16);
        std::vector<float> scales((size_t)N * bc);
        for (int n = 0; n < N; n++) {
            const q4_0_block * blk = (const q4_0_block *)((const uint8_t *)q4_data + n * row) + b0;
            for (int b = 0; b < bc; b++) {
                scales[(size_t)n * bc + b] = fp16_to_fp32(blk[b].d);
                uint8_t * o = &nib[((size_t)n * bc + b) * 16];
                for (int j = 0; j < 16; j++) {
                    auto el = [&](int e) -> uint8_t { return e < 16 ? (blk[b].qs[e] & 0x0F) : (blk[b].qs[e - 16] >> 4); };
                    o[j] = (uint8_t)(el(2 * j) | (el(2 * j + 1) << 4));
                }
            }
        }
        if (neutron_pack_nbits(nib.data(), scales.data(), N, k, out, cap, &L[c]) != 0) return -1;
        const size_t used = ALIGN16_SIZE(L[c].total); const size_t pad = (64 - used % 64) % 64;
        out += used + pad; cap -= used + pad; b0 += bc;
    }
    return 0;
}

int neutron_npk_load(const std::string & path, int N, const std::vector<int> & kc, uint8_t * out, size_t cap, std::vector<neutron_layout> & L) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return -1;
    npk_hdr h; std::vector<npk_chunk> ch;
    bool ok = fread(&h, sizeof h, 1, f) == 1 && h.magic == NPK_MAGIC && h.nchunks == kc.size();
    if (ok) { ch.resize(h.nchunks); ok = fread(ch.data(), sizeof(npk_chunk), h.nchunks, f) == h.nchunks; }
    L.assign(kc.size(), neutron_layout{});
    uint8_t * o = out; size_t rem = cap;
    for (size_t c = 0; ok && c < ch.size(); c++) {
        ok = ch[c].k == (uint32_t)kc[c] && ch[c].total <= rem && fread(o, 1, ch[c].total, f) == ch[c].total;
        if (ok) {
            neutron_layout_from_blob(o, N, ch[c].k, ch[c].weight_len, ch[c].compress_num, &L[c]);
            if (L[c].total != ch[c].total) { ok = false; break; }
            const size_t used = ALIGN16_SIZE(ch[c].total); const size_t pad = (64 - used % 64) % 64;
            o += used + pad; rem -= used + pad;
        }
    }
    fclose(f);
    return ok ? 0 : -1;
}

bool neutron_npk_write(const std::string & path, const std::vector<int> & kc, const std::vector<neutron_layout> & L) {
    std::string dir = path.substr(0, path.rfind('/'));
    for (size_t i = 1; i < dir.size(); i++) if (dir[i] == '/') mkdir(dir.substr(0, i).c_str(), 0755);
    mkdir(dir.c_str(), 0755);
    std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) return false;
    npk_hdr h = { NPK_MAGIC, (uint32_t)kc.size() };
    bool ok = fwrite(&h, sizeof h, 1, f) == 1;
    for (size_t c = 0; ok && c < kc.size(); c++) { npk_chunk ch = { (uint32_t)kc[c], L[c].weight_len, (uint32_t)L[c].compress_num, 0, L[c].total }; ok = fwrite(&ch, sizeof ch, 1, f) == 1; }
    for (size_t c = 0; ok && c < kc.size(); c++) ok = fwrite(L[c].header, 1, L[c].total, f) == L[c].total;
    ok = (fclose(f) == 0) && ok;
    if (!ok) { remove(tmp.c_str()); return false; }
    return rename(tmp.c_str(), path.c_str()) == 0;
}

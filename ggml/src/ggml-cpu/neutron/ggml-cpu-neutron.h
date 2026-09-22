#pragma once
#include "ggml-backend.h"
#ifdef __cplusplus
extern "C" {
#endif
// NXP Neutron NPU offload of Q4_0 MUL_MAT as a CPU extra buffer type. Returns NULL if /dev/neutron0 / CMA is unavailable.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cpu_neutron_buffer_type(void);
#ifdef __cplusplus
}
#endif

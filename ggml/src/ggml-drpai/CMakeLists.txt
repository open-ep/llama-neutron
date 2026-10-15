# Renesas DRP-AI (RZ/V2H) backend.  Needs the DRP-AI TVM runtime from the Renesas AI SDK:
#   GGML_DRPAI_TVM_DIR  checkout of https://github.com/renesas-rz/rzv_drp-ai_tvm (headers: tvm/include, 3rdparty)
#   GGML_DRPAI_LIB_DIR  directory with libmera2_runtime.so libmera2_plan_io.so libdrp_tvm_rt.so libtvm_runtime.so
#                       (AI SDK sysroot usr/lib, or the board's /usr/local/lib)
set(GGML_DRPAI_TVM_DIR "$ENV{GGML_DRPAI_TVM_DIR}" CACHE PATH "rzv_drp-ai_tvm checkout")
set(GGML_DRPAI_LIB_DIR "$ENV{GGML_DRPAI_LIB_DIR}" CACHE PATH "DRP-AI TVM runtime libraries")
if (NOT GGML_DRPAI_TVM_DIR)
    message(FATAL_ERROR "GGML_DRPAI needs GGML_DRPAI_TVM_DIR (rzv_drp-ai_tvm checkout)")
endif()

ggml_add_backend_library(ggml-drpai ggml-drpai.cpp MeraDrpRuntimeWrapper.cpp)
target_include_directories(ggml-drpai PRIVATE
    ${GGML_DRPAI_TVM_DIR}/apps/include
    ${GGML_DRPAI_TVM_DIR}/tvm/include
    ${GGML_DRPAI_TVM_DIR}/tvm/3rdparty/dlpack/include
    ${GGML_DRPAI_TVM_DIR}/tvm/3rdparty/dmlc-core/include
    ${GGML_DRPAI_TVM_DIR}/tvm/3rdparty/compiler-rt
    ${GGML_DRPAI_TVM_DIR}/3rdparty/spdlog/include
    ${GGML_DRPAI_TVM_DIR}/3rdparty/asio/asio/include)
target_compile_definitions(ggml-drpai PRIVATE KDLDRPAI)
if (GGML_DRPAI_LIB_DIR)
    target_link_directories(ggml-drpai PRIVATE ${GGML_DRPAI_LIB_DIR})
    target_link_options(ggml-drpai PUBLIC "-Wl,-rpath-link,${GGML_DRPAI_LIB_DIR}" "-Wl,-rpath,${GGML_DRPAI_LIB_DIR}")
endif()
target_link_libraries(ggml-drpai PRIVATE mera2_runtime mera2_plan_io drp_tvm_rt tvm_runtime pthread dl)

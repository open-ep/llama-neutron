# Cross toolchain for NXP i.MX 95 (Cortex-A55). Point it at an NXP Yocto SDK or recipe sysroot:
#   IMX95_SYSROOT = target sysroot
#   IMX95_CROSS   = compiler prefix, e.g. /opt/fsl-imx-wayland/<ver>/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
if(NOT DEFINED ENV{IMX95_SYSROOT} OR NOT DEFINED ENV{IMX95_CROSS})
    message(FATAL_ERROR "set IMX95_SYSROOT (target sysroot) and IMX95_CROSS (compiler prefix)")
endif()
set(CMAKE_SYSROOT       $ENV{IMX95_SYSROOT})
set(CMAKE_C_COMPILER    $ENV{IMX95_CROSS}gcc)
set(CMAKE_CXX_COMPILER  $ENV{IMX95_CROSS}g++)
set(CMAKE_C_FLAGS_INIT   "-march=armv8.2-a+crypto+dotprod+fp16 -mcpu=cortex-a55")
set(CMAKE_CXX_FLAGS_INIT "-march=armv8.2-a+crypto+dotprod+fp16 -mcpu=cortex-a55")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

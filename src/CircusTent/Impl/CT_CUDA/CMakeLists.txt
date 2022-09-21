# src/Impl/CT_CUDA CMakeLists.txt
# Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

set(CTCUDASrcs
)

if (ENABLE_CUDA)
  set(CTCUDASrcs ${CTCUDASrcs} CT_CUDA.cuh CT_CUDA.cu CT_CUDA_IMPL.cu)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ")
endif()

include_directories(${CT_INCLUDE_PATH})
include_directories(./)

add_library(CT_CUDA_OBJS OBJECT ${CTCUDASrcs})
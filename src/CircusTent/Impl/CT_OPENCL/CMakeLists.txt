# src/Impl/CT_OPENCL CMakeLists.txt
# Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

set(CTOPENCLSrcs
)

if (ENABLE_OPENCL)
  set(CTOPENCLSrcs ${CTOPENCLSrcs} CT_OPENCL.h CT_OPENCL.cpp)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
endif()

include_directories(${CT_INCLUDE_PATH})
include_directories(./)

add_library(CT_OPENCL_OBJS OBJECT ${CTOPENCLSrcs})
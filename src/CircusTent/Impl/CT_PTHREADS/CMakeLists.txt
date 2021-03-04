# src/Impl/CT_PTHREADS CMakeLists.txt
# Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

set(CTPTHREADSSrcs
)

if (ENABLE_PTHREADS)
  set(CTPTHREADSSrcs ${CTPTHREADSSrcs} CT_PTHREADS.h CT_PTHREADS.cpp CT_PTHREADS_IMPL.c)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
endif()

include_directories(${CT_INCLUDE_PATH})
include_directories(./)

add_library(CT_PTHREADS_OBJS OBJECT ${CTPTHREADSSrcs})

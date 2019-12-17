# src/Impl/CT_OMP CMakeLists.txt
# Copyright (C) 2017-2020 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

set(CTOMPSrcs
)

if (ENABLE_OMP)
  set(CTOMPSrcs ${CTOMPSrcs} CT_OMP.h CT_OMP.cpp CT_IMPL.c)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
endif()

include_directories(${CT_INCLUDE_PATH})
include_directories(./)

add_library(CT_OMP_OBJS OBJECT ${CTOMPSrcs})


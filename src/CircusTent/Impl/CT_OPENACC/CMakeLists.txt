# src/Impl/CT_OPENACC CMakeLists.txt
# Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

set(CTOPENACCSrcs
)

if (ENABLE_OPENACC)
  set(CTOPENACCSrcs ${CTOPENACCSrcs} CT_OPENACC.h CT_OPENACC.cpp CT_OPENACC_IMPL.c)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
endif()

include_directories(${CT_INCLUDE_PATH})
include_directories(./)

add_library(CT_OPENACC_OBJS OBJECT ${CTOPENACCSrcs})

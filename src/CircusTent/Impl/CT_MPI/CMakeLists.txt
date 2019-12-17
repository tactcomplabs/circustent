# src/Impl/CT_MPI CMakeLists.txt
# Copyright (C) 2017-2020 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

set(CTMPISrcs
)

if (ENABLE_MPI)
  set(CTMPISrcs ${CTMPISrcs} CT_MPI.h CT_MPI.cpp CT_MPI_IMPL.c)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
endif()

include_directories(${CT_INCLUDE_PATH})
include_directories(./)

add_library(CT_MPI_OBJS OBJECT ${CTMPISrcs})

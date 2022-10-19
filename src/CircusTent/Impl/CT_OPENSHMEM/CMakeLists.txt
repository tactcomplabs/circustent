# src/Impl/CT_OPENSHMEM CMakeLists.txt
# Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

set(CTSHMEMSrcs
)

if (ENABLE_OPENSHMEM)
  set(CTSHMEMSrcs ${CTSHMEMSrcs} CT_SHMEM.h CT_SHMEM.cpp CT_SHMEM_IMPL.c)
endif()

include_directories(${CT_INCLUDE_PATH})
include_directories(./)

add_library(CT_SHMEM_OBJS OBJECT ${CTSHMEMSrcs})

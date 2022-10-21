# src/Impl/CT_OMP_TARGET CMakeLists.txt
# Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

set(CTOMPTargetSrcs
)

if (ENABLE_OMP_TARGET)
  set(CTOMPTargetSrcs ${CTOMPTargetSrcs} CT_OMP_TARGET.h CT_OMP_TARGET.cpp CT_OMP_TARGET_IMPL.c)
endif()

include_directories(${CT_INCLUDE_PATH})
include_directories(./)

add_library(CT_OMP_TARGET_OBJS OBJECT ${CTOMPTargetSrcs})

# src/Impl/CT_YGM CMakeLists.txt
# Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

set(CTYGMSrcs
)

if (ENABLE_YGM)
  set(CTYGMSrcs ${CTYGMSrcs} CT_YGM.h CT_YGM.cpp CT_YGM_IMPL.cpp)
endif()

include_directories(${YGM_INCLUDE_PATH})
include_directories(${CT_INCLUDE_PATH})
include_directories(./)

add_library(CT_YGM_OBJS OBJECT ${CTYGMSrcs})

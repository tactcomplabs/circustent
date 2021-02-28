# src/Impl/CT_XBGAS CMakeLists.txt
# Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

set(CTXBGASSrcs
)

if (ENABLE_XBGAS)
  set(CTXBGASSrcs ${CTXBGASSrcs} CT_XBGAS.h CT_XBGAS.cpp CT_XBGAS_IMPL.c)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
endif()

include_directories(${CT_INCLUDE_PATH})
include_directories(./)

add_library(CT_XBGAS_OBJS OBJECT ${CTXBGASSrcs})

# src/Impl/CT_CPP_STD CMakeLists.txt
# Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

set(CT_CPP_STD_SRCS
)

if (ENABLE_CPP_STD)
  set(CT_CPP_STD_SRCS ${CT_CPP_STD_SRCS} CT_CPP_STD.h CT_CPP_STD.cpp CT_CPP_STD_IMPL.cpp)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

include_directories(${CT_INCLUDE_PATH})
include_directories(./)

add_library(CT_CPP_STD_OBJS OBJECT ${CT_CPP_STD_SRCS})

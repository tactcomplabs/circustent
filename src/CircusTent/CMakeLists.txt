# src/CircusTent CMakeLists.txt
# Copyright (C) 2017-2019 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

add_subdirectory(Impl)

set(CTSrcs
CT_Main.cpp
CTOpts.cpp
)

include_directories(${CT_INCLUDE_PATH})
include_directories(${CT_SRC_PATH}/Impl/CT_OMP)

add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_OMP_OBJS>
                                    $<TARGET_OBJECTS:CT_SHMEM_OBJS>)

install(TARGETS circustent DESTINATION bin)

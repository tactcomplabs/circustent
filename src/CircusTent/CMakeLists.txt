# src/CircusTent CMakeLists.txt
# Copyright (C) 2017-2020 Tactical Computing Laboratories, LLC
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

if (ENABLE_OMP)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_OMP_OBJS>)
endif()

if (ENABLE_OPENSHMEM)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_SHMEM_OBJS>)
  target_link_libraries(circustent oshmem mpi)
endif()

if (ENABLE_MPI)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_MPI_OBJS>)
endif()

if (ENABLE_XBGAS)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_XBGAS_OBJS>)
  target_link_libraries(circustent $ENV{RISCV}/lib/libxbrtime.a)
endif()

install(TARGETS circustent DESTINATION bin)

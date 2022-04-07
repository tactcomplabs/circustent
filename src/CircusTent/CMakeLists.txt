# src/CircusTent CMakeLists.txt
# Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
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

if (ENABLE_OMP)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_OMP_OBJS>)
endif()

if (ENABLE_OMP_TARGET)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_OMP_TARGET_OBJS>)
endif()

if (ENABLE_OPENSHMEM)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_SHMEM_OBJS>)
endif()

if (ENABLE_MPI)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_MPI_OBJS>)
endif()

if (ENABLE_XBGAS)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_XBGAS_OBJS>)
  target_link_libraries(circustent $ENV{RISCV}/lib/libxbrtime.a)
endif()

if (ENABLE_PTHREADS)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_PTHREADS_OBJS>)
endif()

if (ENABLE_OPENACC)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_OPENACC_OBJS>)
endif()

if (ENABLE_OPENCL)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_OPENCL_OBJS>)
  target_link_libraries(circustent OpenCL)
endif()

if (ENABLE_CPP_STD)
  add_executable(circustent ${CTSrcs} $<TARGET_OBJECTS:CT_CPP_STD_OBJS>)
endif()

install(TARGETS circustent DESTINATION bin)

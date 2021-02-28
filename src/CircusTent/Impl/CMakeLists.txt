# src/Impl CMakeLists.txt
#
# Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#

if (ENABLE_OMP)
  add_subdirectory(CT_OMP)
endif()

if (ENABLE_OMP_TARGET)
  add_subdirectory(CT_OMP_TARGET)
endif()

if (ENABLE_OPENSHMEM)
  add_subdirectory(CT_OPENSHMEM)
endif()

if (ENABLE_MPI)
  add_subdirectory(CT_MPI)
endif()

if (ENABLE_XBGAS)
  add_subdirectory(CT_XBGAS)
endif()

if (ENABLE_PTHREADS)
  add_subdirectory(CT_PTHREADS)
endif()

if (ENABLE_OPENACC)
  add_subdirectory(CT_OPENACC)
endif()

# EOF

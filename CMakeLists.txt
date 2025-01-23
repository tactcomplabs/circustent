# CircusTent Top-Level CMake
#
# Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

cmake_minimum_required(VERSION 3.14)
project(CircusTent)

set(VERSION "0.2.0")

#------------------------------------------------------------------------
#-- EXTERNAL CMAKE SCRIPTS
#------------------------------------------------------------------------
set(CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake"
    ${CMAKE_MODULE_PATH}
)

include(UseLATEX OPTIONAL)

#------------------------------------------------------------------------
#-- DERIVE THE RELEASE CODENAME
#------------------------------------------------------------------------
execute_process(COMMAND lsb_release -cs
                OUTPUT_VARIABLE CT_RELEASE_CODENAME
                OUTPUT_STRIP_TRAILING_WHITESPACE)

#------------------------------------------------------------------------
#-- COMPILER OPTIONS  
#------------------------------------------------------------------------
if( CT_CFLAGS )
  message(STATUS "USING MACHINE-SPECIFIC C_FLAGS: ${CT_CFLAGS}")
endif()
if( CT_CXXFLAGS )
  message(STATUS "USING MACHINE-SPECIFIC CXX_FLAGS: ${CT_CXXFLAGS}")
endif()
if( CT_LINKER_FLAGS )
  message(STATUS "USING MACHINE-SPECIFIC LINKER_FLAGS: ${CT_LINKER_FLAGS}")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c99 ${CT_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 ${CT_CXXFLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -g ${CT_CXXFLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -std=c++11 ${CT_CXXFLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CT_LINKER_FLAGS}")

#------------------------------------------------------------------------
#-- IMPLEMENTATION-SPECIFIC FLAGS
#------------------------------------------------------------------------
if (ENABLE_OMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_OMP_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_OMP_")
  message(STATUS "ENABLING OpenMP Implementation")
else()
  message(STATUS "DISABLING OpenMP Implementation")
endif()

if (ENABLE_OMP_TARGET)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_OMP_TARGET_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_OMP_TARGET_")
  message(STATUS "ENABLING OpenMP Target Implementation")
else()
  message(STATUS "DISABLING OpenMP Target Implementation")
endif()

if (ENABLE_OPENSHMEM)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_OPENSHMEM_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_OPENSHMEM_")
  message(STATUS "ENABLING OpenSHMEM Implementation")
else()
  message(STATUS "DISABLING OpenSHMEM Implementation")
endif()

if (ENABLE_MPI)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_MPI_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_MPI_")
  message(STATUS "ENABLING MPI Implementation")
else()
  message(STATUS "DISABLING MPI Implementation")
endif()

if (ENABLE_XBGAS)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_XBGAS_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_XBGAS_")
  message(STATUS "ENABLING XBGAS Implementation")
else()
  message(STATUS "DISABLING XBGAS Implementation")
endif()

if (ENABLE_PTHREADS)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_PTHREADS_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_PTHREADS_")
  message(STATUS "ENABLING Pthreads Implementation")
else()
  message(STATUS "DISABLING Pthreads Implementation")
endif()

if (ENABLE_OPENACC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_OPENACC_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_OPENACC_")
  message(STATUS "ENABLING OpenACC Implementation")
else()
  message(STATUS "DISABLING OpenACC Implementation")
endif()

if (ENABLE_OPENCL)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_OPENCL_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_OPENCL_")
  message(STATUS "ENABLING OpenCL Implementation")
else()
  message(STATUS "DISABLING OpenCL Implementation")
endif()

if (ENABLE_CPP_STD)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_CPP_STD_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_CPP_STD_")
  message(STATUS "ENABLING C++ Standard Atomics Implementation")
else()
  message(STATUS "DISABLING C++ Standard Atomics Implementation")
endif()

if (ENABLE_CUDA)
  cmake_minimum_required(VERSION 3.17)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_CUDA_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_CUDA_")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -D_ENABLE_CUDA_")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_ENABLE_CUDA_")

  message(STATUS "ENABLING CUDA Implementation")
else()
  message(STATUS "DISABLING CUDA Implementation")
endif()

if (ENABLE_YGM)

  # Gather dependencies for CT_YGM Impl

  include(FetchContent)

  #
  #  MPI
  find_package(MPI)

  #
  #  ygm
  find_package(ygm CONFIG)

  if(NOT ygm_FOUND)
    if(DEFINED ENV{YGM_BENCH_YGM_REPO})
		  set(YGM_REPO $ENV{YGM_BENCH_YGM_REPO})
	  else()
		  set(YGM_REPO "https://github.com/llnl/ygm.git")
	  endif()

    if(DEFINED ENV{YGM_BENCH_YGM_TAG})
      set(YGM_TAG $ENV{YGM_BENCH_YGM_TAG})
    else()
      set(YGM_TAG "v0.6")
    endif()

    FetchContent_Declare(
    ygm
    GIT_REPOSITORY 	${YGM_REPO}
    GIT_TAG 		${YGM_TAG}
    )      

    set(JUST_INSTALL_YGM ON)

    FetchContent_MakeAvailable(ygm)
    message(STATUS "Cloned ygm dependency " ${ygm_SOURCE_DIR})
  else()
	  message(STATUS "Found ygm dependency " ${ygm_DIR})
  endif()

  # Set flags for YGM specific options

  if (SHORTCUT_RPC)
    set(SHORTCUT_ARGS " -D_SHORTCUT_RPC_")
  else()
    set(SHORCUT_ARGS "")
  endif()

  if (SKIP_PROGRESS_PTRCHASE)
    set(PROGRESS_ARGS "")
  else ()
    set(PROGRESS_ARGS " -D_PROGRESS_PTRCHASE_")
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_YGM_${SHORTCUT_ARGS}${PROGRESS_ARGS}")
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -D_ENABLE_YGM_${SHORTCUT_ARGS}${PROGRESS_ARGS}")

  message(STATUS "ENABLING YGM Implementation")
else()
  message(STATUS "DISABLING YGM Implementation")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

#------------------------------------------------------------------------
# Add package paths
#------------------------------------------------------------------------
if(NOT CT_INCLUDE_PATH)
  set(CT_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")
  message(STATUS "CT_INCLUDE_PATH set to ${CT_INCLUDE_PATH}")
endif()

if(NOT CT_SRC_PATH)
  set(CT_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/CircusTent")
  message(STATUS "CT_SRC_PATH set to ${CT_SRC_PATH}")
endif()

#------------------------------------------------------------------------
#-- TESTING
#------------------------------------------------------------------------
# Enable testing
enable_testing()

#------------------------------------------------------------------------
#-- PACKAGING
#------------------------------------------------------------------------
#-- builds a standard redhat/centos rpm
if(CIRCUSTENT_BUILD_RPM)
  set(CPACK_PACKAGE_VERSION ${VERSION})
  set(CPACK_GENERATOR "RPM")
  set(CPACK_PACKAGE_NAME "CircusTent")
  set(CPACK_PACKAGE_RELEASE 1)
  set(CPACK_PACKAGE_CONTACT "John Leidel")
  set(CPACK_PACKAGE_VENDOR "Tactical Computing Laboratories, LLC")
  set(CPACK_RPM_PACKAGE_REQUIRES "devtoolset-7-gcc-c++")
  set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")
  include(CPack)
endif()

#-- builds a standard debian/ubuntu deb
if(CIRCUSTENT_BUILD_DEB)
  set(CPACK_PACKAGE_VERSION ${VERSION})
  set(CPACK_GENERATOR "DEB")
  set(CPACK_PACKAGE_NAME "CircusTent")
  set(CPACK_PACKAGE_RELEASE 1)
  set(CPACK_PACKAGE_CONTACT "John Leidel")
  set(CPACK_PACKAGE_VENDOR "Tactical Computing Laboratories, LLC")
  #-- setup package deps
  if( "${CT_RELEASE_CODENAME}" STREQUAL "bionic" )
    message(STATUS "Setting package dependencies for Ubuntu 18.04 (bionic)")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
  elseif( "${CT_RELEASE_CODENAME}" STREQUAL "xenial" )
    message(STATUS "Setting package dependencies for Ubuntu 16.04 (xenial)")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
  else()
    message(STATUS "Unknown debian version: ${CT_RELEASE_CODENAME}")
    message(STATUS "No debian package dependencies are set!!!")
  endif()
  set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")
  include(CPack)
endif()

#-- builds a tgz
if(CIRCUSTENT_BUILD_TGZ)
  set(CPACK_PACKAGE_VERSION ${VERSION})
  set(CPACK_GENERATOR "TGZ")
  set(CPACK_PACKAGE_NAME "CircusTent")
  set(CPACK_PACKAGE_RELEASE 1)
  set(CPACK_PACKAGE_CONTACT "John Leidel")
  set(CPACK_PACKAGE_VENDOR "Tactical Computing Laboratories, LLC")
  set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")
  include(CPack)
endif()

#-- builds an osx dmg
if(CIRCUSTENT_BUILD_DMG)
  set(CPACK_PACKAGE_VERSION ${VERSION})
  set(CPACK_GENERATOR "DMG")
  set(CPACK_PACKAGE_NAME "CircusTent")
  set(CPACK_PACKAGE_RELEASE 1)
  set(CPACK_PACKAGE_CONTACT "John Leidel")
  set(CPACK_PACKAGE_VENDOR "Tactical Computing Laboratories, LLC")
  set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")
  include(CPack)
endif()

#------------------------------------------------------------------------
#-- SUBDIRECTORIES
#------------------------------------------------------------------------
# Add subdirectories
add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(test)

# EOF

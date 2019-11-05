# CircusTent Top-Level CMake
#
# Copyright (C) 2017-2019 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# See LICENSE in the top level directory for licensing details
#

cmake_minimum_required(VERSION 3.4.3)
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
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

#------------------------------------------------------------------------
#-- COMPILER OPTIONS
#------------------------------------------------------------------------
if( CT_C_FLAGS )
  message(STATUS "USING MACHINE-SPECIFIC C-FLAGS: ${CT_C_FLAGS}")
endif()
if( CT_CXX_FLAGS )
  message(STATUS "USING MACHINE-SPECIFIC CXX-FLAGS: ${CT_CXX_FLAGS}")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c99 -Wall ${CT_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wall ${CT_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")

#------------------------------------------------------------------------
#-- IMPLEMENTATION-SPECIFIC FLAGS
#------------------------------------------------------------------------
if (ENABLE_OMP)
  #set(CTOMPSrcs ${CTOMPSrcs} CT_OMP.h CT_OMP.cpp)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -D_ENABLE_OMP_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -D_ENABLE_OMP_")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
  message(STATUS "ENABLING OpenMP Implementation")
else()
  message(STATUS "DISABLING OpenMP Implementation")
endif()

if (ENABLE_OPENSHMEM)
  #set(CMAKE_C_COMPILER "oshcc")
  #set(CMAKE_CXX_COMPILER "oshcxx")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_OPENSHMEM_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_OPENSHMEM_")
  message(STATUS "ENABLING OpenSHMEM Implementation")
else()
  message(STATUS "DISABLING OpenSHMEM Implementation")
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

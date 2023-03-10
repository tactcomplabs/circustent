#!/bin/bash

##################################################
# Flags for each GPU impl
##################################################
CMAKE_CUDA_FLAGS="-DENABLE_CUDA=ON"

# CMAKE_OMP_FLAGS="-mp=gpu -Minfo=mp -gpu -DENABLE_OMP_TARGET=ON"
CMAKE_OMP_FLAGS="-DENABLE_OMP_TARGET=ON"

# CMAKE_OACC_FLAGS="-acc -gpu=cc70 -DENABLE_OPENACC=ON"
CMAKE_OACC_FLAGS="-DENABLE_OPENACC=ON"

# Set flags for desired backend here
# CMAKE_FLAGS=$CMAKE_CUDA_FLAGS
# CMAKE_FLAGS=$CMAKE_OMP_FLAGS
CMAKE_FLAGS=$CMAKE_OACC_FLAGS


##################################################
# Don't edit below
##################################################
cd ../
CT_ROOT=$(pwd)
CT_BUILD=$CT_ROOT/build

if [ -d $CT_BUILD ] ; then
    rm -rf $CT_BUILD
fi

mkdir $CT_BUILD
cd $CT_BUILD
cmake $CMAKE_FLAGS ../
make

cd $CT_ROOT/scripts
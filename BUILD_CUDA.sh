#!/bin/bash

export CT_ROOT=/home/mibeebe/ct/circustent_dev_main
export CT_BUILD=$CT_ROOT/build

if [ -d $CT_BUILD ] ; then
    rm -rf $CT_BUILD
fi

mkdir $CT_BUILD
cd $CT_BUILD
cmake -DENABLE_CUDA=ON ../
make

cd $CT_ROOT
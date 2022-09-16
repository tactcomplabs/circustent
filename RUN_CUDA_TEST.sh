#!/bin/bash

export CT_ROOT=/home/mibeebe/ct/circustent_cuda
export CT_BUILD=$CT_ROOT/build
export EXE=$CT_BUILD/src/CircusTent/circustent


# $EXE --help
# echo

$EXE --bench RAND_ADD --memsize 1024 --pes 1 -i 100 --blocks 1 --threads 1


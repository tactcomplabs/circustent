#!/bin/bash

HLINE="------------------------------------------------"

cd ../
CT_ROOT=$(pwd)
CT_BUILD=$CT_ROOT/build
EXE=$CT_BUILD/src/CircusTent/circustent
cd .tests

#########################################################################
#                           2048 total theads
#########################################################################
echo "=============================================="
echo "              2048 total"
echo "=============================================="
echo $HLINE
echo "Dimensions: 2 x 1024"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 2 --threads 1024
echo

echo $HLINE
echo "Dimensions: 4 x 512"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 4 --threads 512
echo

echo $HLINE
echo "Dimensions: 8 x 256"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 8 --threads 256
echo

echo $HLINE
echo "Dimensions: 16 x 128"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 16 --threads 128
echo

echo $HLINE
echo "Dimensions: 32 x 64"
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 32 --threads 64
echo

echo $HLINE
echo "Dimensions: 64 x 32"
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 64 --threads 32
echo

echo $HLINE
echo "Dimensions: 128 x 16"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 128 --threads 16
echo

echo $HLINE
echo "Dimensions: 256 x 8"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 256 --threads 8
echo

echo $HLINE
echo "Dimensions: 512 x 4"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 512 --threads 4
echo

echo $HLINE
echo "Dimensions: 1024 x 2"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 1024 --threads 2
echo
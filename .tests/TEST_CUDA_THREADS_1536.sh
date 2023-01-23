#!/bin/bash

HLINE="------------------------------------------------"

cd ../
CT_ROOT=$(pwd)
CT_BUILD=$CT_ROOT/build
EXE=$CT_BUILD/src/CircusTent/circustent
cd .tests

#########################################################################
#                           1536 total threads
#########################################################################
echo "=============================================="
echo "              1536 total"
echo "=============================================="
echo $HLINE
echo "Dimensions: 3 x 512"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 3 --threads 512
echo

echo $HLINE
echo "Dimensions: 6 x 256"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 6 --threads 256
echo

echo $HLINE
echo "Dimensions: 12 x 128"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 12 --threads 128
echo

echo $HLINE
echo "Dimensions: 24 x 64"
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 24 --threads 64
echo

echo $HLINE
echo "Dimensions: 32 x 48"
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 32 --threads 48
echo


#########################################################################


echo $HLINE
echo "Dimensions: 48 x 32"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 48 --threads 32
echo

echo $HLINE
echo "Dimensions: 64 x 24"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 64 --threads 24
echo

echo $HLINE
echo "Dimensions: 128 x 12"
echo $HLINE
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 128 --threads 12
echo

echo $HLINE
echo "Dimensions: 256 x 6"
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 256 --threads 6
echo

echo $HLINE
echo "Dimensions: 512 x 3"
$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 512 --threads 3
echo
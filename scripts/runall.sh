#!/bin/bash
#
# CircusTent benchmark script
# usage: runall.sh /path/to/circustent
#
# Freely modify the options to specify the memory size and number of iterations
# Also freely modify the number of cores
#

CT=$1
OPTS="-m 16488974000 -i 20000000"

BENCH="RAND_ADD RAND_CAS STRIDE1_ADD STRIDE1_CAS PTRCHASE_ADD PTRCHASE_CAS CENTRAL_ADD CENTRAL_CAS SG_ADD SG_CAS SCATTER_ADD SCATTER_CAS GATHER_ADD GATHER_CAS"
CORES="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"

#-- execute everything except stride-n
for B in $BENCH; do
  touch $B.txt
  for C in $CORES; do
    echo "RUNNING $B FOR $C CORES"
    touch tmp.txt
    $CT -b $B -p $C $OPTS >> tmp.txt 2>&1
    TIMING=`cat tmp.txt | grep Timing | awk '{print $4}'`
    GAMS=`cat tmp.txt | grep GAMS | awk '{print $5}'`
    echo "$C $TIMING $GAMS" >> $B.txt 2>&1
    rm tmp.txt
  done
done

#-- execute the stride-N benchmarks
touch STRIDEN_ADD.txt
for C in $CORES; do
  echo "RUNNING STRIDEN_ADD FOR $C CORES"
  touch tmp.txt
  $CT -b STRIDEN_ADD -p $C -m 16488974000 -i 2000000 -s 9 >> tmp.txt 2>&1
  TIMING=`cat tmp.txt | grep Timing | awk '{print $4}'`
  GAMS=`cat tmp.txt | grep GAMS | awk '{print $5}'`
  echo "$C $TIMING $GAMS" >> STRIDEN_ADD.txt 2>&1
  rm tmp.txt
done

touch STRIDEN_CAS.txt
for C in $CORES; do
  echo "RUNNING STRIDEN_CAS FOR $C CORES"
  touch tmp.txt
  $CT -b STRIDEN_CAS -p $C -m 16488974000 -i 2000000 -s 9 >> tmp.txt 2>&1
  TIMING=`cat tmp.txt | grep Timing | awk '{print $4}'`
  GAMS=`cat tmp.txt | grep GAMS | awk '{print $5}'`
  echo "$C $TIMING $GAMS" >> STRIDEN_CAS.txt 2>&1
  rm tmp.txt
done


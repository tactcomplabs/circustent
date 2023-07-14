#!/bin/bash

DIRNAME=ruby_16N

touch results_table_$DIRNAME.csv

echo "Kernel, Wall Time, GAMS" >> results_table_$DIRNAME.csv 2>&1

for file in ./$DIRNAME/*
do

  KNL=``
  TIMING=``
  GAMS=``

  KNL=`cat $file | grep Kernel | awk '{print $4}'`
  TIMING=`cat $file | grep Timing | awk '{print $4}'`
  GAMS=`cat $file | grep GAMS | awk '{print $5}'`

  knl_tokens=( $KNL )
  wtc_tokens=( $TIMING )
  gam_tokens=( $GAMS )

  for i in $(seq 0 ${knl_tokens[*]})
  do 

    echo ${knl_tokens[i]}, ${wtc_tokens[i]}, ${gam_tokens[i]} >> results_table_$DIRNAME.csv 2>&1
  done

done
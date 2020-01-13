#-- GAMS.gnuplot

set pointintervalbox 3
set term png size 1920,1080
set title "Giga AMOs/sec for system XXXXXX"
set output "GAMS.png"
set ylabel "Giga AMOs/sec (GAMS)"
set xlabel "PEs"

plot  'CENTRAL_ADD.txt' using 1:3 with linespoints title 'CENTRAL\_ADD',\
      'CENTRAL_CAS.txt' using 1:3 with linespoints title 'CENTRAL\_CAS',\
      'GATHER_ADD.txt' using 1:3 with linespoints title 'GATHER\_ADD',\
      'GATHER_CAS.txt' using 1:3 with linespoints title 'GATHER\_CAS',\
      'PTRCHASE_ADD.txt' using 1:3 with linespoints title 'PTRCHASE\_ADD',\
      'PTRCHASE_CAS.txt' using 1:3 with linespoints title 'PTRCHASE\_CAS',\
      'RAND_ADD.txt' using 1:3 with linespoints title 'RAND\_ADD',\
      'RAND_CAS.txt' using 1:3 with linespoints title 'RAND\_CAS',\
      'SCATTER_ADD.txt' using 1:3 with linespoints title 'SCATTER\_ADD',\
      'SCATTER_CAS.txt' using 1:3 with linespoints title 'SCATTER\_CAS',\
      'SG_ADD.txt' using 1:3 with linespoints title 'SG\_ADD',\
      'SG_CAS.txt' using 1:3 with linespoints title 'SG\_CAS',\
      'STRIDE1_ADD.txt' using 1:3 with linespoints title 'STRIDE1\_ADD',\
      'STRIDE1_CAS.txt' using 1:3 with linespoints title 'STRIDE1\_CAS',\
      'STRIDEN_ADD.txt' using 1:3 with linespoints title 'STRIDEN\_ADD',\
      'STRIDEN_CAS.txt' using 1:3 with linespoints title 'STRIDEN\_CAS'

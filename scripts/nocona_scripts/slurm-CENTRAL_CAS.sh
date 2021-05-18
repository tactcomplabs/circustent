#!/bin/bash
#
# scripts/slurm.sh
#
# Copyright (C) 2017-2018 Tactical Computing Laboratories, LLC
# All Rights Reserved
# contact@tactcomplabs.com
#
# This file is a part of the PAN-RUNTIME package.  For license
# information, see the LICENSE file in the top level directory of
# this distribution.
#
#
# Sample SLURM batch script
#
# Usage: sbatch -N6 slurm.sh
#
# This command requests 6 nodes for execution
#

PES="$1"
PPR="$2"
TOTAL_PES="$3"
TEST="CENTRAL_CAS"
HOSTFILE="output-$TEST-$PES-$PPR.txt"

#-- Stage 1: load the necessary modules
module load gcc/10.1.0
module load openmpi/4.0.4

#-- Stage 2: change directories to the base directory
cd /home/brodwill/nocona/circustent/build/mpi/bin

#-- Stage 3: create the hostfile
srun hostname > $HOSTFILE

#-- Stage 4: execute SST
echo "mpirun -display-map --hostfile $HOSTFILE -np $TOTAL_PES circustent -b $TEST -m 4294967296 -p $TOTAL_PES -i 100000"
mpirun -display-map --hostfile $HOSTFILE -np $TOTAL_PES circustent -b $TEST -m 2147483648 -p $TOTAL_PES -i 100000

# EOF

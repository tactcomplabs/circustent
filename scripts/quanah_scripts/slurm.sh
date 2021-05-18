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

#-- Stage 1: load the necessary modules
module load gnu7/7.3.0
module load openmpi3/3.0.0

#-- Stage 2: change directories to the base directory
cd /home/brodwill/quanah/circustent/build/mpi/bin

#-- Stage 3: create the hostfile
srun hostname > output.txt

#-- Stage 4: execute SST
mpirun -display-map --map-by ppr:1:node --hostfile output.txt -n 2 circustent -b RAND_ADD -m 4294967296 -p 2 -i 100000

# EOF

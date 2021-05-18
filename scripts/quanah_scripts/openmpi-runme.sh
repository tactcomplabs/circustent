#!/bin/bash

TIME="00:05:00"
PART="quanah"
PERNODE="1 2 4 8 16"
NODES="2 3 4 5 6 7"
for PPN in $PERNODE;do
  for NODE in $NODES;do
    TASKS=$( bc -l <<<"$PPN*$NODE" )
    echo "SUBMITTING TOTAL_TASKS=$TASKS; NODES=$NODE; TASKS PER NODE=$PPN"

    #-- RAND
    sbatch -o circustent-openmpi-RAND_ADD-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-RAND_ADD.sh $NODE $PPN $TASKS
    sbatch -o circustent-openmpi-RAND_CAS-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-RAND_CAS.sh $NODE $PPN $TASKS

    #-- CENTRAL
    sbatch -o circustent-openmpi-CENTRAL_ADD-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-CENTRAL_ADD.sh $NODE $PPN $TASKS
    sbatch -o circustent-openmpi-CENTRAL_CAS-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-CENTRAL_CAS.sh $NODE $PPN $TASKS

    #-- GATHER
    sbatch -o circustent-openmpi-GATHER_ADD-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-GATHER_ADD.sh $NODE $PPN $TASKS
    sbatch -o circustent-openmpi-GATHER_CAS-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-GATHER_CAS.sh $NODE $PPN $TASKS

    #-- SCATTER
    sbatch -o circustent-openmpi-SCATTER_ADD-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-SCATTER_ADD.sh $NODE $PPN $TASKS
    sbatch -o circustent-openmpi-SCATTER_CAS-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-SCATTER_CAS.sh $NODE $PPN $TASKS

    #-- SCATTER/GATHER
    sbatch -o circustent-openmpi-SG_ADD-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-SG_ADD.sh $NODE $PPN $TASKS
    sbatch -o circustent-openmpi-SG_CAS-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-SG_CAS.sh $NODE $PPN $TASKS

    #-- PTRCHASE
    sbatch -o circustent-openmpi-PTRCHASE_ADD-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-PTRCHASE_ADD.sh $NODE $PPN $TASKS
    sbatch -o circustent-openmpi-PTRCHASE_CAS-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-PTRCHASE_CAS.sh $NODE $PPN $TASKS

    #-- STRIDE1
    sbatch -o circustent-openmpi-STRIDE1_ADD-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-STRIDE1_ADD.sh $NODE $PPN $TASKS
    sbatch -o circustent-openmpi-STRIDE1_CAS-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-STRIDE1_CAS.sh $NODE $PPN $TASKS

    #-- STRIDEN
    sbatch -o circustent-openmpi-STRIDEN_ADD-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-STRIDEN_ADD.sh $NODE $PPN $TASKS
    sbatch -o circustent-openmpi-STRIDEN_CAS-$NODE-$PPN.out --partition=$PART --time=$TIME --ntasks=$TASKS --nodes=$NODE --ntasks-per-node=$PPN --exclusive slurm-STRIDEN_CAS.sh $NODE $PPN $TASKS
  done;
done;

# CircusTent // YGM Guide

Hello all, I hope that this is helpful when trying to run CircusTent benchmarks with the YGM message platform. There is likely a much better/cleaner way to set this up, but I know that this works for me. Let me (@prestonpiercey-tamu) know if you have any trouble with it.

# Installing the project

Before proceeding, please note that MPI is *required* for YGM and this project will not run without it. 

Start by creating an empty directory to hold the project. Throughout this guide we refer to this as <new_project_dir> for readability and convenience

    cd <new_project_dir>  

Next, load any of supported compilers. gcc 8, 9, 10 are tested. I have been using 12.1.1 throughout this project and seen no compatibility issues, but it is not explicitly supported by ygm documentation. 

    module load gcc/12.1.1 

## Installing YGM

Navigate to <new_project_dir>. We will install YGM here. 
Once we have navigated to the directory, we want to create a place to eventually install ygm:

    mkdir ./install

Now, clone YGM and install to the directory we just created.

Note that we *must* use the develop branch of YGM.

    git clone -b develop --single-branch https://github.com/LLNL/ygm.git

    cd ./ygm

    mkdir ./build; cd ./build

    cmake -DCMAKE_INSTALL_PREFIX=<new_project_dir>/install -DYGM_INSTALL=ON ..

    make

    make install

This should install YGM and required dependencies to <new_project_dir>/install/, where it can easily be found later when compiling the CircusTent project. Return to <new_project_dir>

    cd ../..

## Installing CircusTent with on YGM dependency

From "<new_project_dir>", clone the ygm branch of CircusTent.

    git clone -b ygm --single-branch https://github.com/tactcomplabs/circustent.git

    cd ./circustent

Build CircusTent, telling CMake that we want to build the YGM version and where to find an installation of YGM. 

    mkdir ./build; cd ./build

    cmake -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_INSTALL_PREFIX=<new_project_dir>/install -DYGM_INCLUDE_PATH=<new_project_dir>/install/include/ -DENABLE_YGM=ON -DNAIVE_RPC_YGM=ON ../

    make

# Using the benchmark suite

## Running single benchmarks

Generally, the format to run CT_YGM will be 

    <mpi_job_launcher> <launcher_args> <new_project_dir>/circustent/build/src/CircusTent/circustent -b &BENCH -p &NTKS -m 2000000000 -i 20000000

To run benchmarks on a system that uses the SLURM workload manager, use the format

    srun -p &PART -N &NNDS --ntasks &NTKS <new_project_dir>/circustent/build/src/CircusTent/circustent -b &BENCH -p &NTKS -m 2000000000 -i 20000000

where &NNDS is the number of nodes, &NTKS is the *total number of pe's*, and &BENCH is the benchmark you wish to run. 

When running either of the StrideN kernels, please specify the stride length, which is commonly 9 to ensure we are avoiding accessing the same cache line with adjacent operations.

    srun -p &PART -N &NNDS --ntasks &NTKS <new_project_dir>/circustent/build/src/CircusTent/circustent -b &BENCH -p &NTKS -m 2000000000 -i 20000000 -s 9

Note that some kernels (PTRCHASE) may take much longer than the default time limit, which should be specified in the srun arguments. 

## Running multiple benchmarks with run_sbatch_from_csv.py

For convenience when conducting scaling experiments, there is a python script in /circustent/scripts that will construct sbatch jobs from a csv file containing the necessary parameters.

To use this file, change the path in the variable 'host_dir' to wherever the csv parameter file is stored. The results of the benchmark will be stored in a subirectory of host_dir. There must be a file 'job_params.csv' in the host_dir directory.

The account that the job is charged to should also be changed in the script, currently "\<account>".

The format and arguments required by the script are listed in the first line of the csv file.
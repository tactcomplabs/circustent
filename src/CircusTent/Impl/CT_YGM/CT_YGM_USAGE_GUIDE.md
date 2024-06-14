# CircusTent // YGM Guide

# Installing the project

Before proceeding, please note that MPI is *required* for YGM and this project will not run without it. 

If there are no issues with CMake finding the correct installation of MPI, the rest of the YGM installation will 
be handled entirely by CMake. 

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
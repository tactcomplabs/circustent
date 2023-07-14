#!/usr/tce/bin/python3

import sys
import subprocess
import glob
import math
import os

# read all job data from CSV
import csv

job_params = []

fields = []

host_dir = '~/TCL/circustent/scripts/'

with open(host_dir + 'job_params.csv', 'r') as job_data:
    
	job_table = csv.reader(job_data, delimiter=',', skipinitialspace=True)

	fields = next(job_table)
    
	for row in job_table:

		job_params.append(row)



for job in job_params:

	# general parameters every job will need

	if job[4] == '':
		job[4] = "00:30:00"

	if job[1] == '':
		job[1] = "pbatch"


	machine = job[0]
	partition = job[1]
	num_nodes = job[2]
	num_tasks = job[3]
	timelimit = job[4]
	bin_path = job[5]


	# specific to circustent

	if len(job) < 10:
		job.append('9')

	if job[9] == '':
		job[9] = '9'

	benchmark = job[6]
	memSize = job[7]
	iters = job[8]
	stride = job[9]

	print(job)

	for param in job:
		if param == '':
			print("Some required fields are empty for following job request:")
			print(job)
			exit() # maybe this should be continue?
			

	# construct output dirs and files

	job_title = benchmark + "_" + num_nodes

	output_dir = host_dir + "job_output/" + machine + '_' + num_nodes + '/'

	output_path = output_dir + benchmark + ".txt"

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# create job scripts and submit

	if __name__ == "__main__":
    
		proc = subprocess.Popen(['rm', output_path], stdout=subprocess.PIPE).wait()

		with open(output_path, "a") as results_file:
			results_file.write('N: ' + num_nodes + ', ntasks: ' + num_tasks + ', binary: ' +  bin_path + ", memSize: " + memSize + ", iters: " + iters + ", stride: " + stride +'\n')

		with open(job_title + ".job", "w") as batch_file:
			if (benchmark == "STRIDEN_CAS" or benchmark == "STRIDEN_ADD"):
				batch_file.write('#!/bin/bash\n' \
					+ '#SBATCH -N ' + num_nodes + '\n' \
					+ '#SBATCH -t ' + timelimit + '\n' \
					+ '#SBATCH -p ' + partition + '\n'
					+ '#SBATCH -A <account>\n' \
					+ '#SBATCH -o ' + output_path + '\n' \
					+ '#SBATCH -J ' + job_title + '\n\n' \
					+ 'cd ~\n'
					+ 'srun -N ' + num_nodes + ' --ntasks ' + num_tasks + ' ' +  bin_path + ' -b ' + benchmark + " -p " + num_tasks + " -m " + memSize + " -i " + iters + " -s " + stride)
			else:
				batch_file.write('#!/bin/bash\n' \
					+ '#SBATCH -N ' + num_nodes + '\n' \
					+ '#SBATCH -t ' + timelimit + '\n' \
					+ '#SBATCH -p ' + partition + '\n'
					+ '#SBATCH -A <account>\n' \
					+ '#SBATCH -o ' + output_path + '\n' \
					+ '#SBATCH -J ' + job_title + '\n\n' \
					+ 'cd ~\n'
					+ 'srun -N ' + num_nodes + ' --ntasks ' + num_tasks + ' ' +  bin_path + ' -b ' + benchmark + " -p " + num_tasks + " -m " + memSize + " -i " + iters)

		proc = subprocess.Popen(["sbatch", job_title + ".job"], stdout=subprocess.PIPE).wait()

		proc = subprocess.Popen(['rm', job_title + '.job'], stdout=subprocess.PIPE).wait()

		
    
    		

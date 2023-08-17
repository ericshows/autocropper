#!/usr/bin/env python3

import os
import subprocess
# modules for parallel work
import multiprocessing
from joblib import Parallel, delayed
# provides a nice progress bar
import tqdm


server_folder = input("\nPlease drag in a folder to create QC jpgs from tif files and press [Enter]\n\n").rstrip()
QC_folder = os.path.join(server_folder,'QC')


def get_files():
	# traverse root directory, and list directories as dirs and files as files
	file_list = []
	for root, dirs, files in os.walk(server_folder):
		for file in files:
			file_list.append(os.path.join(root, file))

	return file_list

def make_dir():
	# create QC directory
	try:
		os.mkdir(QC_folder)
	except:
		pass

def create_jpg(i):
	# create jpg derivatives
	cmd = 'convert {}[0] -quiet -resize 760x760 -quality 100 {}.jpg'.format(os.path.join(server_folder,i), os.path.join(QC_folder,os.path.basename(i[0:-4])))
	pro = subprocess.call(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid) 

def parallel_processing(file_list):
	# run create_jpg in parallel, using all processing cores
	print('\nWorking...\n')
	num_cores = multiprocessing.cpu_count()
	Parallel(n_jobs=num_cores)(delayed(create_jpg)(file) for file in tqdm.tqdm(file_list))

def main():
	fileget = get_files()
	make_dir()
	parallel_processing(fileget)

if __name__ == '__main__':
	main()
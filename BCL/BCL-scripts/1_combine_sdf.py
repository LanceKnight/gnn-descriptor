# This script will combine the actives and inactives SDF files into one SDF file.

import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm


dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290', '9999']
dataset_list = ['8999']


def concate(input_file1, input_file2, output_file):
	with open(input_file1) as input1:
		with open(input_file2) as input2:
			with open(output_file, 'w+') as output:
				output.write(input1.read())
				output.write('\n')
			with open(output_file, 'a') as output:
				output.write(input2.read())


def combine_data(dataset):
	concate(f'../data/raw/{dataset}_actives_new.sdf', f'../data/raw/{dataset}_inactives_new.sdf', f'../data/BCL_processed_data/{dataset}_combined.sdf')

	print(f'finished combining {dataset}')


if __name__ == '__main__':
	mp.set_start_method('spawn')

	for dataset in dataset_list:
		p=mp.Process(target=combine_data, args=(dataset,))
		p.start()

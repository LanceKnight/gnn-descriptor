# This script add hydrogen, neutralize molecule and filter out those molecules without simple atom types. 
# Molecules that do not match the filtering criteria or gives processing erros are store in unmatched_{dataset}.sdf

import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm


dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290', '9999']
dataset_list = ['8999']

def filter_data(dataset):
	os.system(f'bcl.exe molecule:Filter\
		-input_filenames ../data/BCL_processed_data/id_{dataset}_combined.sdf\
		-output_matched ../data/BCL_processed_data/cleaned_{dataset}.sdf\
		-output_unmatched ../data/BCL_processed_data/unmatched_{dataset}.sdf\
		-add_h\
		-neutralize\
		-defined_atom_types -simple\
		')
	print(f'finished filtering {dataset}')


if __name__ == '__main__':
	mp.set_start_method('spawn')

	for dataset in dataset_list:
		p=mp.Process(target=filter_data, args=(dataset,))
		p.start()

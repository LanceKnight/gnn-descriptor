# This scipt adds id to each molecules in the SDF file for the ease of debugging
import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm


dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290', '9999']
dataset_list = ['8999']

def add_id(dataset):
	counter = 0
	is_first_line = True
	with open(f'../data/BCL_processed_data/{dataset}_combined.sdf') as in_file:
		with open(f'../data/BCL_processed_data/id_{dataset}_combined.sdf', 'w+') as out_file:
			lines = in_file.readlines()
			for line in tqdm(lines):
				if is_first_line == True:
					line = f'-{counter}-'+line
					is_first_line == False

				if '$$$$' in line:
					counter +=1
					is_first_line = True
				else:
					is_first_line = False
			
				out_file.write(line)

	print(f'{dataset}: numer of molecules:{counter}')


if __name__ == '__main__':
	mp.set_start_method('spawn')
	queue = mp.Queue()

	for dataset in dataset_list:
		p=mp.Process(target=add_id, args=(dataset,))
		p.start()

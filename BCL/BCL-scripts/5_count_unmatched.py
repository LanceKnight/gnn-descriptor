# This script counts how many molecules are filtered out

import subprocess
import multiprocessing as mp

dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290', '9999']
dataset_list = ['8999']

def count_unmatch(dataset):

	unmatched_ids = []
	with open(f'../data/BCL_processed_data/unmatched_{dataset}.sdf', 'r') as in_file:
		lines = in_file.readlines()
	for line in lines:
		if line[0] == '-':
			# print(line)
			components = line.split('-')
			# print(components)
			id = int(components[1])
			unmatched_ids.append(id)
	# result = subprocess.run([f"zgrep -c '$$$$' ../data/BCL_processed_data/unmatched_{dataset}.sdf"], stdout=subprocess.PIPE)
	print(f'unmatched_ids = {unmatched_ids}')
	print(f'there are {len(unmatched_ids)} invalid molecules in {dataset}\n')

	with open(f'logs/{dataset}_filtered_ids.txt', 'w+') as out_file:
		for id in unmatched_ids:
			out_file.write(f'{id}\n')


if __name__ == '__main__':
	mp.set_start_method('spawn')
	queue = mp.Queue()

	for dataset in dataset_list:
		p=mp.Process(target=count_unmatch, args=(dataset,))
		p.start()

# This script generates BCL features based on the BCL-feature-config-file
import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm

BCL_feature_config_file = f'BCL/BCL-feature-config/RSR.object' # Change this if you want to use different BCL features

dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290', '9999']
dataset_list = ['8999']

def process_dataset(dataset):
	os.system('mkdir bcl-feats')

	input_filesname=f'../data/BCL_processed_data/cleaned_{dataset}.sdf'

	os.system(f'bcl.exe descriptor:GenerateDataset \
		-source "SdfFile(filename={input_filesname})"\
		-feature_labels {BCL_feature_config_file}\
		-result_labels "Combine(\
								Greater(lhs=Subtract(lhs=Log(Constant(1e+06))\
													, rhs=Log(MiscProperty(Activity,values per molecule=1))\
										   			)\
										,rhs=Constant(3.5)\
										)\
								)"\
		-output bcl-feats/{dataset}-bcl-feat.csv\
		-id_labels "FileID({input_filesname})"\
		')
		

if __name__ == '__main__':
	mp.set_start_method('spawn')
	queue = mp.Queue()

	for dataset in dataset_list:
		p=mp.Process(target=process_dataset, args=(dataset,))
		p.start()






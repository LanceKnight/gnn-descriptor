# This script cleans out intermediate files in BCL_processed_data folder to save space

import os

def clean():
	os.system('rm ../data/BCL_processed_data/*')


if __name__ == '__main__':	
	clean()
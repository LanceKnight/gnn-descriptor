import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm

dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290', '9999']
# dataset_list = ['1798']
dataset_list = ['485290']


def process_dataset(dataset):
    with open(f'feat/{dataset}.RSR.csv') as bcl_feat_file:
        lines = bcl_feat_file.readlines()
        new_data_list = []

        num_data = (dataset_info[dataset]['num_active']+dataset_info[dataset]['num_inactive'])
        id_list = list(range(0,num_data))
        # print(id_list[68744])
        # print(id_list[68745])
        # print(id_list[68746])

        for line in tqdm(lines):
            values = line.split(',')
            id = int(values[0])
            feats = list(map(float,values[1:-1]))
            try:
                id_list.remove(id)
            except Exception as e:
                print(f'*****{dataset}:error id:{id} msg:{e}')

        new_id_list=[]
        for id in id_list:
            if id <dataset_info[dataset]['num_active']:
                label = 'act'
            else:
                label = 'inact'
            new_id_list.append({'id':id, 'label':label})
        print(f'-----{dataset} {len(new_id_list)} were removed)\n')
        # print(f'-----{dataset} {new_id_list})\n')


dataset_info = {
    '435008':{'num_active':233, 'num_inactive':217923},#{'num_active':233, 'num_inactive':217925},
    '1798':{'num_active':187, 'num_inactive':61645},#{'num_active':187, 'num_inactive':61645},
    '435034': {'num_active':362, 'num_inactive':61393},#{'num_active':362, 'num_inactive':61394},
    '1843': {'num_active':172, 'num_inactive':301318},#{'num_active':172, 'num_inactive':301321},
    '2258': {'num_active':213, 'num_inactive':302189},#{'num_active':213, 'num_inactive':302192},
    '463087': {'num_active':703, 'num_inactive':100171},#{'num_active':703, 'num_inactive':100172},
    '488997': {'num_active':252, 'num_inactive':302051},#{'num_active':252, 'num_inactive':302054},
    '2689': {'num_active':172, 'num_inactive':319617},#{'num_active':172, 'num_inactive':319620},
    '485290': {'num_active':278, 'num_inactive':341026},#{'num_active':281, 'num_inactive':341084},
    '9999':{'num_active':37, 'num_inactive':227},
}


if __name__ == '__main__':


    mp.set_start_method('spawn')
    queue = mp.Queue()

    for dataset in dataset_list:
        p=mp.Process(target=process_dataset, args=(dataset,))
        p.start()
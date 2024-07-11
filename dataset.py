from torch_geometric.data import InMemoryDataset
import torch
import os
from tqdm import tqdm
from rdkit import Chem, RDLogger
from argparse import ArgumentParser
import pandas as pd

from featurization import mol2graph, add_bcl_feature

class QSARDataset(InMemoryDataset):
    def __init__(self, root, dataset, model_type, split_scheme, pre_transform):
            self.root = root
            self.dataset = dataset
            self.model_type = model_type
            self.split_scheme = split_scheme
            self.pre_transform = pre_transform
            super(QSARDataset, self).__init__(root, pre_transform=pre_transform)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return [f'scaled-CatDG-{self.dataset}-{self.model_type}.pt']

    def download(self):
        pass

    def process(self):
        print(f'processing dataset {self.dataset}')
        RDLogger.DisableLog('rdApp.*')

        data_list = []
        invalid_id_list = []
        mol_id = 0
        for file_name, label in [(f'{self.dataset}_actives_new.sdf', 1),
                                 (f'{self.dataset}_inactives_new.sdf', 0)]:
            sdf_path = os.path.join(self.root, 'raw', file_name)
            sdf_supplier = Chem.SDMolSupplier(sdf_path)
            for i, mol in tqdm(enumerate(sdf_supplier)):
                data = mol2graph(mol, self.model_type)

                if data.valid == False: # Invalid mol is still included in the dataset since different methods may
                    # generate different invalid methods. The invalid ids will be recorded and removed in get_idx_split()
                    invalid_id_list.append([mol_id, label])

                data.y = torch.tensor([label], dtype=torch.float)
                data.mol_id = torch.tensor([mol_id], dtype=torch.int)
                data_list.append(data)
                mol_id += 1

        if self.pre_transform is not None:
            print('doing pre_transforming...')
            data_list = [self.pre_transform(data) for data in data_list]

        # Save invalid_id_list
        pd.DataFrame(invalid_id_list).to_csv(
            os.path.join(self.processed_dir, f'CatDG-{self.dataset}-{self.model_type}-invalid_id.csv')
            , header=None, index=None)
        data_list= add_bcl_feature(data_list, self.dataset)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    def get_idx_split(self):
        split_dict = torch.load(f'data_split/{self.dataset}_{self.split_scheme}.pt')
        print(f'using {self.split_scheme} split scheme')


        if 'scaffold' in self.split_scheme:
            print(f'using scaffold split scheme')
            # remove invalid ids
            filtered_ids = []
            with open(f'logs/{self.dataset}_filtered_ids.txt', 'r') as in_file:
                lines = in_file.readlines()
                for line in lines:
                    filtered_ids.append(int(line))

            for filtered_id in filtered_ids:
                try:
                    split_dict['train'].remove(filtered_id)
                    split_dict['test'].remove(filtered_id)
                except Exception as e:
                    continue

            new_id_list = []
            for id in split_dict['train']:
                for filtered_id in filtered_ids:
                    if id >= filtered_id:
                        id-=1
                new_id_list.append(id)
            split_dict['train'] = new_id_list

            new_id_list = []
            for id in split_dict['test']:
                for filtered_id in filtered_ids:
                    if id >= filtered_id:
                        id-=1
                new_id_list.append(id)
            split_dict['test'] = new_id_list


            set1 = set(split_dict['train'])
            print(f'no duplicates:{len(set1)==len(split_dict["train"])}')
            set2 = set(split_dict['test'])
            print(f'no duplicates:{len(set2)==len(split_dict["test"])}')




        try:
            invalid_id_list = pd.read_csv(os.path.join(self.processed_dir, f'CatDG-{self.dataset}-{self.model_type}-invalid_id.csv')
                                      , header=None).values.tolist()
            if len(invalid_id_list) == 0:
                print(f'invalid_id_list is empty')

            for id, label in invalid_id_list:
                print(f'checking invalid id {id}')
                if label == 1:
                    print('====warning: a positive label is removed====')
                if id in split_dict['train']:
                    split_dict['train'].remove(id)
                    print(f'found in train and removed')
                if id in split_dict['test']:
                    split_dict['test'].remove(id)
                    print(f'found in test and removed')
        except Exception as e:
            print(f'Cannot open invalid mol file: {e}')

        return split_dict



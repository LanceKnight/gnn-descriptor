import math
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors, rdPartialCharges
import rdkit.Chem.EState as EState
import torch
from torch_geometric.data import Data

from molkgnn_feat import mol2graph as molkgnn_mol2graph

def add_bcl_feature(data_list, dataset):
    '''
    This function adds the BCL feature to the data_list. Those without BCL feature will be filtered out.
    '''

    filtered_ids = []
    with open(f'logs/{dataset}_filtered_ids.txt', 'r') as in_file:
        lines = in_file.readlines()
        for line in lines:
            filtered_ids.append(int(line))

    num_filtered = len(filtered_ids)
    print(f'num_filtered: {num_filtered}')

    with open(f'../bcl-feats/scaled-{dataset}-bcl-feat.csv') as bcl_feat_file:
        lines = bcl_feat_file.readlines()

        if (len(lines) + num_filtered) != len(data_list):
            raise Exception(f'featurization.py::add_bcl_feature():number of mol in bcl feature plus filtered-out-number'
                            f'({(len(lines) + num_filtered)}) does not equal to the number of mol in input ('
                            f'{len(data_list)})')

        new_data_list = []
        id_list = [i for i in range(len(data_list))]

        for line_id, line in enumerate(lines):

            values = line.split(',')
            id = int(values[0])

            for filtered_id in filtered_ids:
                if id >= filtered_id:
                    id+=1

            feats = list(map(float,values[1:-2]))
            bcl_weight = float(values[-1]) # The BCL feature csv file is normalized column-wise. The last column is
            # the unnormalized weight for checking alignment with graph data. The -2 column is the label.
            bcl_y = int(values[-2])
            try:
                y = data_list[id].y
            except:
                raise Exception(f'featurization.py::add_bcl_feature(): id {id} is not in data_list')
            if  y!= bcl_y:
                raise Exception(f'id{id} does not have the same y label. BCL has a label of {bcl_y}, while data has a '
                                f'label of {y}')
            bcl_feat = torch.tensor(feats).unsqueeze(0)
            data_list[id].bcl_feat = bcl_feat
            data_list[id].bcl_weight = bcl_weight
            new_data_list.append(data_list[id])
            try:
                id_list.remove(id)
            except:
                raise Exception(f'featurization.py::add_bcl_feature(): id {id} is not in id_list')
        print(f'len(new_data_list): {len(new_data_list)}')
        for id in filtered_ids:
            try:
                id_list.remove(id)
            except:
                print(f'featurization.py::add_bcl_feature(): id {id} is not in filtered_list')
        if len(id_list) != 0:
            raise Exception(f'featurization.py::add_bcl_feature(): id_list is not empty. id_list: {id_list}')

        return new_data_list



def featurization_A(mol):
    '''
    This is the featurization used for SchNet and SphereNet
    '''
    try:
        conformer = mol.GetConformer()
        adj = rdkit.Chem.GetAdjacencyMatrix(mol)
        adj = np.triu(np.array(adj, dtype=int))  # keeping just upper triangular entries from sym matrix
        array_adj = np.array(np.nonzero(adj), dtype=int)  # indices of non-zero values in adj matrix
        edge_index = np.zeros((2, 2 * array_adj.shape[1]), dtype=int)  # placeholder for undirected edge list
        edge_index[:, ::2] = array_adj
        edge_index[:, 1::2] = np.flipud(array_adj)

        atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
        node_features = np.array(
            [atom.GetAtomicNum() for atom in atoms])  # Z
        positions = np.array(
            [conformer.GetAtomPosition(atom.GetIdx()) for atom in atoms])  # xyz positions
        edge_index, Z, pos = edge_index, node_features, positions
        data = Data(
            x=torch.as_tensor(Z).unsqueeze(1),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            atomic_num=torch.as_tensor(Z, dtype=torch.int),
            pos=torch.as_tensor(pos, dtype=torch.float),
            valid=True
        )

        return data
    except Exception as e:
        print('Error in featurization_A: ', e)
        return Data(x=torch.zeros(1, 1),
                    edge_index=torch.zeros(2, 0),
                    pos=torch.zeros(1, 3),
                    valid=False  # Indicate mol is not valid
                    )


def one_hot_vector(val, lst):
	'''
	Converts a value to a one-hot vector based on options in lst
	'''
	if val not in lst:
		val = lst[-1]
	return map(lambda x: x == val, lst)



def get_atom_rep(atom):
    features = []
    # H, C, N, O, F, Si, P, S, Cl, Br, I, other
    features += one_hot_vector(atom.GetAtomicNum(), [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 999])
    features += one_hot_vector(len(atom.GetNeighbors()), list(range(1, 5)))

    features.append(atom.GetFormalCharge())
    features.append(atom.IsInRing())
    features.append(atom.GetIsAromatic())
    features.append(atom.GetExplicitValence())
    features.append(atom.GetMass())

    # Add Gasteiger charge and set to 0 if it is NaN or Infinite
    gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
    if math.isnan(gasteiger_charge) or math.isinf(gasteiger_charge):
        gasteiger_charge = 0
    features.append(gasteiger_charge)

    # Add Gasteiger H charge and set to 0 if it is NaN or Infinite
    gasteiger_h_charge = float(atom.GetProp('_GasteigerHCharge'))
    if math.isnan(gasteiger_h_charge) or math.isinf(gasteiger_h_charge):
        gasteiger_h_charge = 0

    features.append(gasteiger_h_charge)
    return features

def get_extra_atom_feature(all_atom_features, mol):
    '''
    Get more atom features that cannot be calculated only with atom,
    but also with mol
    :param all_atom_features:
    :param mol:
    :return:
    '''
    # Crippen has two parts: first is logP, second is Molar Refactivity(MR)
    all_atom_crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    all_atom_TPSA_contrib = rdMolDescriptors._CalcTPSAContribs(mol)
    all_atom_ASA_contrib = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]
    all_atom_EState = EState.EStateIndices(mol)

    new_all_atom_features = []
    for atom_id, feature in enumerate(all_atom_features):
        crippen_logP = all_atom_crippen[atom_id][0]
        crippen_MR = all_atom_crippen[atom_id][1]
        atom_TPSA_contrib = all_atom_TPSA_contrib[atom_id]
        atom_ASA_contrib = all_atom_ASA_contrib[atom_id]
        atom_EState = all_atom_EState[atom_id]

        feature.append(crippen_logP)
        feature.append(crippen_MR)
        feature.append(atom_TPSA_contrib)
        feature.append(atom_ASA_contrib)
        feature.append(atom_EState)

        new_all_atom_features.append(feature)
    return new_all_atom_features


def featurization_B(mol):
    '''
    This is the featurization used for GCN
    '''
    try:
        conf = mol.GetConformer()

        atom_pos = []
        atomic_num_list = []
        all_atom_features = []

        # Get atom attributes and positions
        rdPartialCharges.ComputeGasteigerCharges(mol)

        for i, atom in enumerate(mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            atomic_num_list.append(atomic_num)
            atom_feature = get_atom_rep(atom)

            # Add atom position
            atom_pos.append([conf.GetAtomPosition(i).x,
                             conf.GetAtomPosition(i).y,
                             conf.GetAtomPosition(i).z])
            all_atom_features.append(atom_feature)
        # Add extra features that are needs to calculate using mol
        all_atom_features = get_extra_atom_feature(all_atom_features, mol)

        # Get bond attributes
        edge_list = []
        edge_attr_list = []
        for idx, bond in enumerate(mol.GetBonds()):
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_attr = []
            bond_attr += one_hot_vector(
                bond.GetBondTypeAsDouble(),
                [1.0, 1.5, 2.0, 3.0]
            )

            is_aromatic = bond.GetIsAromatic()
            is_conjugate = bond.GetIsConjugated()
            is_in_ring = bond.IsInRing()
            bond_attr.append(is_aromatic)
            bond_attr.append(is_conjugate)
            bond_attr.append(is_in_ring)

            edge_list.append((i, j))
            edge_attr_list.append(bond_attr)

            edge_list.append((j, i))
            edge_attr_list.append(bond_attr)
        x = torch.tensor(all_atom_features, dtype=torch.float32)
        p = torch.tensor(atom_pos, dtype=torch.float32)
        edge_index = torch.tensor(edge_list).t().contiguous().int()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        atomic_num = torch.tensor(atomic_num_list, dtype=torch.int)

        data = Data(x=x,
                    p=p,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    atomic_num=atomic_num,
                    valid=True)
        return data
    except Exception as e:
        print(f'error in featurization B:{e}')
        return Data(x=torch.zeros([1, 28]),
                    p=torch.zeros([1, 3]),
                    edge_index=torch.zeros([2,0], dtype=torch.int),
                    edge_attr=torch.zeros([0, 7]),
                    atomic_num=torch.zeros([1]),
                    valid = False
        )

def featurization_C(mol):
    try:
        res =  molkgnn_mol2graph(mol)
    except Exception as e:
        print(f'error in featurization C:{e}')
        return Data(x=torch.zeros([1, 28]), p=torch.zeros([1,3]), edge_index=torch.zeros([2,0]),
                edge_attr=torch.zeros([1, 28]), atomic_num=torch.zeros([1]),
                    valid = False)
    return res





def mol2graph(mol, model_type):
    if model_type == 'schnet' or model_type == 'spherenet':
        return featurization_A(mol)
    elif model_type == 'gcn':
        return featurization_B(mol)
    elif model_type == 'mlp':
        return Data(valid=True)
    elif model_type == 'molkgnn':
        return featurization_C(mol)
    else:
        raise ValueError(f'gnn_type {model_type} not supported')


if __name__ == "__main__":

    smiles = 'CC'
    mol = Chem.MolFromSmiles(smiles)
    res = AllChem.EmbedMultipleConfs(mol, numConfs=1)

    model_type = 'gcn'
    data = mol2graph(mol, model_type)
    print(data)
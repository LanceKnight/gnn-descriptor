import configparser
from argparse import ArgumentParser
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import torch
import random
import numpy as np
import os

from dataset import QSARDataset
from molkgnn_feat import ToXAndPAndEdgeAttrForDeg
from evaluation import calculate_logAUC, cal_EF, cal_MAP, cal_DCG, cal_BEDROC_score
from models.GCN.GCN import GCN_Model
from models.SchNet.SchNet import SchNet
from models.MolKGNN.MolKGNNNet import MolKGNNNet
from models.SphereNet.SphereNet import SphereNet
from models.MLP.mlp import MLP
from loader import get_train_loader, get_test_loader
from scheduler import get_scheduler, get_lr
from utils.rank_prediction import rank_prediction
from utils.plot_loss import plot_epoch

import warnings
warnings.filterwarnings("ignore")

def train(model, loader, optimizer, scheduler, device,  loss_fn):
    model.train()
    all_loaders = tqdm(loader, miniters=100)
    loss_list = []

    for i, batch in enumerate(all_loaders):
        batch.to(device)
        y_pred = model(batch)
        loss= loss_fn(y_pred.view(-1), batch.y.view(-1).float() )
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    loss = np.mean(loss_list)
    return loss

def test(model, loader, device, type, save_result=False):
    model.eval()
    filename = f'result/per_molecule_pred_of_{type}_set.res'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    all_pred_y = []
    all_true_y = []

    for i, batch in enumerate(tqdm(loader)):
        batch.to(device)
        pred_y = model(batch).cpu().view(-1).detach().numpy()
        true_y = batch.y.view(-1).cpu().numpy()
        for j, y in enumerate(pred_y):
            all_pred_y.append(pred_y[j])
            all_true_y.append(true_y[j])

    with open(filename, 'w') as out_file:
        if save_result:
            for k, y in enumerate(all_pred_y):
                out_file.write(f'{all_pred_y[k]}\ty={all_true_y[k]}\n')

    if save_result:
        rank_prediction(type)
    all_pred_y = np.array(all_pred_y)
    all_true_y = np.array(all_true_y)
    logAUC = calculate_logAUC(all_true_y, all_pred_y)
    EF = cal_EF(all_true_y, all_pred_y, 100)
    MAP = cal_MAP(all_true_y, all_pred_y)
    DCG = cal_DCG(all_true_y, all_pred_y, 100)
    BEDROC = cal_BEDROC_score(all_true_y, all_pred_y)
    return logAUC, EF, MAP, DCG, BEDROC

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/dummy.cfg') # Specify config file
    parser.add_argument('--test', action='store_true') # Skip running and just test
    parser.add_argument('--no_train_eval', action='store_true') # Skip evaluation at training time

    args = parser.parse_args()
    config_file = args.config
    config = configparser.ConfigParser()
    config.read(config_file)

    # ===============NEED TO MODIFY=================
    num_epochs = int(config['TRAIN']['num_epochs'])
    seed = int(config['GENERAL']['seed'])
    num_workers = int(config['GENERAL']['num_workers'])
    dataset_name = config['DATA']['dataset_name']
    batch_size = int(config['TRAIN']['batch_size'])
    model_type = config['MODEL']['model_type']
    split_scheme = config['DATA']['split_scheme']
    print(f'====Model: {model_type}====')
    # ==============================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if model_type== 'molkgnn':
        transform = ToXAndPAndEdgeAttrForDeg()
    else:
        transform = None
    dataset = QSARDataset(root=config['DATA']['root'],
                          dataset=dataset_name,
                          model_type=model_type,
                          split_scheme=split_scheme,
                          pre_transform=transform)
    train_id_list = dataset.get_idx_split()['train']
    test_id_list =  dataset.get_idx_split()['test']

    train_dataset = dataset[train_id_list]
    test_dataset = dataset[test_id_list]
    print(f'full dataset size:{len(dataset)}')

    train_loader = get_train_loader(train_dataset, batch_size=batch_size, num_workers=num_workers, seed=seed, drop_last=True)
    test_loader = get_test_loader(test_dataset, batch_size=batch_size, num_workers=num_workers, seed=seed)
    print(f'len(train_loader)={len(train_loader)}\n'
          f'len(test_loader)={len(test_loader)}')


    if model_type == 'spherenet':

        model = SphereNet(
                energy_and_force=config.getboolean('MODEL', 'energy_and_force'),  # False
                cutoff=float(config['MODEL']['cutoff']),  # 5.0
                num_layers=int(config['MODEL']['num_layers']),  # 4
                hidden_channels=int(config['MODEL']['hidden_channels']),  # 128
                out_channels=int(config['MODEL']['out_channels']),  # 1
                int_emb_size=int(config['MODEL']['int_emb_size']),  # 64
                basis_emb_size_dist=int(config['MODEL']['basis_emb_size_dist']),  # 8
                basis_emb_size_angle=int(config['MODEL']['basis_emb_size_angle']),  # 8
                basis_emb_size_torsion=int(config['MODEL']['basis_emb_size_torsion']),  # 8
                out_emb_channels=int(config['MODEL']['out_emb_channels']),  # 256
                num_spherical=int(config['MODEL']['num_spherical']),  # 7
                num_radial=int(config['MODEL']['num_radial']),  # 6
                envelope_exponent=int(config['MODEL']['envelope_exponent']),  # 5
                num_before_skip=int(config['MODEL']['num_before_skip']),  # 1
                num_after_skip=int(config['MODEL']['num_after_skip']),  # 2
                num_output_layers=int(config['MODEL']['num_output_layers']),  # 3
                with_bcl=config.getboolean('MODEL', 'with_bcl'),  # False
                bcl_dim=int(config['MODEL']['bcl_dim'])  # 64
            ).to(device)
    elif model_type == 'molkgnn':
        model = MolKGNNNet(num_layers=int(config['MODEL']['num_layers']),
                           num_kernel1_1hop=int(config['MODEL']['num_kernel1_1hop']),
                           num_kernel2_1hop=int(config['MODEL']['num_kernel2_1hop']),
                           num_kernel3_1hop=int(config['MODEL']['num_kernel3_1hop']),
                           num_kernel4_1hop=int(config['MODEL']['num_kernel4_1hop']),
                           num_kernel1_Nhop=int(config['MODEL']['num_kernel1_Nhop']),
                           num_kernel2_Nhop=int(config['MODEL']['num_kernel2_Nhop']),
                           num_kernel3_Nhop=int(config['MODEL']['num_kernel3_Nhop']),
                           num_kernel4_Nhop=int(config['MODEL']['num_kernel4_Nhop']),
                           predefined_kernelsets=True,
                           x_dim=int(config['MODEL']['x_dim']),
                           p_dim=3,
                           edge_attr_dim=int(config['MODEL']['edge_attr_dim']),
                           drop_ratio=float(config['MODEL']['drop_ratio']),
                           graph_embedding_dim=int(config['MODEL']['graph_embedding_dim']),
                           with_bcl=config.getboolean('MODEL', 'with_bcl'),  # False
                           bcl_dim=int(config['MODEL']['bcl_dim']),  # 64
                           out_channels=int(config['MODEL']['out_channels']),  # 32
                           ).to(device)

    elif model_type == 'gcn':
        model = GCN_Model(in_channels=int(config['MODEL']['in_channels']),
                          hidden_channels=int(config['MODEL']['hidden_channels']),
                          num_layers=int(config['MODEL']['num_layers']),
                          with_bcl=config.getboolean('MODEL', 'with_bcl'),
                          bcl_dim=int(config['MODEL']['bcl_dim']),
                      ).to(device)

    elif model_type == 'schnet':
        model = SchNet(energy_and_force=config.getboolean('MODEL', 'energy_and_force'),
                       cutoff=float(config['MODEL']['cutoff']),
                       num_layers=int(config['MODEL']['num_layers']),
                       hidden_channels=int(config['MODEL']['hidden_channels']),
                       num_filters=int(config['MODEL']['num_filters']),
                       num_gaussians=int(config['MODEL']['num_gaussians']),
                       out_channels=int(config['MODEL']['out_channels']),
                       with_bcl=config.getboolean('MODEL', 'with_bcl'),
                       bcl_dim = int(config['MODEL']['bcl_dim']),
                       ).to(device)

    elif model_type == 'mlp':
        model = MLP(bcl_feat_dim=int(config['MODEL']['bcl_feat_dim']),
                    hidden_dim=int(config['MODEL']['hidden_dim']),
                    ).to(device)

    loss_fn = BCEWithLogitsLoss()
    print(f'num of data in {dataset_name}={len(dataset)}')



    optimizer = AdamW(model.parameters())
    scheduler = get_scheduler(optimizer, config, train_dataset)

    filename = f'saved_models/trained_{dataset}_{model_type}_{split_scheme}.pt'
    # Training
    if not args.test:
        print('training...')
        total_epochs = (range(num_epochs))

        interations=1
        os.makedirs('logs', exist_ok=True)
        with open(f'logs/loss_per_epoch.log', 'w+') as out_file:
            for epoch in total_epochs:
                loss = train(model, train_loader, optimizer, scheduler, device, loss_fn)
                lr = get_lr(optimizer)
                if not args.no_train_eval:
                    train_logAUC, train_EF, train_MAP, train_DCG, train_BEDROC = test(model=model, loader=train_loader,
                                                                             device=device, type = 'train',
                                                                                       save_result=True)
                    if epoch % 10==0:
                        print(f'current_epoch={epoch} train_logAUC={train_logAUC} lr={lr}')
                    out_file.write(f'{epoch}\tloss={loss}\tlogAUC={train_logAUC}\tlr={lr}\t\n')
                else:
                    if epoch % 10==0:
                        print(f'current_epoch={epoch} lr={lr}')
                    out_file.write(f'{epoch}\tloss={loss}\tlr={lr}\t\n')


        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(model.state_dict(), filename)
        # Save the loss_per_epoch to a file

        print(f'finished training')


    # Testing
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
    else:
        raise Exception(f'No trained model found. Please train the model first')
    print('testing...')
    logAUC, EF, MAP, DCG, BEDROC  = test(model, test_loader, device, type='test', save_result=True)
    print(f'logAUC={logAUC}\tEF={EF}\tMAP={MAP}\tDCG={DCG}\tBEDROC={BEDROC}')
    with open('result/test_result.res', 'w+') as result_file:# Save the result to a file
        result_file.write(f'logAUC={logAUC}\tEF={EF}\tMAP={MAP}\tDCG={DCG}\tBEDROC={BEDROC}')

    # Plot the loss, performance, lr per epoch
    plot_epoch()
from .gcn import GCN as Encoder
from torch_geometric.nn import global_add_pool
import torch
from torch.nn import Linear, ReLU, Dropout, BatchNorm1d


class GCN_Model(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_layers,
            dropout = 0.0,
            act = 'relu',
            act_first = False,
            act_kwargs= None,
            norm = None,
            norm_kwargs = None,
            jk= None,
            with_bcl = False,
            bcl_dim = 391
    ):
        super(GCN_Model, self).__init__()
        self.encoder = Encoder(
            in_channels = in_channels,
            hidden_channels = hidden_channels,
            num_layers = num_layers,
            dropout = dropout,
            act = act,
            act_first = act_first,
            act_kwargs= act_kwargs,
            norm = norm,
            norm_kwargs = norm_kwargs,
            jk= jk
        )
        self.pool = global_add_pool


        self.lin1 = Linear(hidden_channels, 64)
        self.lin2 = Linear(64, 1)
        self.activate_func = ReLU()
        self.ff_dropout = Dropout(p=0.25)
        self.with_bcl = with_bcl

        self.ffn_dropout = Dropout(p=0.25)
        self.bcl_lin1 = Linear(hidden_channels+64, 64)
        self.lin1 = Linear(hidden_channels, 64)
        self.lin2 = Linear(64, 1)
        self.activate_func = ReLU()
        self.ff_dropout = Dropout(p=0.25)
        self.with_bcl = with_bcl
        self.batch_norm = BatchNorm1d(64)
        self.mol_lin1 = Linear(bcl_dim, 64)
        self.mol_lin2 = Linear(64, 64)
        self.mol_dropout = Dropout(p=0.25)
        self.mol_batch_norm = BatchNorm1d(64)
    def forward(self, batch_data):
        node_embedding = self.encoder(batch_data.x, batch_data.edge_index)
        graph_embedding = self.pool(node_embedding, batch_data.batch)
        graph_embedding = self.ffn_dropout(graph_embedding)
        if self.with_bcl:
            mol_embedding = self.mol_lin2(self.mol_batch_norm(self.activate_func(self.mol_dropout(self.mol_lin1(
                batch_data.bcl_feat)))))
            cat_embedding = torch.cat((graph_embedding, mol_embedding), 1)
            prediction = self.lin2(self.batch_norm(self.activate_func(self.ff_dropout(self.bcl_lin1(cat_embedding)))))
        else:
            prediction = self.lin2(self.batch_norm(self.activate_func(self.ff_dropout(self.lin1(graph_embedding)))))
        return prediction




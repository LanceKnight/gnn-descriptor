from .schnet import SchNet as Encoder
import torch
from torch.nn import Dropout, Linear, ReLU, BatchNorm1d

class SchNet(torch.nn.Module):

    def __init__(
            self, energy_and_force=False, cutoff=10.0, num_layers=6, hidden_channels=128, num_filters=128,
            num_gaussians=50, out_channels=32, with_bcl = False, bcl_dim = 391):
        super(SchNet, self).__init__()
        self.encoder = Encoder(energy_and_force=energy_and_force,
                               cutoff=cutoff,
                               num_layers=num_layers,
                               hidden_channels=hidden_channels,
                               num_filters=num_filters,
                               num_gaussians=num_gaussians,
                               out_channels=out_channels
                            )
        self.ffn_dropout = Dropout(p=0.25)
        self.bcl_lin1 = Linear(out_channels+64, 64)
        self.lin1 = Linear(out_channels, 64)
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
        batch_data.z = batch_data.x.squeeze()
        graph_embedding = self.encoder(batch_data)

        graph_embedding = self.ffn_dropout(graph_embedding)
        if self.with_bcl:
            mol_embedding = self.mol_lin2(self.mol_batch_norm(self.activate_func(self.mol_dropout(self.mol_lin1(
                batch_data.bcl_feat)))))
            cat_embedding = torch.cat((graph_embedding, mol_embedding), 1)
            prediction = self.lin2(self.batch_norm(self.activate_func(self.ff_dropout(self.bcl_lin1(cat_embedding)))))
        else:
            prediction = self.lin2(self.batch_norm(self.activate_func(self.ff_dropout(self.lin1(graph_embedding)))))
        return prediction


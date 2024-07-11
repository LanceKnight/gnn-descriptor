from torch.nn import Module, Linear, Sequential, Sigmoid, Dropout, BatchNorm1d
import torch


class MLP(Module):
    def __init__(self, bcl_feat_dim, hidden_dim = 32):
        super().__init__()


        self.mlp = Sequential(
            BatchNorm1d(bcl_feat_dim),
            Dropout(0.05),
            Linear(bcl_feat_dim, hidden_dim),
            Dropout(0.35),
            Sigmoid(),
            Linear(hidden_dim, 1)
        )


    def forward(self, data):
        x = self.mlp(data.bcl_feat)
        return x


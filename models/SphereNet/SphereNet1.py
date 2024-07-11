from dig.threedgraph.method import SphereNet as Encoder
from torch.nn import Dropout, Linear, ReLU
import torch

class SphereNet(torch.nn.Module):
    r"""
        codes adapted from https://github.com/keiradams/ChIRo
         The spherical message passing neural network SphereNet from the
         `"Spherical Message Passing for 3D Graph Networks"
         <https://arxiv.org/abs/2102.05013>`_ paper.

        Args:
            energy_and_force (bool, optional): If set to :obj:`True`,
            will predict energy and take the negative of the derivative
            of the energy with respect to the atomic positions as
            predicted forces. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (
            default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (
            default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (
            default: :obj:`1`)
            int_emb_size (int, optional): Embedding size used for
            interaction triplets. (default: :obj:`64`)
            basis_emb_size_dist (int, optional): Embedding size used in
            the basis transformation of distance. (default: :obj:`8`)
            basis_emb_size_angle (int, optional): Embedding size used in
            the basis transformation of angle. (default: :obj:`8`)
            basis_emb_size_torsion (int, optional): Embedding size used
            in the basis transformation of torsion. (default: :obj:`8`)
            out_emb_channels (int, optional): Embedding size used for
            atoms in the output block. (default: :obj:`256`)
            num_spherical (int, optional): Number of spherical
            harmonics. (default: :obj:`7`)
            num_radial (int, optional): Number of radial basis
            functions. (default: :obj:`6`)
            envelop_exponent (int, optional): Shape of the smooth
            cutoff. (default: :obj:`5`)
            num_before_skip (int, optional): Number of residual layers
            in the interaction blocks before the skip connection. (
            default: :obj:`1`)
            num_after_skip (int, optional): Number of residual layers in
            the interaction blocks before the skip connection. (default:
            :obj:`2`)
            num_output_layers (int, optional): Number of linear layers
            for the output blocks. (default: :obj:`3`)
            act_name: (function, optional): The activation funtion. (default:
            :obj:`swish`)
            output_init: (str, optional): The initialization fot the
            output. It could be :obj:`GlorotOrthogonal` and
            :obj:`zeros`. (default: :obj:`GlorotOrthogonal`)
        """

    def __init__(
            self, energy_and_force=False, cutoff=5.0, num_layers=4,
            hidden_channels=128, out_channels=1, int_emb_size=64,
            basis_emb_size_dist=8, basis_emb_size_angle=8,
            basis_emb_size_torsion=8, out_emb_channels=256,
            num_spherical=3, num_radial=6, envelope_exponent=5,
            num_before_skip=1, num_after_skip=2, num_output_layers=3, with_bcl = False, bcl_dim = 391,):
        super(SphereNet, self).__init__()
        self.encoder = Encoder(
                                energy_and_force=energy_and_force,  # False
                                cutoff=cutoff,  # 5.0
                                num_layers=num_layers,  # 4
                                hidden_channels=hidden_channels,  # 128
                                out_channels=out_channels,  # 1
                                int_emb_size=int_emb_size,  # 64
                                basis_emb_size_dist=basis_emb_size_dist,  # 8
                                basis_emb_size_angle=basis_emb_size_angle,  # 8
                                basis_emb_size_torsion=basis_emb_size_torsion,  # 8
                                out_emb_channels=out_emb_channels,  # 256
                                num_spherical=num_spherical,  # 7
                                num_radial=num_radial,  # 6
                                envelope_exponent=envelope_exponent,  # 5
                                num_before_skip=num_before_skip,  # 1
                                num_after_skip=num_after_skip,  # 2
                                num_output_layers=num_output_layers,  # 3
                                # act_name='swish',
                                # output_init=output_init,
                                # use_node_features=use_node_features,
                                # MLP_hidden_sizes=MLP_hidden_sizes,  # [] for contrastive
                            )
        self.ffn_dropout = Dropout(p=0.25)
        self.bcl_lin1 = Linear(out_channels + bcl_dim, 64)
        self.lin1 = Linear(out_channels, 64)
        self.lin2 = Linear(64, 1)
        self.activate_func = ReLU()
        self.ff_dropout = Dropout(p=0.25)
        self.with_bcl = with_bcl
    def forward(self, batch_data):
        batch_data.z = batch_data.x.squeeze()
        graph_embedding = self.encoder(batch_data)
        graph_embedding = self.ffn_dropout(graph_embedding)
        if self.with_bcl:
            cat_embedding = torch.cat((graph_embedding, batch_data.bcl_feat), 1)
            prediction = self.lin2(self.activate_func(self.ff_dropout(self.bcl_lin1(cat_embedding))))
        else:
            prediction = self.lin2(self.activate_func(self.ff_dropout(self.lin1(graph_embedding))))
        return prediction



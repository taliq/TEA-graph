import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU, SELU, PReLU, GELU, Dropout, Conv1d, ELU, LeakyReLU
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.nn import LayerNorm, MessageNorm, PairNorm, BatchNorm, GraphSizeNorm
from torch_geometric.utils import degree
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import softmax
# from torch_geometric.transforms import ToSparseTensor
from torch_sparse import SparseTensor
from torch_scatter import scatter
from torch_sparse import set_diag

from tqdm import tqdm
from models.model_utils import weight_init

class BasicLinear_module(torch.nn.Module):

    def __init__(self, in_f, out_f, dropout_rate):

        super(BasicLinear_module, self).__init__()
        self.Linear = Linear(in_f, out_f)
        #self.LayerNorm = LayerNorm([1] * out_f)
        self.LeakyReLU = LeakyReLU(negative_slope=0.2)
        self.Dropout = Dropout(dropout_rate)

    def reset_parameters(self):

        self.Linear.apply(weight_init)
        #self.LayerNorm.reset_parameters()

    def forward(self, input_x, batch):

        out_x = self.Linear(input_x)
        #out_x = self.LayerNorm(out_x)
        out_x = self.LeakyReLU(out_x)
        out_x = self.Dropout(out_x)

        return out_x

class postprocess(torch.nn.Module):

    def __init__(self, graph_feature_dim, graph_layer_num, last_input_dim, post_layer_num, dropout_rate):
        super(postprocess, self).__init__()

        #poststartlayer = 1280
        poststartlayer = graph_feature_dim * graph_layer_num + last_input_dim
        postlayerpreset = [int(poststartlayer / 2.0),
                           int(poststartlayer / 4.0),
                           int(poststartlayer / 8.0)]
        self.postlayernum = []
        self.postlayernum.append(poststartlayer)
        for i in range(0, post_layer_num):
            self.postlayernum.append(postlayerpreset[i])


        postlayer_list = [BasicLinear_module(in_f, out_f, dropout_rate)
                           for in_f, out_f in zip(self.postlayernum, self.postlayernum[1:])]
        self.postlayer_blocks = nn.ModuleList(postlayer_list)

        #postlayer_blocks.append(Linear(self.postlayernum[-1], 512))

        """
        if Argument.DatasetType == "BORAME_Meta":
            postlayer_blocks.append(Linear(self.postlayernum[-1], 5))
        else:
            postlayer_blocks.append(Linear(self.postlayernum[-1], 1))
        """

    def reset_parameters(self):

        for i in range(len(self.postlayer_blocks)):
            self.postlayer_blocks[i].reset_parameters()

    def forward(self, input_x, batch):

        postprocessed_data = input_x
        for i in range(len(self.postlayer_blocks)):
            postprocessed_data = self.postlayer_blocks[i](postprocessed_data, batch)

        return postprocessed_data

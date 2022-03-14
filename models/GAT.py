# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU, SELU, PReLU, GELU, Dropout, Conv1d, ELU, LeakyReLU, LayerNorm
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.nn import MessageNorm, PairNorm, BatchNorm, GraphSizeNorm
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
from models.model_utils import decide_loss_type
from torch_geometric.nn.inits import kaiming_uniform

from models.pre_layer import preprocess
from models.post_layer import postprocess


class GAT_module(torch.nn.Module):

    def __init__(self, input_dim, output_dim, head_num, dropedge_rate, graph_dropout_rate, loss_type, with_edge, simple_distance, norm_type):

        super(GAT_module, self).__init__()
        self.conv = GATConv(input_dim, output_dim, heads=head_num, dropout=dropedge_rate, with_edge=with_edge, simple_distance=simple_distance)
        self.norm_type = norm_type
        #self.gbn = GraphSizeNorm()
        #self.bn = LayerNorm([1.0] * (output_dim * int(self.conv.heads)))
        if norm_type == "layer":
            self.bn = LayerNorm(output_dim * int(self.conv.heads))
        else:
            self.bn = BatchNorm(output_dim * int(self.conv.heads))
        self.prelu = decide_loss_type(loss_type, output_dim * int(self.conv.heads))
        self.dropout_rate = graph_dropout_rate
        self.with_edge = with_edge

    def reset_parameters(self):

        self.conv.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_attr, edge_index, batch):

        if self.training:
            drop_node_mask = x.new_full((x.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_node_mask = torch.bernoulli(drop_node_mask)
            drop_node_mask = torch.reshape(drop_node_mask, (1, drop_node_mask.shape[0]))
            drop_node_feature = x * drop_node_mask

            drop_edge_mask = edge_attr.new_full((edge_attr.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_edge_mask = torch.bernoulli(drop_edge_mask)
            drop_edge_mask = torch.reshape(drop_edge_mask, (1, drop_edge_mask.shape[0]))
            drop_edge_attr = edge_attr * drop_edge_mask
        else:
            drop_node_feature = x
            drop_edge_attr = edge_attr

        if self.with_edge == "Y":
            x_before = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=drop_edge_attr)
        else:
            x_before = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=None)
        #x_before = self.gbn(x_before, batch)
        out_x_temp = 0
        for c, item in enumerate(torch.unique(batch)):
            if self.norm_type == "layer":
                temp = self.bn(x_before[batch == item])
            else:
                temp = self.bn(x_before[batch == item] / sum(batch == item))

            if c == 0:
                out_x_temp = temp
            else:
                out_x_temp = torch.cat((out_x_temp, temp), 0)

        x_after = self.prelu(out_x_temp)

        return x_after

class GAT(torch.nn.Module):

    def __init__(self, dropout_rate, dropedge_rate, Argument):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        dim = Argument.initial_dim
        self.dropout_rate = dropout_rate
        self.dropedge_rate = dropedge_rate
        self.Argument = Argument
        self.heads_num = Argument.attention_head_num
        self.noise_var = Argument.noise_var_value
        self.noise_portion = Argument.noise_node_portion
        self.include_edge_feature = Argument.with_distance
        self.layer_num = Argument.number_of_layers
        self.graph_dropout_rate = Argument.graph_dropout_rate
        self.residual = Argument.residual_connection
        self.with_noise = Argument.with_noise
        self.norm_type = Argument.norm_type

        postNum = 0
        self.preprocess = preprocess(Argument)
        #self.conv1 = GAT_module(256, dim, self.heads_num, self.dropedge_rate, self.dropout_rate, Argument.loss_type, with_edge=Argument.with_distance)
        #postNum += int(self.heads_num)
        self.conv_list = nn.ModuleList([GAT_module(dim * self.heads_num, dim, self.heads_num, self.dropedge_rate,
                                                   self.graph_dropout_rate, Argument.loss_type,
                                                   with_edge=Argument.with_distance,
                                                   simple_distance=Argument.simple_distance,
                                                   norm_type=Argument.norm_type) for _ in
                                        range(int(Argument.number_of_layers))])
        #self.conv_list = nn.ModuleList([GAT_module(dim * self.heads_num, dim, self.heads_num, self.dropedge_rate, self.graph_dropout_rate, Argument.loss_type, with_edge=Argument.with_distance, simple_distance=Argument.simple_distance) for _ in range(int(Argument.number_of_layers/2.0))])
        postNum += int(self.heads_num) * len(self.conv_list)

        self.postprocess = postprocess(dim * self.heads_num, self.layer_num, dim * self.heads_num, Argument.postlayernum, dropout_rate)

        if self.with_noise == "Y":
            self.node_denoise_layer = [nn.Linear(dim * self.heads_num, dim * self.heads_num), nn.ReLU(), nn.Linear(dim * self.heads_num, 1792)]
            self.node_denoise_layer = nn.Sequential(*self.node_denoise_layer)

            self.edge_denoise_layer = [nn.Linear(dim * self.heads_num, dim * self.heads_num), nn.ReLU(), nn.Linear(dim * self.heads_num, 2)]
            self.edge_denoise_layer = nn.Sequential(*self.edge_denoise_layer)
            #self.risk_prediction_layer = nn.Linear(self.postprocess.postlayernum[-1], 1)

        self.risk_prediction_layer = nn.Linear(self.postprocess.postlayernum[-1], 1)

    def reset_parameters(self):

        self.preprocess.reset_parameters()
        #self.conv1.reset_parameters()
        for i in range(int(self.Argument.number_of_layers)):
            self.conv_list[i].reset_parameters()
        self.postprocess.reset_parameters()
        self.risk_prediction_layer.apply(weight_init)
        if self.with_noise == "Y":
            self.node_denoise_layer.apply(weight_init)
            self.edge_denoise_layer.apply(weight_init)

    def forward(self, data):

        original_x = data.x
        original_edge = data.edge_attr
        row, col, _ = data.adj_t.coo()

        preprocessed_input, preprocess_edge_attr = self.preprocess(data)
        batch = data.batch

        x0_glob = global_mean_pool(preprocessed_input, batch)
        x_concat = x0_glob

        #x_temp_out = self.conv1(preprocessed_input, preprocess_edge_attr, data.adj_t, batch)
        #x_glob = global_mean_pool(x_temp_out, batch)
        #x_concat = torch.cat((x_concat, x_glob), 1)
        #x_out = x_temp_out + preprocessed_input
        #final_x = x_out
        attention_concat = []

        x_out = preprocessed_input
        final_x = x_out
        count = 0
        for i in range(int(self.layer_num)):
            select_idx = int(i)
            #select_idx = int(i // 2)
            x_temp_out = self.conv_list[select_idx](x_out, preprocess_edge_attr, data.adj_t, batch)
            #if len(attention_concat) == 0:
            #    attention_concat = attention_value
            #else:
            #    attention_concat = torch.cat((attention_concat, attention_value), 1)

            x_glob = global_mean_pool(x_temp_out, batch)
            x_concat = torch.cat((x_concat, x_glob), 1)
            if self.residual == "Y":
                x_out = x_temp_out + x_out
            else:
                x_out = x_temp_out
            #x_out = x_temp_out + x_out
            final_x = x_out
            count = count + 1

        postprocessed_output = self.postprocess(x_concat, data.batch)

        #print(postprocessed_output)
        risk = self.risk_prediction_layer(postprocessed_output)
        node_noise_value = torch.tensor(0.0).to(risk.device)
        edge_noise_value = torch.tensor(0.0).to(risk.device)

        if self.with_noise == "Y":
            node_predict_x = self.node_denoise_layer(final_x)
            edge_predict_x = self.edge_denoise_layer((final_x[row, :] + final_x[col, :]))
            node_noise_value = F.mse_loss(node_predict_x, original_x)
            node_noise_value = torch.reshape(node_noise_value, (1,1))
            edge_noise_value = F.mse_loss(edge_predict_x, original_edge)
            edge_noise_value = torch.reshape(edge_noise_value, (1,1))

        return risk, node_noise_value, edge_noise_value

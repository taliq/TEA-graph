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
from torch_geometric.nn.inits import kaiming_uniform
# from torch_geometric.transforms import ToSparseTensor
from torch_sparse import SparseTensor
from torch_scatter import scatter
from torch_sparse import set_diag
from torch_scatter import scatter_add

from tqdm import tqdm
from models.model_utils import weight_init
from models.model_utils import decide_loss_type

from models.pre_layer import preprocess
from models.post_layer import postprocess

def BasicLinear(in_f, out_f, dropout_rate):
    return Sequential(
        Linear(in_f, out_f),
        BatchNorm(out_f),
        PReLU(init=0.2),
        Dropout(dropout_rate)
    )

class attention_module(torch.nn.Module):
    
    def __init__(self, dim_1, dim_2):
        
        super(attention_module, self).__init__()
        self.que_transform = nn.Parameter(torch.Tensor(dim_2, dim_1))
        self.gate_transform = nn.Parameter(torch.Tensor(dim_2, dim_1))
        self.weight = nn.Parameter(torch.Tensor(dim_2, 1))
        
    def reset_parameters(self):

        #self.que_transform.apply(weight_init)
        init.kaiming_normal_(self.weight.data)
        init.kaiming_normal_(self.que_transform.data)
        init.kaiming_normal_(self.gate_transform.data)

    def forward(self, x, batch):
        
        que = F.tanh(torch.matmul(self.que_transform, torch.transpose(x, 1, 0)))
        gated = F.sigmoid(torch.matmul(self.gate_transform, torch.transpose(x, 1, 0)))
        key = torch.matmul(torch.transpose(self.weight, 1, 0),torch.mul(que, gated))
        key = torch.transpose(key, 1, 0)

        attention_value = softmax(key, batch)
        out = scatter_add(torch.mul(x, attention_value), batch, dim=0)
        
        return out
        

class MLP_module(torch.nn.Module):

    def __init__(self, input_dim, output_dim, head_num, dropedge_rate, graph_dropout_rate, loss_type, with_edge):

        super(MLP_module, self).__init__()
        self.conv = Linear(input_dim, output_dim * head_num)
        #self.gbn = GraphSizeNorm()
        self.bn = LayerNorm(output_dim * int(head_num))
        self.prelu = decide_loss_type(loss_type)
        self.dropout = Dropout(dropedge_rate)
        self.with_edge = with_edge
        self.heads_num = head_num
        self.graph_dropout_rate = graph_dropout_rate
        self.final_attention = attention_module(output_dim * head_num, 100)

    def reset_parameters(self):

        self.conv.apply(weight_init)
        self.bn.reset_parameters()
        self.final_attention.reset_parameters()

    def forward(self, x, edge_attr, edge_index, batch):

        drop_node_mask = x.new_full((x.size(1),), 1 - self.graph_dropout_rate, dtype=torch.float)
        drop_node_mask = torch.bernoulli(drop_node_mask)
        drop_node_mask = torch.reshape(drop_node_mask, (1, drop_node_mask.shape[0]))
        drop_node_feature = x * drop_node_mask

        drop_edge_mask = edge_attr.new_full((edge_attr.size(1),), 1 - self.graph_dropout_rate, dtype=torch.float)
        drop_edge_mask = torch.bernoulli(drop_edge_mask)
        drop_edge_mask = torch.reshape(drop_edge_mask, (1, drop_edge_mask.shape[0]))
        drop_edge_attr = edge_attr * drop_edge_mask

        if self.with_edge == "Y":
            x_before = self.conv(drop_node_feature)
        else:
            x_before = self.conv(drop_node_feature)

        x_after = self.dropout(self.prelu(self.bn(x_before, batch)))
        x_after_attention = self.final_attention(x_after, batch)

        return x_after, x_after_attention

class MLP_attention(torch.nn.Module):

    def __init__(self, dropout_rate, dropedge_rate, Argument):
        super(MLP_attention, self).__init__()
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
        self.with_noise = Argument.with_noise

        postNum = 0
        self.preprocess = preprocess(Argument)
        #self.conv1 = GAT_module(256, dim, self.heads_num, self.dropedge_rate, self.dropout_rate, Argument.loss_type, with_edge=Argument.with_distance)
        #postNum += int(self.heads_num)

        self.conv_list = nn.ModuleList([MLP_module(dim * self.heads_num, dim, self.heads_num, self.dropedge_rate, self.graph_dropout_rate, Argument.loss_type, with_edge=Argument.with_distance) for _ in range(int(Argument.number_of_layers))])
        postNum += int(self.heads_num) * len(self.conv_list)

        self.postprocess = postprocess(dim * self.heads_num, self.layer_num, dim * self.heads_num, Argument.postlayernum, dropout_rate)

        if self.with_noise == "Y":
            self.node_denoise_layer = nn.Linear(dim * self.heads_num, 1794)
            self.edge_denoise_layer = nn.Linear(dim * self.heads_num, 2)
            #self.risk_prediction_layer = nn.Linear(self.postprocess.postlayernum[-1], 1)

        self.final_attention = attention_module(dim * self.heads_num, 100)
        self.risk_prediction_layer = nn.Linear(self.postprocess.postlayernum[-1], 1)

    def reset_parameters(self):

        self.preprocess.reset_parameters()
        #self.conv1.reset_parameters()
        for i in range(self.Argument.number_of_layers):
            self.conv_list[i].reset_parameters()
        self.postprocess.reset_parameters()
        self.risk_prediction_layer.apply(weight_init)
        if self.with_noise == "Y":
            self.node_denoise_layer.apply(weight_init)
            self.edge_denoise_layer.apply(weight_init)
        self.final_attention.reset_parameters()

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
        for i in range(int(self.layer_num)):
            x_temp_out, x_temp_out_attention = self.conv_list[i](x_out, preprocess_edge_attr, data.adj_t, batch)
            #if len(attention_concat) == 0:
            #    attention_concat = attention_value
            #else:
            #    attention_concat = torch.cat((attention_concat, attention_value), 1)
            #x_glob = self.final_attention(x_temp_out, batch)
            #x_glob = global_mean_pool(x_temp_out, batch)
            x_concat = torch.cat((x_concat, x_temp_out_attention), 1)
            x_out = x_temp_out
            #x_out = x_temp_out + x_out
            final_x = x_out

        #x_concat = self.final_attention(x_out, batch)
        postprocessed_output = self.postprocess(x_concat, batch)
        #attentioned_product = self.final_attention(postprocessed_output, batch)

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

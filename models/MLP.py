# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, Sequential, PReLU, Dropout, LayerNorm
from torch_geometric.nn import global_mean_pool

from models.model_utils import weight_init
from models.model_utils import decide_loss_type

from models.pre_layer import preprocess
from models.post_layer import postprocess

def BasicLinear(in_f, out_f, dropout_rate):
    return Sequential(
        Linear(in_f, out_f),
        LayerNorm(out_f),
        PReLU(init=0.2),
        Dropout(dropout_rate)
    )

class MLP_module(torch.nn.Module):

    def __init__(self, input_dim, output_dim, head_num, dropedge_rate, graph_dropout_rate, loss_type, with_edge):

        super(MLP_module, self).__init__()
        self.conv = Linear(input_dim, output_dim * head_num)
        self.bn = LayerNorm(output_dim * int(head_num))
        self.prelu = decide_loss_type(loss_type)
        self.dropout = Dropout(dropedge_rate)
        self.with_edge = with_edge
        self.heads_num = head_num
        self.graph_dropout_rate = graph_dropout_rate
        self.norm_type = "layer"

    def reset_parameters(self):

        self.conv.apply(weight_init)
        self.bn.reset_parameters()

    def forward(self, x, edge_attr, edge_index, batch):

        if self.training:
            drop_node_mask = x.new_full((x.size(1),), 1 - self.graph_dropout_rate, dtype=torch.float)
            drop_node_mask = torch.bernoulli(drop_node_mask)
            drop_node_mask = torch.reshape(drop_node_mask, (1, drop_node_mask.shape[0]))
            drop_node_feature = x * drop_node_mask

            drop_edge_mask = edge_attr.new_full((edge_attr.size(1),), 1 - self.graph_dropout_rate, dtype=torch.float)
            drop_edge_mask = torch.bernoulli(drop_edge_mask)
            drop_edge_mask = torch.reshape(drop_edge_mask, (1, drop_edge_mask.shape[0]))
            drop_edge_attr = edge_attr * drop_edge_mask
        else:
            drop_node_feature = x
            drop_edge_attr = edge_attr

        if self.with_edge == "Y":
            x_before = self.conv(drop_node_feature)
        else:
            x_before = self.conv(drop_node_feature)

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

        x_after = self.dropout(self.prelu(out_x_temp))

        return x_after

class MLP(torch.nn.Module):

    def __init__(self, dropout_rate, dropedge_rate, Argument):
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        dim = Argument.initial_dim
        self.dropout_rate = dropout_rate
        self.dropedge_rate = dropedge_rate
        self.Argument = Argument
        self.heads_num = Argument.attention_head_num
        self.include_edge_feature = Argument.with_distance
        self.layer_num = Argument.number_of_layers
        self.graph_dropout_rate = Argument.graph_dropout_rate
        self.residual = Argument.residual_connection

        postNum = 0
        self.preprocess = preprocess(Argument)

        self.conv_list = nn.ModuleList([MLP_module(dim * self.heads_num, dim, self.heads_num, self.dropedge_rate, self.graph_dropout_rate, Argument.loss_type, with_edge=Argument.with_distance) for _ in range(int(Argument.number_of_layers))])
        postNum += int(self.heads_num) * len(self.conv_list)

        self.postprocess = postprocess(dim * self.heads_num, self.layer_num, dim * self.heads_num, (Argument.MLP_layernum-1), dropout_rate)
        self.risk_prediction_layer = nn.Linear(self.postprocess.postlayernum[-1], 1)

    def reset_parameters(self):

        self.preprocess.reset_parameters()
        for i in range(self.Argument.number_of_layers):
            self.conv_list[i].reset_parameters()
        self.postprocess.reset_parameters()
        self.risk_prediction_layer.apply(weight_init)

    def forward(self, data):

        original_x = data.x
        original_edge = data.edge_attr
        row, col, _ = data.adj_t.coo()

        preprocessed_input, preprocess_edge_attr = self.preprocess(data)
        batch = data.batch

        x0_glob = global_mean_pool(preprocessed_input, batch)
        x_concat = x0_glob

        x_out = preprocessed_input
        final_x = x_out
        for i in range(int(self.layer_num)):
            x_temp_out = self.conv_list[i](x_out, preprocess_edge_attr, data.adj_t, batch)
            x_glob = global_mean_pool(x_temp_out, batch)
            x_concat = torch.cat((x_concat, x_glob), 1)

            if self.residual == "Y":
                x_out = x_temp_out + x_out
            else:
                x_out = x_temp_out

            final_x = x_out

        postprocessed_output = self.postprocess(x_concat, batch)
        risk = self.risk_prediction_layer(postprocessed_output)

        return risk

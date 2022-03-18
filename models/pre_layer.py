# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from torch.nn import Linear, Dropout, LeakyReLU, LayerNorm
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GraphSizeNorm
from torch_sparse import set_diag

from models.model_utils import weight_init
from model_utils import decide_loss_type

class BasicLinear_module(torch.nn.Module):

    def __init__(self, in_f, out_f, dropout_rate, norm_type):

        super(BasicLinear_module, self).__init__()
        self.ff = Linear(in_f, out_f)
        if norm_type == "layer":
            self.norm = LayerNorm(out_f)
            self.gbn = None
        else:
            self.norm = BatchNorm(out_f)
            self.gbn = GraphSizeNorm()

        self.act = LeakyReLU(negative_slope=0.2)
        self.drop = Dropout(dropout_rate)
        self.norm_type = norm_type

    def reset_parameters(self):

        self.ff.apply(weight_init)
        self.norm.reset_parameters()

    def forward(self, input_x, batch):

        out_x = self.ff(input_x)
        out_x_temp = 0
        if self.norm_type == "layer":
            for c, item in enumerate(torch.unique(batch)):
                temp = self.norm(out_x[batch == item])
                if c == 0:
                    out_x_temp = temp
                else:
                    out_x_temp = torch.cat((out_x_temp, temp), 0)

        else:
            temp = self.gbn(self.norm(out_x), batch)
            out_x_temp = temp

        out_x = self.act(out_x_temp)
        out_x = self.drop(out_x)

        return out_x

class preprocess(torch.nn.Module):

    def __init__(self, argument):
        super(preprocess, self).__init__()

        prelayerpreset = [800, 400, argument.attention_head_num * argument.initial_dim]
        self.prelayernum = []
        self.prelayernum.append(1792)
        for i in range(0, argument.prelayernum -1):
            self.prelayernum.append(prelayerpreset[i])

        dropout_rate = argument.dropout_rate
        norm_type = argument.norm_type

        self.prelayer_blocks = nn.ModuleList([BasicLinear_module(in_f, out_f, dropout_rate, norm_type)
                           for in_f, out_f in zip(self.prelayernum, self.prelayernum[1:])])

        self.prelayer_last = Linear(self.prelayernum[-1], argument.attention_head_num * argument.initial_dim)
        self.edge_position_embedding = nn.Embedding(11, 100)
        self.edge_angle_embedding = nn.Embedding(11, 100)
        self.with_edge = argument.with_distance
        self.simple_distance = argument.simple_distance

    def reset_parameters(self):

        for i in range(len(self.prelayer_blocks)):
            self.prelayer_blocks[i].reset_parameters()
        self.prelayer_last.apply(weight_init)

    def forward(self, data, edge_mask=None):

        if (edge_mask == None):
            input_x = data.x
        else:
            input_x = torch.mul(data.x, torch.reshape(edge_mask, (edge_mask.shape[0], 1)))

        edge_index = data.adj_t
        edge_feature = data.edge_attr
        row, col, _ = edge_index.coo()

        Non_self_feature = edge_feature[~(row == col), :]

        drop_adj_t = set_diag(edge_index)
        drop_diag_row, drop_diag_col, _ = drop_adj_t.coo()
        drop_edge_attr_diag = np.zeros((drop_diag_row.shape[0], edge_feature.shape[1]))
        drop_edge_attr_diag[
            ~(drop_diag_row == drop_diag_col).cpu().detach().numpy()] = Non_self_feature.cpu().detach().numpy()
        drop_edge_attr = torch.tensor(drop_edge_attr_diag)
        drop_edge_attr = drop_edge_attr.type(torch.FloatTensor)

        if self.simple_distance == "Y":
            pass
        else:
            drop_edge_attr_distance = drop_edge_attr[:, 0]
            drop_edge_attr_distance = drop_edge_attr_distance // 0.1
            drop_edge_attr_distance = drop_edge_attr_distance.type(torch.LongTensor)
            drop_edge_attr_distance = drop_edge_attr_distance.to(input_x.device)
            drop_edge_attr_angle = drop_edge_attr[:, 1]
            drop_edge_attr_angle = drop_edge_attr_angle // 0.1
            drop_edge_attr_angle = drop_edge_attr_angle.type(torch.LongTensor)
            drop_edge_attr_angle = drop_edge_attr_angle.to(input_x.device)

            drop_edge_attr_distance = self.edge_position_embedding(drop_edge_attr_distance)
            drop_edge_attr_angle = self.edge_angle_embedding(drop_edge_attr_angle)
            drop_edge_attr = torch.cat((drop_edge_attr_distance, drop_edge_attr_angle), 1)

        preprocessed_data = input_x
        for i in range(len(self.prelayer_blocks)):
            preprocessed_data = self.prelayer_blocks[i](preprocessed_data, data.batch)

        preprocessed_data = self.prelayer_last(preprocessed_data)

        return preprocessed_data, drop_edge_attr

import torch
import torch.nn as nn

from torch.nn import Linear, Dropout, LeakyReLU
from models.model_utils import weight_init

class BasicLinear_module(torch.nn.Module):

    def __init__(self, in_f, out_f, dropout_rate):

        super(BasicLinear_module, self).__init__()
        self.Linear = Linear(in_f, out_f)
        self.LeakyReLU = LeakyReLU(negative_slope=0.2)
        self.Dropout = Dropout(dropout_rate)

    def reset_parameters(self):

        self.Linear.apply(weight_init)

    def forward(self, input_x, batch):

        out_x = self.Linear(input_x)
        out_x = self.LeakyReLU(out_x)
        out_x = self.Dropout(out_x)

        return out_x

class postprocess(torch.nn.Module):

    def __init__(self, graph_feature_dim, graph_layer_num, last_input_dim, post_layer_num, dropout_rate):
        super(postprocess, self).__init__()

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

    def reset_parameters(self):

        for i in range(len(self.postlayer_blocks)):
            self.postlayer_blocks[i].reset_parameters()

    def forward(self, input_x, batch):

        postprocessed_data = input_x
        for i in range(len(self.postlayer_blocks)):
            postprocessed_data = self.postlayer_blocks[i](postprocessed_data, batch)

        return postprocessed_data

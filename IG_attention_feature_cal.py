#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:58:47 2021

@author: taliq
"""
import os
import torch
import random
import torch.nn.functional as F
import torch_geometric.transforms as T
import sklearn.preprocessing as preprocessing

from torch.nn import Linear

from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import DataListLoader

from torch_geometric.utils import dropout_adj
from torch_geometric.utils import to_networkx

from captum.attr import Saliency, IntegratedGradients, GradientShap

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from collections import defaultdict
from tqdm import tqdm

from sklearn.preprocessing import maxabs_scale
from utils import TrainValid_path
from utils import train_test_split
from model_selection import model_selection

import openslide as osd

def model_forward(edge_mask, model, data):
    #edge_mask = edge_mask.to(data.x.device)
    out = model(data, edge_mask)

    return out

def explain(method, data, device, model):
    target = 0
    input_mask = torch.ones(data.x.shape[0], 1).requires_grad_(True).to(device)
    baseline = torch.zeros(data.x.shape[0], 1).requires_grad_(True).to(device)
    ig = IntegratedGradients(model_forward)
    mask = ig.attribute(input_mask, target=target, baselines=baseline,
                        additional_forward_args=(model, data), n_steps=50,
                        internal_batch_size=data.x.shape[0])

    edge_mask = mask.cpu().detach().numpy()

    return edge_mask

def Calculate_feature_attention(d, model):
    d = d.to(torch.device(0))
    input_mask = torch.ones(d.x.shape[0], 1).requires_grad_(True).to(torch.device(0))
    out, updated_feature, attention_list = model(d, input_mask, Interpretation_mode=True)

    return out, updated_feature, attention_list

class CoxGraphDataset(Dataset):
    def __init__(self, filelist, survlist, stagelist, censorlist, metadata, mode, model, transform=None, pre_transform=None):
        super(CoxGraphDataset, self).__init__()
        self.filelist = filelist
        self.survlist = survlist
        self.stagelist = stagelist
        self.censorlist = censorlist
        #random.shuffle(self.filelist)
        self.metadata = metadata
        self.mode = mode
        self.model = model

    def processed_file_names(self):
        return self.filelist

    def len(self):
        return len(self.filelist)

    def get(self, idx):
        data_origin = torch.load(self.filelist[idx])
        transfer = T.ToSparseTensor()
        item = self.filelist[idx].split('/')[-1].split('.pt')[0].split('_')[0]
        mets_class = 0

        survival = self.survlist[idx]
        phase = self.censorlist[idx]
        stage = self.stagelist[idx]

        data_re = Data(x=data_origin.x[:,:1792], edge_index=data_origin.edge_index)
        data = transfer(data_re)
        data.survival = torch.tensor(survival)
        data.phase = torch.tensor(phase)
        data.mets_class = torch.tensor(mets_class)
        data.stage = torch.tensor(stage)
        data.item = item
        data.edge_attr = data_origin.edge_attr
        data.pos = data_origin.pos
        data.absolute_path = self.filelist[idx].split('/')[-1].split('_graph_torch_4.3_artifact_sophis_final.pt')[0]

        return data

def Vis_main(Argument):

    coredir = os.path.join('/'.join(Argument.load_state_dict.split('/')[:-1]), 'IG_analysis')

    if not os.path.isdir(coredir):
        os.mkdir(coredir)

    batch_num = Argument.batch_size
    device = torch.device(int(Argument.gpu))
    Metadata = pd.read_csv('./Sample_data_for_demo/Metadata/KIRC_clinical.tsv', sep='\t')

    TrainRoot = TrainValid_path(Argument.DatasetType)
    Trainlist = os.listdir(TrainRoot)

    Trainlist = [item for c, item in enumerate(Trainlist) if
             (("_0_graph_torch_4.3_artifact_sophis_final.pt" in item))]
    Fi = Argument.FF_number
    Trainlist = os.listdir(TrainRoot)
    Trainlist = Trainlist[:50]

    Test_set = train_test_split(Trainlist, Metadata,Argument.DatasetType, TrainRoot, Fi, Analyze_flag=True)

    TestDataset = CoxGraphDataset(filelist=Test_set[0], survlist=Test_set[1],
                                  stagelist=Test_set[3], censorlist=Test_set[2],
                                  metadata=Metadata, mode=Argument.DatasetType,
                                  model=Argument.model)
    test_loader = DataLoader(TestDataset, batch_size=batch_num, shuffle=True, num_workers=8, pin_memory=False,
                                 drop_last=False)

    model = model_selection(Argument)

    Temp_state_dict = torch.load(Argument.load_state_dict, map_location="cpu")
    Temp_state_dict_list = list(Temp_state_dict.keys())
    for item in Temp_state_dict_list:
        Temp_state_dict[item.split('module.')[1]] = Temp_state_dict[item]
        del Temp_state_dict[item]

    model.load_state_dict(Temp_state_dict)

    model = model.to(torch.device(0))
    model.eval()

    with tqdm(total=len(test_loader)) as pbar:
        for c,d in enumerate(test_loader,1):

            d = d.to(torch.device(0))
            method = "ig"
            batch_count = np.unique(d.batch.cpu().detach().numpy(), return_counts=True)[1].tolist()

            id_path = d.absolute_path
            go_flag = True
            #for id_item_temp in id_path:
            #    patient_target_dir = os.path.join(coredir, id_item_temp)
            #    if not os.path.isdir(patient_target_dir):
            #        go_flag = True

            if go_flag:
                edge_mask = explain(method, d, device, model)
                out, updated_feature, attention_list = Calculate_feature_attention(d, model)

                row, col, _ = d.adj_t.coo()
                adj_batch = d.batch[row]
                adj_batch_count = np.unique(adj_batch.cpu().detach().numpy(), return_counts=True)[1].tolist()

                start_idx_node = 0
                end_idx_node = 0

                start_idx_adj = 0
                end_idx_adj = 0

                node_num_cumul = 0

                for inside_count, (batch_item_num, adj_batch_item_num, id_item) in enumerate(
                        zip(batch_count, adj_batch_count, id_path)):

                    patient_target_dir = os.path.join(coredir, id_item)
                    if not os.path.isdir(patient_target_dir):
                        os.mkdir(patient_target_dir)

                    end_idx_node = start_idx_node + batch_item_num
                    np.save(os.path.join(patient_target_dir, 'Node_Ig_sophis'), edge_mask[start_idx_node:end_idx_node, :])
                    np.save(os.path.join(patient_target_dir, 'whole_feature'),
                            updated_feature[start_idx_node:end_idx_node, :].cpu().detach().numpy())
                    start_idx_node = end_idx_node

                    end_idx_adj = start_idx_adj + adj_batch_item_num

                    np.save(os.path.join(patient_target_dir, 'attention_value'),
                            attention_list[:, start_idx_adj:end_idx_adj, :].cpu().detach().numpy())

                    inside_row = row[start_idx_adj:end_idx_adj].cpu().detach().numpy()
                    inside_col = col[start_idx_adj:end_idx_adj].cpu().detach().numpy()

                    inside_row = np.subtract(inside_row, node_num_cumul)
                    inside_col = np.subtract(inside_col, node_num_cumul)

                    start_idx_adj = end_idx_adj

                    node_num_cumul = node_num_cumul + batch_item_num

            pbar.update()


    return 0
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:58:47 2021

@author: taliq
"""
import os
import torch
import torch_geometric.transforms as T

from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Dataset

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import TrainValid_path
from utils import train_test_split
from model_selection import model_selection

from matplotlib import cm
import pandas as pd
import networkx as nx

import openslide as osd
import cv2 as cv
from PIL import Image
from PIL import ImageFilter

def model_forward(edge_mask, model, data):

    out = model(data, edge_mask)

    return out

def Whole_supernode_vis(WSI, Position_pd, Location_pd, Select_ID, core_dir, edge_mask_norm, row, col):
    WSI_level = 2
    Downsample_ratio = int(WSI.level_downsamples[WSI_level])
    WSI_width, WSI_height = WSI.level_dimensions[WSI_level]
    WSI_image = WSI.read_region((0, 0), WSI_level, (WSI_width, WSI_height))
    Image_mask = np.zeros((WSI_height, WSI_width))
    Image_IG_mask = np.zeros((WSI_height, WSI_width))
    Image_IG_mask_count = np.zeros((WSI_height, WSI_width))
    WSI_patch_dimension = int(256 / Downsample_ratio)

    for col_idx in range(Position_pd.shape[0]):
        Superpatch_node_row = Position_pd.iloc[col_idx].dropna()
        Superpatch = int(Superpatch_node_row[2].tolist())
        Superpatch_pos = Location_pd.iloc[Superpatch]
        Envpatch = Superpatch_node_row[3:].to_numpy().astype(int)
        Envpatch_pos = Location_pd.iloc[Envpatch]

        Image_mask[Superpatch_pos['Y'] * WSI_patch_dimension:(Superpatch_pos['Y'] + 1) * WSI_patch_dimension:,
        Superpatch_pos['X'] * WSI_patch_dimension:(Superpatch_pos['X'] + 1) * WSI_patch_dimension] = 2

        Image_IG_mask[Superpatch_pos['Y'] * WSI_patch_dimension:(Superpatch_pos['Y'] + 1) * WSI_patch_dimension:,
        Superpatch_pos['X'] * WSI_patch_dimension:(Superpatch_pos['X'] + 1) * WSI_patch_dimension] = (
            edge_mask_norm[col_idx])

        Image_IG_mask_count[Superpatch_pos['Y'] * WSI_patch_dimension:(Superpatch_pos['Y'] + 1) * WSI_patch_dimension:,
        Superpatch_pos['X'] * WSI_patch_dimension:(Superpatch_pos['X'] + 1) * WSI_patch_dimension] = 1

        for Env_y, Env_x in zip(Envpatch_pos['Y'].tolist(), Envpatch_pos['X'].tolist()):
            Image_mask[Env_y * WSI_patch_dimension:(Env_y + 1) * WSI_patch_dimension,
            Env_x * WSI_patch_dimension:(Env_x + 1) * WSI_patch_dimension] = 1
            Image_IG_mask[Env_y * WSI_patch_dimension:(Env_y + 1) * WSI_patch_dimension,
            Env_x * WSI_patch_dimension:(Env_x + 1) * WSI_patch_dimension] = Image_IG_mask[Env_y * WSI_patch_dimension:(
                                                                                                                               Env_y + 1) * WSI_patch_dimension,
                                                                             Env_x * WSI_patch_dimension:(
                                                                                                                 Env_x + 1) * WSI_patch_dimension] + (
                                                                                 edge_mask_norm[col_idx])
            Image_IG_mask_count[Env_y * WSI_patch_dimension:(Env_y + 1) * WSI_patch_dimension,
            Env_x * WSI_patch_dimension:(Env_x + 1) * WSI_patch_dimension] = Image_IG_mask_count[
                                                                             Env_y * WSI_patch_dimension:(
                                                                                                                 Env_y + 1) * WSI_patch_dimension,
                                                                             Env_x * WSI_patch_dimension:(
                                                                                                                 Env_x + 1) * WSI_patch_dimension] + 1

    Image_IG_mask = Image_IG_mask / Image_IG_mask_count
    Image_IG_mask = np.uint8(255 * cm.coolwarm(Image_IG_mask))
    Image_IG_mask = Image.fromarray(Image_IG_mask)
    Image_IG_mask = Image_IG_mask.filter(ImageFilter.GaussianBlur)
    Image_IG_mask.save(os.path.join(core_dir, Select_ID + '_WSI_Image_mask_IG_new.gif'))

    Image_mask = Image_mask - Image_mask.min()
    Image_mask = Image_mask / Image_mask.max()
    Image_mask = cv.applyColorMap(np.uint8(255 * Image_mask), cv.COLORMAP_JET)
    Colover_converted_mask = cv.cvtColor(Image_mask, cv.COLOR_BGR2RGB)
    Image_mask = Image.fromarray(Colover_converted_mask)

    Mask_fig = Image_mask
    Mask_fig = Mask_fig.convert('RGBA')

    Mask_fig_IG = Image_IG_mask.convert('RGBA')

    WSI_node_idx = Position_pd['Unnamed: 0.1'].tolist()
    WSI_node_idx = [int(item) for item in WSI_node_idx]
    WSI_Location = Location_pd.iloc[WSI_node_idx]
    X_pos = WSI_Location['X'].tolist()
    Y_pos = WSI_Location['Y'].tolist()
    X_Y_pos = [(X, Y) for X, Y in zip(X_pos, Y_pos)]
    pos_dict = zip(Position_pd['Unnamed: 0'].tolist(), X_Y_pos)
    pos_dict = dict(pos_dict)

    WSI_edge_index = [(row_item, col_item) for row_item, col_item in zip(row, col)]
    WSI_graph = nx.Graph()
    WSI_graph.add_nodes_from(list(range(len(WSI_node_idx))))
    WSI_graph.add_edges_from(WSI_edge_index)
    WSI_graph.remove_edges_from(nx.selfloop_edges(WSI_graph))

    my_dpi = 96
    plt.figure(figsize=(WSI_image.size[0] / my_dpi, WSI_image.size[1] / my_dpi), dpi=96)
    plt.axis('off')
    nx.draw_networkx(WSI_graph, pos=pos_dict, node_size=20, node_color='black', width=0.2, alpha=0.8, arrows=False,
                     with_labels=False)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(core_dir, Select_ID + '_WSI_graph_wo_IG.jpeg'), transparent=True)
    plt.clf()
    plt.cla()
    plt.close()

    my_dpi = 96
    plt.figure(figsize=(WSI_image.size[0] / my_dpi, WSI_image.size[1] / my_dpi), dpi=96)
    plt.axis('off')
    nx.draw_networkx(WSI_graph, pos=pos_dict, node_size=30, width=0.2, node_color=(edge_mask_norm), vmin=0.0, vmax=1.0,
                     cmap=plt.cm.coolwarm, arrows=False, with_labels=False)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(core_dir, Select_ID + '_WSI_graph_w_IG.jpeg'), transparent=True)
    plt.clf()
    plt.cla()
    plt.close()

    return Mask_fig_IG

def IG_TME_vis(IG_nodes, row, col, IG_edge_index, Position_pd, Location_pd, core_dir, Select_ID, edge_mask_norm, edge_mask):

    Mid_IG_WSI_graph = nx.Graph()
    Mid_IG_WSI_graph.add_nodes_from(IG_nodes)
    Mid_IG_WSI_graph.add_edges_from(IG_edge_index)
    np_edge_index = np.concatenate((row, col), 1)
    match_edge_idx = []

    Mid_IG_WSI_graph.remove_edges_from(nx.selfloop_edges(Mid_IG_WSI_graph))
    Mid_IG_subgraph = [Mid_IG_WSI_graph.subgraph(c).copy() for c in nx.connected_components(Mid_IG_WSI_graph) if
                       (len(c) > 10 & len(c) < 200)]

    if len(Mid_IG_subgraph) > 0:
        Mid_IG_subgraph_nodes = []
        Subgraph_pos_max_x = []
        Subgraph_pos_max_y = []
        Subgraph_pos_min_x = []
        Subgraph_pos_min_y = []

        Subgraph_IG_list = []
        Subgraph_IG_norm_list = []

        Supernode_node_idx = Position_pd['Unnamed: 0.1']
        for c_item in Mid_IG_subgraph:

            subgraph = c_item
            subgraph_nodes = subgraph.nodes()

            subgraph_IG_value = np.mean(edge_mask[subgraph_nodes])
            Subgraph_IG_list.append(subgraph_IG_value.item())
            normalized_subgraph_IG_value = np.mean(edge_mask_norm[subgraph_nodes])
            Subgraph_IG_norm_list.append(normalized_subgraph_IG_value.item())

            subgraph_whole_nodes = []
            for node_idx in subgraph_nodes:
                subgraph_whole_nodes.extend(Position_pd.iloc[node_idx].dropna().tolist()[2:])
            subgraph_whole_nodes = list(set(subgraph_whole_nodes))
            subgraph_whole_nodes_location = Location_pd.iloc[subgraph_whole_nodes]

            subgraph_whole_max_X = subgraph_whole_nodes_location['X'].to_numpy().max()
            subgraph_whole_min_X = subgraph_whole_nodes_location['X'].to_numpy().min()
            subgraph_whole_max_Y = subgraph_whole_nodes_location['Y'].to_numpy().max()
            subgraph_whole_min_Y = subgraph_whole_nodes_location['Y'].to_numpy().min()

            Subgraph_pos_max_x.append(subgraph_whole_max_X.item())
            Subgraph_pos_max_y.append(subgraph_whole_max_Y.item())
            Subgraph_pos_min_x.append(subgraph_whole_min_X.item())
            Subgraph_pos_min_y.append(subgraph_whole_min_Y.item())

            if len(Mid_IG_subgraph_nodes) == 0:
                Mid_IG_subgraph_nodes = pd.DataFrame(subgraph_nodes).transpose()
            else:
                Mid_IG_subgraph_nodes = pd.concat(
                    (Mid_IG_subgraph_nodes, pd.DataFrame(subgraph_nodes).transpose()), 0, ignore_index=True)

        if len(Mid_IG_subgraph_nodes) != 0:
            Mid_IG_subgraph_nodes['IG_value'] = Subgraph_IG_list
            Mid_IG_subgraph_nodes['Norm_IG_value'] = Subgraph_IG_norm_list
            Mid_IG_subgraph_nodes['Max_X'] = Subgraph_pos_max_x
            Mid_IG_subgraph_nodes['Min_X'] = Subgraph_pos_min_x
            Mid_IG_subgraph_nodes['Max_Y'] = Subgraph_pos_max_y
            Mid_IG_subgraph_nodes['Min_Y'] = Subgraph_pos_min_y
            Mid_IG_subgraph_nodes['Width'] = (np.array(Subgraph_pos_max_x) - np.array(Subgraph_pos_min_x)).tolist()
            Mid_IG_subgraph_nodes['Height'] = (np.array(Subgraph_pos_max_y) - np.array(Subgraph_pos_min_y)).tolist()

            Mid_IG_subgraph_nodes.to_csv(os.path.join('/'.join(core_dir.split('/')[:-1]),
                                                      Select_ID + '_' + core_dir.split('/')[
                                                          -1] + '_TME_subgraph_small_cap_subgraph_exist_again.csv'))
        else:
            print(Select_ID)

def IG_subgraph_vis(row, col, Position_pd, Location_pd, Select_ID, core_dir, data, edge_mask_norm, edge_mask,
                    top_threshold, low_threshold, mid_threshold_top, mid_threshold_low):
    row = np.reshape(row, (row.shape[0], 1))
    col = np.reshape(col, (col.shape[0], 1))
    Inside_edge_index = np.concatenate((row, col), 1)

    Top_IG_nodes = np.where(edge_mask > top_threshold)[0]
    Top_IG_dir = os.path.join(core_dir, "IG_again")
    Top_IG_dir = os.path.join(Top_IG_dir, "Top_IG")
    if not os.path.isdir(Top_IG_dir):
        os.makedirs(Top_IG_dir)

    if len(Top_IG_nodes) > 0:
        Top_IG_edge_index = np.isin(Inside_edge_index, Top_IG_nodes)
        Top_IG_edge_index = Top_IG_edge_index[:, 0] * Top_IG_edge_index[:, 1]
        Top_IG_edge_index = Inside_edge_index[Top_IG_edge_index]
        IG_TME_vis(Top_IG_nodes, row, col, Top_IG_edge_index, Position_pd, Location_pd,
                   Top_IG_dir, Select_ID, edge_mask_norm, edge_mask)

    Mid_IG_nodes = list(set(np.where(edge_mask < mid_threshold_top)[0]) &
                        set(np.where(edge_mask > mid_threshold_low)[0]))
    Mid_IG_dir = os.path.join(core_dir, "IG_again")
    Mid_IG_dir = os.path.join(Mid_IG_dir, "Mid_IG")
    if not os.path.isdir(Mid_IG_dir):
        os.makedirs(Mid_IG_dir)

    if len(Mid_IG_nodes) > 0:
        Mid_IG_edge_index = np.isin(Inside_edge_index, Mid_IG_nodes)
        Mid_IG_edge_index = Mid_IG_edge_index[:, 0] * Mid_IG_edge_index[:, 1]
        Mid_IG_edge_index = Inside_edge_index[Mid_IG_edge_index]
        IG_TME_vis(Mid_IG_nodes, row, col, Mid_IG_edge_index, Position_pd, Location_pd,
                   Mid_IG_dir, Select_ID, edge_mask_norm, edge_mask)

    Low_IG_nodes = np.where(edge_mask < low_threshold)[0]
    Low_IG_dir = os.path.join(core_dir, "IG_again")
    Low_IG_dir = os.path.join(Low_IG_dir, "Low_IG")
    if not os.path.isdir(Low_IG_dir):
        os.makedirs(Low_IG_dir)

    if len(Low_IG_nodes) > 0:
        Low_IG_edge_index = np.isin(Inside_edge_index, Low_IG_nodes)
        Low_IG_edge_index = Low_IG_edge_index[:, 0] * Low_IG_edge_index[:, 1]
        Low_IG_edge_index = Inside_edge_index[Low_IG_edge_index]
        IG_TME_vis(Low_IG_nodes, row, col, Low_IG_edge_index, Position_pd, Location_pd,
                   Low_IG_dir, Select_ID, edge_mask_norm, edge_mask)


def explain(edge_mask, max_IG, min_IG):
    add_count = 0
    edge_mask_norm = edge_mask.copy()
    upper_threshold_node = np.where(edge_mask > max_IG)[0]
    lower_threshold_node = np.where(edge_mask < min_IG)[0]
    edge_mask_norm[upper_threshold_node] = max_IG
    edge_mask_norm[lower_threshold_node] = min_IG
    edge_mark_remove = edge_mask_norm.copy()


    edge_mask_norm = (edge_mask_norm - min_IG) / (max_IG - min_IG)

    local_edge_mask_norm = edge_mark_remove.copy()
    edge_mask_positive = np.where(edge_mark_remove >= 0)[0]
    edge_mask_negative = np.where(edge_mark_remove < 0)[0]
    posnorm = edge_mark_remove[edge_mask_positive] / max_IG
    local_edge_mask_norm[edge_mask_positive] = posnorm
    negnorm = -1 * edge_mark_remove[edge_mask_negative] / min_IG
    local_edge_mask_norm[edge_mask_negative] = negnorm

    local_edge_mask_norm = local_edge_mask_norm / 2.0 + 0.5

    return edge_mask_norm, local_edge_mask_norm

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

def whole_IG_normalize(rootdir):
    Patients = os.listdir(rootdir)
    Whole_IG_value = []

    for item in Patients:
        patient_root = os.path.join(rootdir, item)
        patient_features = os.listdir(patient_root)
        patient_features = [item for item in patient_features if 'Node_Ig_sophis.npy' in item]
        if len(patient_features) > 0:
            if len(Whole_IG_value) == 0:
                Whole_IG_value = np.load(os.path.join(patient_root,
                                                      patient_features[0]))
            else:
                Whole_IG_value = np.concatenate((Whole_IG_value,
                                                 np.load(os.path.join(patient_root,
                                                                      patient_features[0]))))

    pos_term = np.where(Whole_IG_value >= 0)[0]
    neg_term = np.where(Whole_IG_value < 0)[0]

    upper_outlier = np.quantile(Whole_IG_value[pos_term], 0.98)
    lower_outlier = np.quantile(Whole_IG_value[neg_term], 0.02)

    outlier_removed_IG_value = Whole_IG_value[
        list(set(np.where(Whole_IG_value < upper_outlier)[0]) & set(np.where(Whole_IG_value > lower_outlier)[0]))]

    max_IG = outlier_removed_IG_value.max()
    min_IG = outlier_removed_IG_value.min()

    pos_term = np.where(outlier_removed_IG_value >= 0)[0]
    neg_term = np.where(outlier_removed_IG_value < 0)[0]

    top_threshold = np.quantile(outlier_removed_IG_value[pos_term], 0.88)
    low_threshold = np.quantile(outlier_removed_IG_value[neg_term], 0.12)
    mid_threshold_top = np.quantile(outlier_removed_IG_value[pos_term], 0.05)
    mid_threshold_low = np.quantile(outlier_removed_IG_value[neg_term], 0.05)

    return max_IG, min_IG, top_threshold, low_threshold, mid_threshold_top, mid_threshold_low

def whole_attention_normalize(rootdir):
    Patients = os.listdir(rootdir)
    Whole_IG_value = []

    with tqdm(total=len(Patients)) as pbar:
        for item in Patients:
            patient_root = os.path.join(rootdir, item)
            patient_features = os.listdir(patient_root)
            patient_features = [item for item in patient_features if 'attention_value.npy' in item]
            if len(patient_features) > 0:
                if len(Whole_IG_value) == 0:
                    Whole_IG_value = np.mean(np.mean(np.load(os.path.join(patient_root,
                                                                             patient_features[0])), axis=0), axis=1)
                else:
                    Whole_IG_value = np.concatenate((Whole_IG_value,
                                                     np.mean(np.mean(np.load(os.path.join(patient_root,
                                                                                          patient_features[0])),
                                                                     axis=0), axis=1)))
            pbar.update()

    upper_outlier = np.quantile(Whole_IG_value, 0.98)
    lower_outlier = np.quantile(Whole_IG_value, 0.02)

    outlier_removed_IG_value = Whole_IG_value[
        list(set(np.where(Whole_IG_value < upper_outlier)[0]) & set(np.where(Whole_IG_value > lower_outlier)[0]))]

    max_IG = outlier_removed_IG_value.max()
    min_IG = outlier_removed_IG_value.min()

    top_threshold = np.quantile(outlier_removed_IG_value, 0.88)
    low_threshold = np.quantile(outlier_removed_IG_value, 0.12)
    mid_threshold_top = np.quantile(outlier_removed_IG_value, 0.55)
    mid_threshold_low = np.quantile(outlier_removed_IG_value, 0.45)

    return max_IG, min_IG, top_threshold, low_threshold, mid_threshold_top, mid_threshold_low

def Calculate_IG(d, d_count, row, col, edge_mask, edge_mask_norm, patient_target_dir,
                 top_threshold, low_threshold, mid_threshold_top, mid_threshold_low, dataset_root):

    p_id = d.absolute_path[d_count].split('_')[0]
    id_path = d.absolute_path[d_count]

    WSI_dir = os.path.join('./Sample_data_for_demo/Raw_WSI/TCGAFilter/', '-'.join(p_id.split('-')[0:3]))
    WSI_dir_list = os.listdir(WSI_dir)
    WSI_dir_list = [item for item in WSI_dir_list if p_id in item]
    WSI = osd.open_slide(os.path.join(WSI_dir, WSI_dir_list[0]))
    Position_pd = pd.read_csv(dataset_root + id_path + '_4.3_artifact_sophis_final.csv')
    Location_pd = pd.read_csv(
        dataset_root + '_'.join(id_path.split('_')[0:1]) + '_node_location_list.csv')

    Mask_fig_IG = Whole_supernode_vis(WSI, Position_pd, Location_pd, id_path, patient_target_dir, edge_mask_norm, row,
                                      col)

    IG_subgraph_vis(row, col, Position_pd, Location_pd, id_path, patient_target_dir, d, edge_mask_norm, edge_mask,
                    top_threshold, low_threshold, mid_threshold_top, mid_threshold_low)

    return 0



def Subgraph_analysis_vis(Argument):

    coredir = os.path.join('/'.join(Argument.load_state_dict.split('/')[:-1]), 'IG_analysis')
    max_IG, min_IG, top_threshold, low_threshold, mid_threshold_top, mid_threshold_low = whole_IG_normalize(coredir)
    max_att, min_att, top_threshold_att, low_threshold_att, mid_threshold_top_att, mid_threshold_low_att = whole_attention_normalize(coredir)

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

    calculate_loader = DataLoader(TestDataset, batch_size=batch_num, shuffle=True, num_workers=8, pin_memory=False,
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

    with tqdm(total=len(calculate_loader)) as tbar:
        with torch.set_grad_enabled(False):
            for c, d in enumerate(calculate_loader, 1):

                # d = d.to(torch.device(0))
                method = "ig"
                batch_count = np.unique(d.batch.cpu().detach().numpy(), return_counts=True)[1].tolist()
                id_path = d.absolute_path
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
                    if os.path.isfile(os.path.join(patient_target_dir, 'Node_Ig_sophis.npy')):
                        edge_mask = np.load(os.path.join(patient_target_dir, 'Node_Ig_sophis.npy'))
                        attention_np = np.load(os.path.join(patient_target_dir, 'attention_value.npy'))

                        edge_mask_norm, edge_mask_norm_local = explain(edge_mask, max_IG, min_IG)

                        end_idx_adj = start_idx_adj + adj_batch_item_num

                        inside_row = row[start_idx_adj:end_idx_adj].cpu().detach().numpy()
                        inside_col = col[start_idx_adj:end_idx_adj].cpu().detach().numpy()

                        inside_row = np.subtract(inside_row, node_num_cumul)
                        inside_col = np.subtract(inside_col, node_num_cumul)

                        start_idx_adj = end_idx_adj

                        node_num_cumul = node_num_cumul + batch_item_num
                        print(id_item)
                        Calculate_IG(d, inside_count, inside_row, inside_col, edge_mask,
                                     edge_mask_norm_local, patient_target_dir,
                                     top_threshold, low_threshold,
                                     mid_threshold_top, mid_threshold_low, dataset_root=TrainRoot)
                tbar.update()

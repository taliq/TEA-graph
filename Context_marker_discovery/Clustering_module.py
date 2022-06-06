import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import openslide as osd
from tqdm import tqdm
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from MulticoreTSNE import MulticoreTSNE as TSNE
from lifelines.utils import restricted_mean_survival_time
from lifelines.statistics import multivariate_logrank_test
import numpy.linalg
# from spherecluster.spherecluster import SphericalKMeans

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statistics import median
from random import seed
import cv2 as cv
import networkx as nx
import seaborn as sns
from collections import Counter
from random import sample
import pickle

from Context_marker_discovery.Clustering_utils import run_tsne
from Context_marker_discovery.Clustering_utils import Visualize_tsne

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')

def Subgraph_feature_prep(rootdir, pt_rootdir, IG_type, Patient_metadata):

    Patient_dir_list = os.listdir(rootdir)
    Patient_ID_list = [item for item in Patient_dir_list]
    Patient_dir_list = [os.path.join(rootdir, item) for item in Patient_dir_list]

    Surv = np.zeros(len(Patient_dir_list))
    Event = np.zeros(len(Patient_dir_list))

    IG_whole_feature = []
    IG_ID_list = []
    IG_length_list = []

    exclude_candidate = []

    with tqdm(total=len(Patient_dir_list)) as pbar:
        for c, Patient_ID in enumerate(Patient_dir_list):
            p_id = '-'.join(Patient_ID.split('/')[-1].split('-')[0:3])
            Match_item = Patient_metadata[Patient_metadata["case_submitter_id"] == p_id]
            if Match_item['vital_status'].tolist()[0] == "Alive":
                Surv[c] = int(float(Match_item['days_to_last_follow_up'].tolist()[0]))
                Event[c] = 0
            else:
                Surv[c] = int(float(Match_item['days_to_death'].tolist()[0]))
                Event[c] = 1

            Subgraph_saved_dir = os.path.join(Patient_ID, 'IG_again')
            if os.path.isdir(Subgraph_saved_dir):

                Subgraph_dir_item = os.listdir(Subgraph_saved_dir)
                Patient_item = [os.path.join(Subgraph_saved_dir, item) for item in Subgraph_dir_item if
                                '_exist_again.csv' in item]
                for IG_item in Patient_item:
                    if IG_type in IG_item:
                        if (os.path.isfile(os.path.join(Patient_ID, 'whole_feature.npy')) & os.path.isfile(os.path.join(pt_rootdir, Patient_ID.split('/')[-1].split('_')[0]
                                         + '_0_graph_torch_4.3_artifact_sophis_final.pt'))):
                            Graphfeature = np.load(os.path.join(Patient_ID, 'whole_feature.npy'))
                            Morphfeature = torch.load(
                                os.path.join(pt_rootdir, Patient_ID.split('/')[-1].split('_')[0]
                                             + '_0_graph_torch_4.3_artifact_sophis_final.pt'),
                                map_location='cpu').x
                            Morphfeature = Morphfeature.detach().cpu().numpy()
                            Wholefeature = np.concatenate((Graphfeature, Morphfeature), 1)
                            Subgraph_whole_list = []
                            IG_subgraph_node_info = pd.read_csv(IG_item)
                            length_list = []
                            ### Collect the Subgraph feature of specific IG(High, Mid, Low
                            for item in range(IG_subgraph_node_info.shape[0]):
                                IG_list = np.array(IG_subgraph_node_info.iloc[item].dropna())[1:]
                                if len(IG_list) > 5:
                                    length_list.append(len(IG_list))
                                    if len(Subgraph_whole_list) == 0:
                                        Subgraph_whole_temp = Wholefeature[IG_list.astype(int)].mean(axis=0)
                                        Subgraph_whole_temp = np.reshape(Subgraph_whole_temp,
                                                                            (1, Subgraph_whole_temp.shape[0]))
                                        Subgraph_whole_list = Subgraph_whole_temp
                                    else:
                                        Subgraph_whole_temp = Wholefeature[IG_list.astype(int)].mean(axis=0)
                                        Subgraph_whole_temp = np.reshape(Subgraph_whole_temp,
                                                                            (1, Subgraph_whole_temp.shape[0]))
                                        Subgraph_whole_list = np.concatenate((Subgraph_whole_list, Subgraph_whole_temp), 0)

                            if len(Subgraph_whole_list) > 0:
                                IG_ID_list.extend([Patient_ID_list[c]] * Subgraph_whole_list.shape[0])
                                IG_length_list.extend(length_list)
                                if len(IG_whole_feature) == 0:
                                    IG_whole_feature = Subgraph_whole_list
                                else:
                                    IG_whole_feature = np.concatenate((IG_whole_feature, Subgraph_whole_list), 0)

            pbar.update()

    Scaler = StandardScaler()
    IG_whole_feature = Scaler.fit_transform(IG_whole_feature)

    IG_ID_list = np.array(IG_ID_list)

    return IG_whole_feature, Patient_ID_list, IG_ID_list, IG_length_list, Surv, Event

def Save_dir_create(rootdir, savedir_name, IG_type):

    savedir_root = os.path.join(rootdir, "biomarker_cluster")
    if not os.path.exists(savedir_root):
        os.mkdir(savedir_root)

    savedir = os.path.join(savedir_root, savedir_name)
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    graph_savedir = os.path.join(savedir, 'graph_level_analysis')
    if not os.path.exists(graph_savedir):
        os.mkdir(graph_savedir)
    patch_savedir = os.path.join(savedir, 'patch_level_analysis')
    if not os.path.exists(patch_savedir):
        os.mkdir(patch_savedir)

    graph_cluster_results_dir = os.path.join(graph_savedir, IG_type)
    if not os.path.exists(graph_cluster_results_dir):
        os.mkdir(graph_cluster_results_dir)
    patch_cluster_results_dir = os.path.join(patch_savedir, IG_type)
    if not os.path.exists(patch_cluster_results_dir):
        os.mkdir(patch_cluster_results_dir)

    return graph_cluster_results_dir, patch_cluster_results_dir

def Area_under_the_plot(graph_cluster_num, Whole_count_df, graph_save_dir):

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    rmst_high_dict = {}
    rmst_low_dict = {}
    rmst_difference_dict = {}
    pvalue_dict = {}

    for label_num in range(graph_cluster_num):
        temp = Whole_count_df[Whole_count_df[label_num] == 1]
        kmf_high = kmf.fit(temp['Surv'], temp['Event'], label="High count")
        rmst_high = restricted_mean_survival_time(kmf_high, t=max(Whole_count_df['Surv']))
        rmst_high_dict[label_num] = rmst_high
        kmf.plot_survival_function()
        temp = Whole_count_df[Whole_count_df[label_num] == 0]
        duration = Whole_count_df['Surv']
        event = Whole_count_df['Event']
        label_group = list(map(int, list(Whole_count_df[label_num])))
        result = multivariate_logrank_test(duration, label_group, event)
        pvalue_dict[label_num] = result.p_value
        kmf_low = kmf.fit(temp['Surv'], temp['Event'], label='Low count')
        rmst_low = restricted_mean_survival_time(kmf_low, t=max(Whole_count_df['Surv']))
        rmst_low_dict[label_num] = rmst_low
        rmst_difference = rmst_low - rmst_high
        rmst_difference_dict[label_num] = rmst_difference
        kmf.plot_survival_function()
        plt.savefig(os.path.join(graph_save_dir, str(label_num) + '.png'), transparent=True)
        plt.savefig(os.path.join(graph_save_dir, str(label_num) + '.pdf'), transparent=True)
        plt.cla()
        plt.clf()
        plt.close()

    sorted_diff = {str(k): v for k, v in sorted(rmst_difference_dict.items(), key=lambda item: item[1], reverse=True)}
    bar = plt.bar(*zip(*sorted_diff.items()))
    cmap = matplotlib.cm.get_cmap("Set1", graph_cluster_num)
    for idx, key in enumerate(sorted_diff.keys()):
        bar[idx].set_color(cmap(int(key)))
    plt.savefig(os.path.join(graph_save_dir, 'Areaunderplot.pdf'), transparent=True)
    plt.cla()
    plt.clf()
    plt.close()

    return rmst_difference_dict

def Subgraph_level_cluster_analysis(IG_feature, Patient_ID_list, graph_cluster_num,
                        IG_ID_list, length_list, Surv, Event,
                        IG_type, graph_save_dir):

    print("run tsne")
    embeddings = run_tsne(IG_feature)
    kmeans = KMeans(n_clusters=graph_cluster_num, random_state=0, verbose=1).fit(IG_feature)
    Visualize_tsne(embeddings, IG_type, graph_cluster_num,
                   graph_save_dir, kmeans.labels_)
    print("end tsne")

    _indices = kmeans.labels_

    Unique_patient_ID, Unique_patient_index \
        = np.unique(IG_ID_list, return_inverse=True)
    Subgraph_list = Unique_patient_ID[Unique_patient_index].tolist()
    for patient_index_sort in range(max(Unique_patient_index) + 1):
        Index_match_pos = np.where(Unique_patient_index == patient_index_sort)[0]
        for match_count, match_pos in enumerate(Index_match_pos):
            Subgraph_list[match_pos] = Subgraph_list[match_pos] + '_' + str(match_count)

    subgraph_df = pd.DataFrame(Subgraph_list, columns=['subgraph'])
    subgraph_df.to_csv(os.path.join(graph_save_dir, 'Subgraph_label_list.csv'))

    # label_subgraph_dict = subgraph_list per label
    label_subgraph_dict = {}
    label_length_dict = {}
    for index, label in enumerate(_indices):
        if label in label_subgraph_dict.keys():
            label_subgraph_dict[label].append(Subgraph_list[index])
            label_length_dict[label].append(length_list[index])
        else:
            label_subgraph_dict[label] = [Subgraph_list[index]]
            label_length_dict[label] = [length_list[index]]

    length_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in label_length_dict.items()]))
    label_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in label_subgraph_dict.items()]))
    label_df.to_csv(os.path.join(graph_save_dir, 'Subgraph_list_per_label.csv'))

    Label_count = np.zeros((len(Patient_ID_list), graph_cluster_num))

    # Count the label per patient for survival analysis
    for c, item in enumerate(Patient_ID_list):
        ID_match = np.isin(IG_ID_list, item)
        index_match = _indices[ID_match]
        index_count = np.unique(index_match, return_counts=True)
        for index_value, index_count_value in zip(index_count[0], index_count[1]):
            Label_count[c][index_value] = index_count_value

    Group_count = pd.DataFrame(Label_count)
    Patient_filter = Group_count.sum(axis=1) > 0

    Group_count_filter = Group_count[Patient_filter.to_numpy().flatten()]
    Group_count_filter.to_csv(os.path.join(graph_save_dir, 'Absolute_graph_cluster_count_per_patient.csv'))

    Patient_ID_list_filter = np.array(Patient_ID_list)[Patient_filter.to_numpy().flatten()]
    Surv_filter = Surv[Patient_filter.to_numpy().flatten()]
    Event_filter = Event[Patient_filter.to_numpy().flatten()]

    Label_group_index = np.zeros((len(Patient_ID_list_filter), graph_cluster_num))
    label_threshold_list = []
    for label_index in range(Group_count_filter.shape[1]):
        label_threshold = Group_count_filter.iloc[:,label_index].mean()
        if label_threshold == 0.0:
            label_threshold = 1
        label_threshold_list.append(label_threshold)
        Label_group_index[:, label_index] = Group_count_filter.iloc[:, label_index] > label_threshold


    Label_group_index = pd.DataFrame(Label_group_index)
    Whole_count_df = pd.DataFrame({'ID': Patient_ID_list_filter, 'Surv': Surv_filter, 'Event': Event_filter})
    Whole_count_df = pd.concat((Whole_count_df, Label_group_index), 1)
    Whole_count_df.to_csv(os.path.join(graph_save_dir, 'Graph_cluster_count_final_results.csv'))

    rmst_difference_dict = Area_under_the_plot(graph_cluster_num, Whole_count_df, graph_save_dir)

    return subgraph_df, label_df, length_df, rmst_difference_dict

def Patch_level_cluster_analysis(rootdir, patch_savedir, label_df,
                                 patch_cluster_num, length_df, pt_rootdir,
                                 IG_type, WSI_rootdir):

    patch_df, graph_label_df, Subgraph_whole_feature, embeddings, patch_label_kmean =\
        patch_clustering_sampling(rootdir, patch_savedir, label_df,
                                  patch_cluster_num, length_df, pt_rootdir, IG_type)

    patch_cluster_save(graph_label_df, patch_cluster_num, patch_savedir,
                       rootdir, pt_rootdir, IG_type, WSI_rootdir)

    return patch_df, graph_label_df, patch_label_kmean


def patch_clustering_sampling(rootdir, savedir, label_df, patch_cluster_num,
                           length_df, pt_rootdir, IG_type):

    label_list = label_df.iloc[:, :]
    patch_dict = {}
    patch_list = []
    sample_dict = {}

    for column in label_list.columns:
        label = int(column)
        sample_list = np.array(list(label_list[column].dropna()))
        length_list = np.array(list(length_df[column].dropna()))
        sample_list_select = list(sample_list[np.where(length_list < 100)])
        if len(sample_list_select) > 5:
            sample_list_select = sample(sample_list_select, 5)
            for samples in sample_list_select:
                sample_dict[samples] = label
            patch_dict[label] = sample_list_select
            patch_list += sample_list_select

    patch_list.sort()
    #patient_list = list(set(patch_list))
    patient_list = list(set([patient.split('_')[0] + '_0' for patient in patch_list]))
    patch_df = pd.DataFrame(patch_list, columns=['patient'])
    Subgraph_whole_list = []
    patch_ID_list = []
    patch_label_list = []

    for Patient in patient_list:
        Patient_ID = os.path.join(rootdir, Patient)
        Patient_item = os.listdir(Patient_ID)
        Graphfeature = np.load(os.path.join(Patient_ID, 'whole_feature.npy'))
        Morphfeature = torch.load(
            os.path.join(pt_rootdir, Patient_ID.split('/')[-1].split('_')[0] + '_0_graph_torch_4.3_artifact_sophis_final.pt'),
            map_location='cpu').x
        Morphfeature = Morphfeature.detach().cpu().numpy()

        Wholefeature = np.concatenate((Graphfeature, Morphfeature), 1)
        New_dir = os.path.join(Patient_ID, 'IG_again')
        New_item = os.listdir(New_dir)
        Patient_item = [os.path.join(New_dir, item) for item in New_item if
                        '_small_cap_subgraph_exist_again.csv' in item]

        Patient_item = [item for item in Patient_item if IG_type in item]
        subgraph_num_list = [patch.split('_')[-1] for patch in patch_list if
                             Patient in patch]

        for IG_item in Patient_item:
            IG_pd = pd.read_csv(IG_item)
            length_list = []
            for item in subgraph_num_list:
                item = int(item)
                IG_list = np.array(IG_pd.iloc[item].dropna())[1:]
                IG_list = IG_list.astype(int)
                length_list.append(len(IG_list))
                Patient_subgraph = Patient + '_' + str(item)
                patch_ID_list.extend([Patient_subgraph] * len(IG_list))
                patch_label_list.extend([sample_dict[Patient_subgraph]] * len(IG_list))

                if len(Subgraph_whole_list) == 0:
                    Subgraph_whole_temp = Wholefeature[IG_list]
                    Subgraph_whole_temp = np.reshape(Subgraph_whole_temp, (-1, Subgraph_whole_temp.shape[1]))
                    Subgraph_whole_list = Subgraph_whole_temp
                else:
                    Subgraph_whole_temp = Wholefeature[IG_list]
                    Subgraph_whole_temp = np.reshape(Subgraph_whole_temp, (-1, Subgraph_whole_temp.shape[1]))
                    Subgraph_whole_list = np.concatenate((Subgraph_whole_list, Subgraph_whole_temp), 0)

    Scaler = StandardScaler()
    Subgraph_whole_feature = Scaler.fit_transform(Subgraph_whole_list)

    graph_label_df = pd.DataFrame(patch_ID_list, columns=["column"])
    embeddings = run_tsne(Subgraph_whole_feature)
    kmeans = KMeans(n_clusters=patch_cluster_num, random_state=0, verbose=1).fit(embeddings)
    Visualize_tsne(embeddings, str(patch_cluster_num), patch_cluster_num, savedir, kmeans.labels_, c1="tab20b")
    graph_label_df['label'] = kmeans.labels_
    graph_label_df['graph_label'] = patch_label_list

    return patch_df, graph_label_df, Subgraph_whole_feature, embeddings, kmeans

def patch_cluster_save(patch_list, patch_cluster_num, patch_save_dir,
                       root_dir, csv_rootdir, IG_type, WSI_rootdir):

    subgraph_name = patch_list['column']
    node_list = []
    before_subgraph = 1
    i = 0
    for subgraph in subgraph_name:
        if subgraph == before_subgraph:
            i += 1
            before_subgraph = subgraph
            node_list.append(i)
        else:
            i = 0
            before_subgraph = subgraph
            node_list.append(i)
    patch_list['node_number'] = node_list

    for cluster in range(patch_cluster_num):
        patch_dir = os.path.join(patch_save_dir, 'patchcluster_' + str(cluster))
        if not os.path.exists(patch_dir):
            os.mkdir(patch_dir)

        cluster_patch = patch_list[patch_list['label'] == cluster]
        sampled_df = cluster_patch.sample(n=10, random_state=1)
        WSI_list = os.listdir(WSI_rootdir)
        WSI_list_split = [item.split('.')[0] for item in WSI_list]

        for subgraph, patchnum in zip(list(sampled_df['column']), list(sampled_df['node_number'])):
            wsi_id = subgraph.split('_')[0]
            if wsi_id in WSI_list_split:
                match_wsi = WSI_list_split.index(wsi_id)
                subgraph_num = int(subgraph.split('_')[-1])
                patient_item = root_dir + '/' + wsi_id + '/IG_again/' + wsi_id + '_' + \
                               IG_type + '_IG_TME_subgraph_small_cap_subgraph_exist_again.csv'
                WSI = osd.open_slide(os.path.join(WSI_rootdir, WSI_list[match_wsi]))
                WSI_level = 1
                WSI_Downsample_ratio = int(WSI.level_downsamples[WSI_level])
                Patch_dim = int(256 / WSI_Downsample_ratio)

                Position_pd = pd.read_csv(patient_item)
                Superpatch_pd = pd.read_csv(os.path.join(csv_rootdir, wsi_id + '_75_4.3_artifact_sophis_final.csv'))
                Location_pd = pd.read_csv(os.path.join(csv_rootdir, wsi_id + '_node_location_list.csv'))

                subgraph_whole_nodes = np.array(Position_pd.iloc[int(subgraph_num)].dropna())[1:]
                subgraph_whole_nodes = list(subgraph_whole_nodes)
                superpatch_node = subgraph_whole_nodes[patchnum]
                superpatch_subnode = list(Superpatch_pd.iloc[int(superpatch_node)].iloc[2:].dropna())
                subgraph_subnode_location = Location_pd.iloc[superpatch_subnode]

                superpatch_whole_max_X = np.array(subgraph_subnode_location['X']).max()
                superpatch_whole_min_X = np.array(subgraph_subnode_location['X']).min()
                superpatch_whole_max_Y = np.array(subgraph_subnode_location['Y']).max()
                superpatch_whole_min_Y = np.array(subgraph_subnode_location['Y']).min()

                TME_width = int(superpatch_whole_max_X -superpatch_whole_min_X  + 1) * Patch_dim
                TME_height = int(superpatch_whole_max_Y -superpatch_whole_min_Y  + 1) * Patch_dim

                patch_image = WSI.read_region((superpatch_whole_min_X * 256, superpatch_whole_min_Y * 256), WSI_level, (TME_width, TME_height))
                patch_image.save(os.path.join(patch_dir, wsi_id + '_' + str(subgraph_num) + '_' + str(patchnum) + '_TME.png'))


def Final_visualization(sampled_patch, kmeans, graph_cluster_num, patch_cluster_num, savedir,
                  label_list, root_dir, pt_rootdir, csv_rootdir, WSI_rootdir, IG_type):

    sampled_patch['label'] = kmeans.labels_
    label_dict = {}

    for label in range(graph_cluster_num):
        label_dir = os.path.join(savedir, str(label))
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        Subgraph_biomarker_visualization(sampled_patch[sampled_patch['graph_label'] == label],
                                         root_dir, pt_rootdir, csv_rootdir,
                                         WSI_rootdir, label_dir, IG_type,
                                         patch_cluster_num, label_dict, label)

def Subgraph_biomarker_visualization(label_df, root_dir, pt_rootdir, csv_rootdir,
                                     WSI_rootdir, label_dir, IG_type,
                                     patch_cluster_num, label_dict, label):

    edge_array = np.zeros((patch_cluster_num, patch_cluster_num))

    sampled_list = list(set(label_df.iloc[:, 0]))

    WSI_list = os.listdir(WSI_rootdir)
    WSI_list_split = [item.split('.')[0] for item in WSI_list]

    for subgraph in sampled_list:
        subgraph_df = label_df[label_df['column'] == subgraph]
        wsi_id = subgraph.split('_')[0]
        if wsi_id in WSI_list_split:
            match_wsi = WSI_list_split.index(wsi_id)
            subgraph_num = int(subgraph.split('_')[-1])
            patient_item = root_dir + '/' + wsi_id + '/IG_again/' + wsi_id + '_' + IG_type + '_IG_TME_subgraph_small_cap_subgraph_exist_formal_setting_normal_filter.csv'

            WSI = osd.open_slide(os.path.join(WSI_rootdir, WSI_list[match_wsi]))
            WSI_level = 1
            WSI_Downsample_ratio = int(WSI.level_downsamples[WSI_level])
            Patch_dim = int(256 / WSI_Downsample_ratio)
            WSI_width, WSI_height = WSI.level_dimensions[WSI_level]
            WSI_width = int(WSI_width)
            WSI_height = int(WSI_height)
            Pytorch_data = torch.load(os.path.join(pt_rootdir, wsi_id + '_0_graph_torch_4.3_artifact_sophis_final.pt'))
            Position_pd = pd.read_csv(patient_item)
            Superpatch_pd = pd.read_csv(os.path.join(csv_rootdir, wsi_id + '_75_4.3_artifact_sophis_final.csv'))
            Location_pd = pd.read_csv(os.path.join(csv_rootdir, wsi_id + '_node_location_list.csv'))

            subgraph_whole_nodes = np.array(Position_pd.iloc[int(subgraph_num)].dropna())[2:]
            subgraph_whole_nodes = list(subgraph_whole_nodes)
            subgraph_real_loc = Superpatch_pd.iloc[subgraph_whole_nodes]['Unnamed: 0.1']
            subgraph_real_loc = list(subgraph_real_loc)
            subgraph_whole_nodes_location = Location_pd.iloc[subgraph_real_loc]
            node_label_dict = {}
            node_label_list = []
            for index, node in enumerate(subgraph_whole_nodes):
                node_label_dict[node] = subgraph_df.iloc[index, 1]
                node_label_list.append(subgraph_df.iloc[index, 1])
            subgraph_whole_max_X = np.array(subgraph_whole_nodes_location['X']).max()
            subgraph_whole_min_X = np.array(subgraph_whole_nodes_location['X']).min()
            subgraph_whole_max_Y = np.array(subgraph_whole_nodes_location['Y']).max()
            subgraph_whole_min_Y = np.array(subgraph_whole_nodes_location['Y']).min()

            TME_width = int(subgraph_whole_max_X - subgraph_whole_min_X + 1) * Patch_dim
            TME_height = int(subgraph_whole_max_Y - subgraph_whole_min_Y + 1) * Patch_dim

            TME_image = WSI.read_region((subgraph_whole_min_X * 256, subgraph_whole_min_Y * 256), WSI_level,
                                        (TME_width, TME_height))
            TME_mask = np.zeros((TME_height, TME_width))
            TME_image.save(os.path.join(label_dir, wsi_id + '_' + str(subgraph_num) + '_TME.png'))
            TME_image = TME_image.convert('RGB')
            subgraph_reference_X = np.array(subgraph_whole_nodes_location['X']) - np.array(
                subgraph_whole_nodes_location['X']).min()
            subgraph_reference_Y = np.array(subgraph_whole_nodes_location['Y']) - np.array(
                subgraph_whole_nodes_location['Y']).min()
            for X, Y in zip(subgraph_reference_X, subgraph_reference_Y):
                TME_mask[Y * Patch_dim: (Y + 1) * Patch_dim, X * Patch_dim: (X + 1) * Patch_dim] = 2
            TME_mask = cv.applyColorMap(np.uint8(255 * TME_mask), cv.COLORMAP_JET)
            TME_with_mask = np.float32(TME_image) + 0.5 * np.float32(TME_mask)
            cv.imwrite(os.path.join(label_dir, wsi_id + '_' + str(subgraph_num) + '_TME_with_mask.png'),
                       np.array(TME_with_mask))

            TME_graph = nx.Graph()
            TME_graph.add_nodes_from(subgraph_whole_nodes)

            row, col = Pytorch_data.edge_index
            row = row.cpu().detach().numpy()
            col = col.cpu().detach().numpy()
            row_select = np.isin(row, np.array(subgraph_whole_nodes))
            col_select = np.isin(col, np.array(subgraph_whole_nodes))
            edge_idx_select = row_select * col_select
            row = row[edge_idx_select]
            col = col[edge_idx_select]
            WSI_edge_index = [(row_item, col_item) for row_item, col_item in zip(row, col)]
            for first, second in zip(row, col):
                label_one = node_label_dict[first]
                label_two = node_label_dict[second]
                if label_one > label_two:
                    edge_array[label_one, label_two] += 1
                else:
                    edge_array[label_two, label_one] += 1
            TME_graph.add_edges_from(WSI_edge_index)

            TME_pos_x = subgraph_reference_X
            TME_pos_y = subgraph_reference_Y
            TME_pos = [(pos_x * Patch_dim + int(Patch_dim / 2), pos_y * Patch_dim + int(Patch_dim / 2))
                       for pos_x, pos_y in zip(TME_pos_x, TME_pos_y)]
            TME_pos = dict(zip(subgraph_whole_nodes, TME_pos))
            my_dpi = 96
            plt.figure(figsize=(TME_image.size[0] / my_dpi, TME_image.size[1] / my_dpi), dpi=96)
            plt.axis('off')
            nx.draw_networkx(TME_graph, pos=TME_pos, node_size=100, width=1,
                             arrows=False, node_color=node_label_list,
                             with_labels=False, labels=node_label_dict,
                             cmap=plt.cm.get_cmap("tab20b", patch_cluster_num),
                             vmin=0, vmax=patch_cluster_num)
            plt.gca().invert_yaxis()
            plt.savefig(os.path.join(label_dir, wsi_id + '_' + str(subgraph_num) + '_TME_graph_wo_IG.pdf'),
                        transparent=True)
            plt.close()
            plt.clf()
            plt.cla()

            my_dpi = 96
            plt.figure(figsize=(TME_image.size[0] / my_dpi, TME_image.size[1] / my_dpi), dpi=96)
            plt.axis('off')
            nx.draw_networkx(TME_graph, pos=TME_pos, node_size=100, width=1,
                             arrows=False, node_color=node_label_list,
                             with_labels=True, labels=node_label_dict,
                             cmap=plt.cm.get_cmap("tab20b", patch_cluster_num),
                             vmin=0, vmax=patch_cluster_num)
            plt.gca().invert_yaxis()
            # plt.subplots_adjust(left=0., right=0., top=0., bottom=0.)
            plt.savefig(os.path.join(label_dir, wsi_id + '_' + str(subgraph_num) + '_TME_graph_wo+IG_w_label.pdf'),
                        transparent=True)
            plt.close()
            plt.clf()
            plt.cla()

    normalize = (edge_array - edge_array.min()) / (edge_array.max() - edge_array.min())
    label_dict[label] = normalize
    sns.heatmap(label_dict[label], cmap='plasma')
    plt.savefig(os.path.join(label_dir, 'edge_heatmap.pdf'), transparent=True)
    plt.cla()
    plt.clf()
    plt.close()

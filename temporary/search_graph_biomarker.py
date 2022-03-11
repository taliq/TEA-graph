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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from random import sample
from random import seed
import cv2 as cv
import networkx as nx
import seaborn as sns
from collections import Counter
import argparse

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')

def calculate_label_distribution(patch_list, graph_cluster_num, cluster_num, save_dir):
    patch_dict = {}
    label_num = {}
    for i in range(graph_cluster_num):
        label_dir = os.path.join(save_dir, str(i))
        if os.path.exists(label_dir) == False:
            os.mkdir(label_dir)
        label_df = patch_list[patch_list['graph_label'] == i]
        label_distrib = list(label_df['label'])
        label_num[str(i)] = len(label_distrib)
        result = Counter(label_distrib)
        label_dict = {}
        for key in result:
            label_dict[key] = result[key]
        for key in range(cluster_num):
            if key not in label_dict.keys():
                label_dict[key] = 0
        plt.figure(figsize=(10,10))
        cmap = matplotlib.cm.get_cmap("tab20b", cluster_num)
        rescale = lambda y:(y-np.min(y)) / (np.max(y) - np.min(y))
        label_dict = dict(sorted(label_dict.items()))
        xtick_label_position = list(range(cluster_num))
        plt.xticks(xtick_label_position, range(cluster_num))
        a = plt.bar(xtick_label_position, label_dict.values())
        for j in range(cluster_num):
            a[j].set_color(cmap(j))
        plt.title('cluster' + str(i), fontsize = 10)
        plt.xlabel('patch label')
        plt.ylabel('num')
        plt.savefig(os.path.join(label_dir,str(i) + '_distribution.pdf'), transparent = True)
        plt.savefig(os.path.join(label_dir, str(i) + '_distribution.png'), transparent=True)
        plt.cla()
        plt.clf()
        plt.close()
        patch_dict[i] = label_dict

    plt.bar(range(graph_cluster_num), label_num.values())
    plt.savefig(os.path.join(save_dir, 'patchnum_distribution.pdf'), transparent=True)
    plt.savefig(os.path.join(save_dir, 'patchnum_distribution.png'), transparent=True)
    plt.cla()
    plt.clf()
    plt.close()

def calculate_label_edge_distribution(label_df, label_dir, edge_array, graph_custer_num, patch_cluster_num, root_dir, pt_rootdir, csv_rootdir, IG_type):
    WSI_root = '/mnt/nvme1n1/CCRCC_WSI/'
    WSI_list = os.listdir(WSI_root)
    WSI_list = [os.path.join(WSI_root, item) for item in WSI_list]
    Dataset_root = '/mnt/nvme0n1/supernode_WSI/'
    Dataset_list = os.listdir(Dataset_root)
    Dataset_list = [os.path.join(Dataset_root, item) for item in Dataset_list]
    WSI_ID_link = pd.read_excel('/home/taliq_lee/DSA/WSI_CCRCC_Match.xlsx')
    sampled_list = list(set(label_df.iloc[:,0]))
    for subgraph in sampled_list:
        subgraph_df = label_df[label_df['column']==subgraph]
        patient = subgraph.split('_')[0]
        wsi_id = ('_').join(subgraph.split('_')[0:5])
        subgraph_num = int(subgraph.split('_')[-1])
        WSI_name = list(WSI_ID_link[WSI_ID_link['path ID'] == patient]['deidentify'])
        WSI_name = [item for item in WSI_name if ('_').join(subgraph.split('_')[1:5]) in item]

        patient_item = root_dir + wsi_id + '_0/IG_again/' + wsi_id + '_0_' + IG_type +'_IG_TME_subgraph_small_cap_subgraph_exist_formal_setting_normal_filter.csv'

        WSI = osd.open_slide('/mnt/nvme1n1/CCRCC_WSI/' + WSI_name[0] + '.svs')
        WSI_level = 1
        WSI_Downsample_ratio = int(WSI.level_downsamples[WSI_level])
        Patch_dim = int(256 / WSI_Downsample_ratio)
        WSI_width, WSI_height = WSI.level_dimensions[WSI_level]
        WSI_width = int(WSI_width)
        WSI_height = int(WSI_height)
        Pytorch_data = torch.load(os.path.join(pt_rootdir, wsi_id + '_0_graph_torch_4.3_artifact_sophis_final.pt'))
        Position_pd = pd.read_csv(patient_item)
        Superpatch_pd = pd.read_csv(os.path.join(csv_rootdir, wsi_id + '_0_4.3_artifact_sophis_final.csv'))
        Location_pd = pd.read_csv(os.path.join(pt_rootdir, wsi_id + '_node_location_list.csv'))

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

        if ((subgraph_whole_min_X * Patch_dim + 30) > (WSI.level_dimensions[WSI_level][0]) or
                (subgraph_whole_max_X * Patch_dim + 30) > (WSI.level_dimensions[WSI_level][0]) or
                (subgraph_whole_min_Y * Patch_dim + 30) > (WSI.level_dimensions[WSI_level][1]) or
                (subgraph_whole_max_Y * Patch_dim + 30) > (WSI.level_dimensions[WSI_level][1])):
            print(wsi_id)
            pass

        else:
            TME_image = WSI.read_region((subgraph_whole_min_X * 256, subgraph_whole_min_Y * 256), WSI_level, (TME_width, TME_height))
            TME_mask = np.zeros((TME_height, TME_width))
            TME_image.save(os.path.join(label_dir, wsi_id + '_' + str(subgraph_num) + '_TME.png'))
            TME_image = TME_image.convert('RGB')
            subgraph_reference_X = np.array(subgraph_whole_nodes_location['X']) - np.array(subgraph_whole_nodes_location['X']).min()
            subgraph_reference_Y = np.array(subgraph_whole_nodes_location['Y']) - np.array(subgraph_whole_nodes_location['Y']).min()
            for X, Y in zip(subgraph_reference_X, subgraph_reference_Y):
                TME_mask[Y * Patch_dim: (Y + 1) * Patch_dim, X * Patch_dim: (X + 1) * Patch_dim] = 2
            TME_mask = cv.applyColorMap(np.uint8(255 * TME_mask), cv.COLORMAP_JET)
            TME_with_mask = np.float32(TME_image) + 0.5 * np.float32(TME_mask)
            cv.imwrite(os.path.join(label_dir, wsi_id + '_' + str(subgraph_num) + '_TME_with_mask.png'), np.array(TME_with_mask))

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
                    edge_array[label_one, label_two] +=1
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
                             arrows=False, node_color = node_label_list,
                             with_labels=False, labels = node_label_dict, cmap=plt.cm.get_cmap("tab20b", patch_cluster_num),
                             vmin = 0, vmax = patch_cluster_num)
            plt.savefig(os.path.join(label_dir, wsi_id + '_' + str(subgraph_num) + '_TME_graph_wo_IG.png'))
            plt.savefig(os.path.join(label_dir, wsi_id + '_' + str(subgraph_num) + '_TME_graph_wo_IG.pdf'), transparent = True)
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
            plt.savefig(os.path.join(label_dir,  wsi_id + '_' + str(subgraph_num) + '_TME_graph_wo_IG_w_label.png'))
            plt.savefig(os.path.join(label_dir, wsi_id + '_' + str(subgraph_num) + '_TME_graph_wo+IG_w_label.pdf'), transparent = True)
            plt.close()
            plt.clf()
            plt.cla()

    return edge_array

def Visualize_tsne(embeddings,feat, cluster_num, save_dir, label, c1 = "tab20c"):
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    cmap = matplotlib.cm.get_cmap(c1, cluster_num)
    plt.scatter(vis_x, vis_y, c = label, cmap = cmap)
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, feat + '_tsne.png'), transparent = True)
    plt.savefig(os.path.join(save_dir, feat + '_tsne.pdf'), transparent = True)
    plt.cla()
    plt.clf()
    plt.close()

def run_tsne(X):
    tsne = TSNE(n_jobs=50)
    embeddings = tsne.fit_transform(X)

    return embeddings

def clustering(feature, Patient_ID_list, cluster_num, ID_list, length_list, Surv, Event, IG, save_dir):

    Figure_dir = os.path.join(save_dir, IG)
    if os.path.exists(Figure_dir) == False:
        os.mkdir(Figure_dir)
    embeddings = run_tsne(feature)
    kmeans = KMeans(n_clusters=cluster_num, random_state=0, verbose=1).fit(embeddings)
    Visualize_tsne(embeddings, IG, cluster_num, Figure_dir, kmeans.labels_)

    _indices = kmeans.labels_
    subgraph_list = []
    before_patient = 0
    patient_num = 0

    # Subgraph name list
    for patient in ID_list:
        if before_patient == patient:
            subgraph_list.append(patient + '_' + str(patient_num))
            patient_num += 1
        else:
            patient_num = 0
            subgraph_list.append(patient + '_' + str(patient_num))
            before_patient = patient
            patient_num += 1

    label_subgraph_dict = {}
    label_length_dict = {}
    for index, label in enumerate(_indices):
        if label in label_subgraph_dict.keys():
            label_subgraph_dict[label].append(subgraph_list[index])
            label_length_dict[label].append(length_list[index])
        else:
            label_subgraph_dict[label] = [subgraph_list[index]]
            label_length_dict[label] = [length_list[index]]

    subgraph_df = pd.DataFrame(subgraph_list, columns = ['subgraph'])
    subgraph_df.to_csv(os.path.join(Figure_dir, 'subgraph_list.csv'))
    length_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in label_length_dict.items()]))
    label_df = pd.DataFrame(dict([(k,pd.Series(v)) for k, v in label_subgraph_dict.items()]))
    label_df.to_csv(os.path.join(Figure_dir, 'label_list.csv'))
    Label_count = np.zeros((len(Patient_ID_list), cluster_num))
    
    #Count the label per patient for survival analysis
    for c, item in enumerate(Patient_ID_list):
        ID_match = np.isin(ID_list, item)
        index_match = _indices[ID_match]
        index_count = np.unique(index_match, return_counts=True)
        for index_value, index_count_value in zip(index_count[0], index_count[1]):
            Label_count[c][index_value] = index_count_value

    Group_count = pd.DataFrame(Label_count)
    Group_count.to_csv(os.path.join(Figure_dir, 'count.csv'))
    Label_group_index = np.zeros((len(Patient_ID_list), cluster_num))

    for item in range(Group_count.shape[1]):
        Label_group_index[:, item][np.array((Group_count[item] > Group_count[item].median()))] = 1

    Label_group_index = pd.DataFrame(Label_group_index)
    Whole_count_df = pd.DataFrame({'ID': Patient_ID_list, 'Surv': Surv,  'Event': Event})
    Whole_count_df = pd.concat((Whole_count_df, Label_group_index), 1)
    Whole_count_df.to_csv(os.path.join(Figure_dir, 'Whole_count.csv'))

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    rmst_high_dict = {}
    rmst_low_dict = {}
    rmst_difference_dict = {}
    pvalue_dict = {}
    for label_num in range(cluster_num):
        temp = Whole_count_df[Whole_count_df[label_num] == 1]
        kmf_high = kmf.fit(temp['Surv'], temp['Event'], label = "High count")
        rmst_high = restricted_mean_survival_time(kmf_high, t = max(Whole_count_df['Surv']))
        rmst_high_dict[label_num] = rmst_high
        kmf.plot_survival_function()
        temp = Whole_count_df[Whole_count_df[label_num] == 0]
        duration = Whole_count_df['Surv']
        event = Whole_count_df['Event']
        label_group = list(map(int,list(Whole_count_df[label_num])))
        result = multivariate_logrank_test(duration, label_group, event)
        pvalue_dict[label_num] = result.p_value
        kmf_low = kmf.fit(temp['Surv'], temp['Event'], label = 'Low count')
        rmst_low = restricted_mean_survival_time(kmf_low, t=max(Whole_count_df['Surv']))
        rmst_low_dict[label_num] = rmst_low
        rmst_difference = rmst_low - rmst_high
        rmst_difference_dict[label_num] = rmst_difference
        kmf.plot_survival_function()
        plt.savefig(os.path.join(Figure_dir, str(label_num) + '.png'), transparent = True)
        plt.savefig(os.path.join(Figure_dir, str(label_num) + '.pdf'), transparent = True)
        plt.cla()
        plt.clf()
        plt.close()

    pvalue_df = pd.DataFrame(pvalue_dict, index = [0])
    pvalue_df.to_csv(os.path.join(Figure_dir, 'p_value.csv'))
    sorted_diff = {str(k): v for k, v in sorted(rmst_difference_dict.items(), key=lambda item: item[1], reverse = True)}
    bar = plt.bar(*zip(*sorted_diff.items()))
    cmap = matplotlib.cm.get_cmap("tab20c", cluster_num)
    for idx, key in enumerate(sorted_diff.keys()):
        bar[idx].set_color(cmap(int(key)))
    plt.savefig(os.path.join(Figure_dir, 'Areaunderplot.png'), transparent = True)
    plt.savefig(os.path.join(Figure_dir, 'Areaunderplot.pdf'), transparent = True)
    plt.cla()
    plt.clf()
    plt.close()
    cph = CoxPHFitter(penalizer=0.001)
    test = cph.fit(Whole_count_df[Whole_count_df.columns[1:]], 'Surv', event_col='Event')
    cph.plot()
    plt.savefig(os.path.join(Figure_dir, 'CPH_HR.png'), transparent = True)
    plt.savefig(os.path.join(Figure_dir, 'CPH_HR.pdf'), transparent = True)
    plt.cla()
    plt.clf()
    plt.close()

    return subgraph_df, label_df, length_df, rmst_difference_dict, pvalue_df

def cluster_analysis(rootdir, pt_rootdir, threshold, IG_type):
    threshold = int(threshold)
    CCRCC_dataset = pd.read_excel('/home/taliq_lee/DSA/1995-2008_CCRCC.xlsx')

    Patient_dir_list = os.listdir(rootdir)

    Patient_ID_list = [('_').join(item.split('_')[0:5]) for item in Patient_dir_list]
    Patient_dir_list = [os.path.join(rootdir, item) for item in Patient_dir_list]

    Surv = np.zeros(len(Patient_dir_list))
    Event = np.zeros(len(Patient_dir_list))

    IG_whole_feature = []
    IG_ID_list = []
    IG_length_list = []

    with tqdm(total=len(Patient_dir_list)) as pbar:
        for c, Patient_ID in enumerate(Patient_dir_list):
            Surv[c] = CCRCC_dataset[CCRCC_dataset['Path No'] == Patient_ID.split('/')[-1].split('_')[0]][CCRCC_dataset.columns[7]].item()
            Event[c] = CCRCC_dataset[CCRCC_dataset['Path No'] == Patient_ID.split('/')[-1].split('_')[0]][CCRCC_dataset.columns[8]].item()
            Patient_item = os.listdir(Patient_ID)
            Graphfeature = torch.load(os.path.join(Patient_ID, 'whole_feature.pt'), map_location = 'cpu')
            Morphfeature = torch.load(os.path.join(pt_rootdir, Patient_ID.split('/')[-1] + '_graph_torch_4.3_artifact_sophis_final.pt'), map_location = 'cpu').x
            Wholefeature = torch.cat((Graphfeature, Morphfeature), 1)
            New_dir = os.path.join(Patient_ID, 'IG_again')
            New_item = os.listdir(New_dir)
            Patient_item = [os.path.join(New_dir,item) for item in New_item if '_exist_formal_setting_normal_filter.csv' in item]
            for IG_item in Patient_item:
                if IG_type in IG_item:
                    Subgraph_whole_list = []
                    IG_pd = pd.read_csv(IG_item)
                    length_list = []
                    ### Collect the Subgraph feature of specific IG(High, Mid, Low
                    for item in range(IG_pd.shape[0]):
                        IG_list = np.array(IG_pd.iloc[item].dropna())[1:]
                        if len(IG_list) > threshold:
                            length_list.append(len(IG_list))
                            if len(Subgraph_whole_list) == 0:
                                Subgraph_whole_temp = Wholefeature[IG_list.astype(int)].mean(dim=0)
                                Subgraph_whole_temp = torch.reshape(Subgraph_whole_temp, (1, Subgraph_whole_temp.shape[0]))
                                Subgraph_whole_list = Subgraph_whole_temp
                            else:
                                Subgraph_whole_temp = Wholefeature[IG_list.astype(int)].mean(dim=0)
                                Subgraph_whole_temp = torch.reshape(Subgraph_whole_temp, (1, Subgraph_whole_temp.shape[0]))
                                Subgraph_whole_list = torch.cat((Subgraph_whole_list, Subgraph_whole_temp), 0)
                    
                    if len(Subgraph_whole_list) > 0:
                        IG_ID_list.extend([Patient_ID_list[c]] * Subgraph_whole_list.shape[0])
                        IG_length_list.extend(length_list)
                        if len(IG_whole_feature) == 0:
                            IG_whole_feature = Subgraph_whole_list
                        else:
                            IG_whole_feature = torch.cat((IG_whole_feature, Subgraph_whole_list), 0)     
            pbar.update()

    Scaler = StandardScaler()
    IG_whole_feature = Scaler.fit_transform(IG_whole_feature.detach().cpu().numpy())


    IG_ID_list = np.array(IG_ID_list)

    return IG_whole_feature, Patient_ID_list, IG_ID_list, IG_length_list, Surv, Event, Patient_item


def patch_sampling(rootdir, savedir, subgraph_df, label_df, patch_cluster_num, length_df, pt_rootdir, IG_type):
    savedir = os.path.join(savedir, str(patch_cluster_num))
    if os.path.exists(savedir) == False:
        os.mkdir(savedir)
    label_list = label_df.iloc[:, :]
    patch_dict = {}
    patch_list = []
    sample_dict = {}
    # sample_list - select 100 subgraphs per label
    # sample_dict - subgraph is annotated with label
    # patch_dict - subgraph in specific label
    # patch_list - all sampled subgraph
    for column in label_list.columns:
        label = int(column)
        sample_list = np.array(list(label_list[column].dropna()))
        length_list = np.array(list(length_df[column].dropna()))
        sample_list = list(sample_list[np.where(length_list < 100)])
        sample_list = sample(sample_list, 100)
        for samples in sample_list:
            sample_dict['_'.join(samples.split('_')[0:5]) + '_0_' + samples.split('_')[-1]] = label
        patch_dict[label] = sample_list
        patch_list += sample_list

    patch_list.sort()
    patient_list = list(set([('_').join(patient.split('_')[0:5] + ['0']) for patient in patch_list]))
    patch_df = pd.DataFrame(patch_list, columns=['patient'])
    Subgraph_whole_list = []
    patch_ID_list = []
    patch_label_list = []

    for Patient in patient_list:
        Patient_ID = os.path.join(rootdir, Patient)
        Patient_item = os.listdir(Patient_ID)
        Graphfeature = torch.load(os.path.join(Patient_ID, 'whole_feature.pt'), map_location='cpu')
        Morphfeature = torch.load(os.path.join(pt_rootdir, Patient_ID.split('/')[-1] + '_graph_torch_4.3_artifact_sophis_final.pt'),
                                  map_location='cpu').x

        Wholefeature = torch.cat((Graphfeature, Morphfeature), 1)
        New_dir = os.path.join(Patient_ID, 'IG_again')
        New_item = os.listdir(New_dir)
        Patient_item = [os.path.join(New_dir, item) for item in New_item if
                        '_exist_formal_setting_normal_filter.csv' in item]

        Patient_item = [item for item in Patient_item if IG_type in item]
        subgraph_num_list = [patch.split('_')[-1] for patch in patch_list if
                             ('_').join(Patient.split('_')[0:5]) in patch]

        for IG_item in Patient_item:
            IG_pd = pd.read_csv(IG_item)
            length_list = []
            for item in subgraph_num_list:
                item = int(item)
                IG_list = np.array(IG_pd.iloc[item].dropna())[1:]
                length_list.append(len(IG_list))
                Patient_subgraph = Patient + '_' + str(item)
                patch_ID_list.extend([Patient_subgraph] * len(IG_list))
                patch_label_list.extend([sample_dict[Patient_subgraph]] * len(IG_list))

                if len(Subgraph_whole_list) == 0:
                    Subgraph_whole_temp = Wholefeature[IG_list]
                    Subgraph_whole_temp = torch.reshape(Subgraph_whole_temp, (-1, Subgraph_whole_temp.shape[1]))
                    Subgraph_whole_list = Subgraph_whole_temp
                else:
                    Subgraph_whole_temp = Wholefeature[IG_list]
                    Subgraph_whole_temp = torch.reshape(Subgraph_whole_temp, (-1, Subgraph_whole_temp.shape[1]))
                    Subgraph_whole_list = torch.cat((Subgraph_whole_list, Subgraph_whole_temp), 0)

    Scaler = StandardScaler()
    Subgraph_whole_feature = Scaler.fit_transform(Subgraph_whole_list.detach().cpu().numpy())
    
    graph_label_df = pd.DataFrame(patch_ID_list, columns=["column"])
    embeddings = run_tsne(Subgraph_whole_feature)
    kmeans = KMeans(n_clusters=patch_cluster_num, random_state=0, verbose=1).fit(embeddings)
    Visualize_tsne(embeddings, str(patch_cluster_num), patch_cluster_num, savedir, kmeans.labels_, c1="tab20b")
    graph_label_df['label'] = kmeans.labels_
    graph_label_df['graph_label'] = patch_label_list

    return patch_df, graph_label_df, Subgraph_whole_feature, embeddings, kmeans

def visualization(sampled_subgraph, sampled_patch, whole_feature, embedding, kmeans, graph_cluster_num, patch_cluster_num, savedir, label_list, root_dir, pt_rootdir, csv_rootdir, IG_type):
    sampled_patch['label'] = kmeans.labels_
    label_dict = {}
    for label in label_list:
        label_dir = os.path.join(savedir, str(label))
        if os.path.exists(label_dir) == False:
            os.mkdir(label_dir)
        edge_array = np.zeros((patch_cluster_num, patch_cluster_num))
        edge_array = calculate_label_edge_distribution(sampled_patch[sampled_patch['graph_label'] == label], label_dir,
                                                       edge_array, graph_cluster_num, patch_cluster_num, root_dir, pt_rootdir, csv_rootdir, IG_type)
        normalize = (edge_array - edge_array.min()) / (edge_array.max() - edge_array.min())
        label_dict[label] = normalize
        sns.heatmap(label_dict[label], cmap='plasma')
        plt.savefig(os.path.join(label_dir, 'edge_heatmap.pdf'), transparent=True)
        plt.savefig(os.path.join(label_dir, 'edge_heatmap.png'), transparent=True)
        plt.cla()
        plt.clf()
        plt.close()

def patch_visualization(patch_list, patch_cluster_num, patch_save_dir, root_dir, pt_rootdir, csv_rootdir, IG_type):

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

    patch_save_dir = os.path.join(patch_save_dir, str(patch_cluster_num))
    if os.path.exists(patch_save_dir)  == False:
        os.mkdir(patch_save_dir)

    for cluster in range(patch_cluster_num):
        patch_dir = os.path.join(patch_save_dir, 'patchcluster_' + str(cluster))
        if os.path.exists(patch_dir) == False:
            os.mkdir(patch_dir)

        cluster_patch = patch_list[patch_list['label'] == cluster]
        sampled_df = cluster_patch.sample(n=100, random_state=1)
        WSI_root = '/mnt/nvme1n1/CCRCC_WSI/'
        WSI_list = os.listdir(WSI_root)
        WSI_list = [os.path.join(WSI_root, item) for item in WSI_list]
        Dataset_root = '/mnt/nvme0n1/supernode_WSI/'
        Dataset_list = os.listdir(Dataset_root)
        Dataset_list = [os.path.join(Dataset_root, item) for item in Dataset_list]
        WSI_ID_link = pd.read_excel('/home/taliq_lee/DSA/WSI_CCRCC_Match.xlsx')
        for subgraph, patchnum in zip(list(sampled_df['column']), list(sampled_df['node_number'])):
            patient = subgraph.split('_')[0]
            wsi_id = ('_').join(subgraph.split('_')[0:5])
            subgraph_num = int(subgraph.split('_')[-1])
            WSI_name = list(WSI_ID_link[WSI_ID_link['path ID'] == patient]['deidentify'])
            WSI_name = [item for item in WSI_name if ('_').join(subgraph.split('_')[1:5]) in item]

            patient_item = root_dir + wsi_id + '_0/IG_again/' + wsi_id + '_0_' + IG_type +'_IG_TME_subgraph_small_cap_subgraph_exist_formal_setting_normal_filter.csv'
            WSI = osd.open_slide('/mnt/nvme1n1/CCRCC_WSI/' + WSI_name[0] + '.svs')
            WSI_level = 1
            WSI_Downsample_ratio = int(WSI.level_downsamples[WSI_level])
            Patch_dim = int(256 / WSI_Downsample_ratio)

            Position_pd = pd.read_csv(patient_item)
            Superpatch_pd = pd.read_csv(os.path.join(csv_rootdir, wsi_id + '_0_4.3_artifact_sophis_final.csv'))
            Location_pd = pd.read_csv(os.path.join(pt_rootdir, wsi_id + '_node_location_list.csv'))

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

def Parser_main():
    
    parser = argparse.ArgumentParser(description="TEA-graph superpatch generation")
    parser.add_argument("--rootdir", default="/mnt/sdb/supernode_WSI/IG_analysis_BORAME_final/2022-01-27_08:58:10/0/IG_analysis/", help="patient_IG_analysis_result_dir", type = str)
    parser.add_argument("--pt_dir",default='/mnt/sdb/supernode_WSI/0.75/' ,help="patient_graph_torch_dir",type=str)
    parser.add_argument("--csv_dir", default = '/mnt/sdb/supernode_WSI/0.75_env_new/', help = "superpatch patch list dir", type = str)
    parser.add_argument("--save_dir",default="/home/seob/DSA/figure_six/220202_figure_six_revision_",help="save prefix",type=str)
    parser.add_argument("--seed",default=1234567,help="random seed",type=int)
    parser.add_argument("--threshold", default = '10', help = "subgraph patch number threshold", type = str)
    parser.add_argument("--graph_cluster_num", default = 25, help ="graph_cluster_number", type = int)
    parser.add_argument("--patch_cluster_num", default = 10, help ="patch_cluster_number", type = int)
    parser.add_argument("--IG_cluster", default = "Top", help = "IG_clster_type(Top, Mid, Low)", type = str)
    return parser.parse_args()


def main():
    Argument = Parser_main()
    rootdir = Argument.rootdir
    pt_rootdir = Argument.pt_dir
    csv_dir = Argument.csv_dir
    threshold = Argument.threshold
    savedir = Argument.save_dir + threshold + '_final'
    seed_num = Argument.seed
    IG_type = Argument.IG_cluster
    if os.path.exists(savedir) == False:
        os.mkdir(savedir)
    graph_savedir = os.path.join(savedir, 'graph')
    if os.path.exists(graph_savedir) == False:
        os.mkdir(graph_savedir)
    patch_savedir = os.path.join(savedir, 'patch')
    if os.path.exists(patch_savedir) == False:
        os.mkdir(patch_savedir)
    

    seed(seed_num)
    graph_cluster_num = Argument.graph_cluster_num
    patch_cluster_num = Argument.patch_cluster_num
    IG_whole_feature, Patient_ID_list, IG_ID_list,IG_length_list, Surv, Event, Patient_item = cluster_analysis(rootdir, pt_rootdir, threshold,IG_type)
    
    graph_save_dir = os.path.join(graph_savedir, str(graph_cluster_num))
    if os.path.exists(graph_save_dir) == False:
        os.mkdir(graph_save_dir)
    subgraph_df, label_df, length_df, diff_df, pvalue_df = clustering(IG_whole_feature, Patient_ID_list, graph_cluster_num, IG_ID_list, IG_length_list, Surv, Event,IG_type, graph_save_dir)
    
    graph_cluster_label_list = range(graph_cluster_num)
    sampled_subgraph, sampled_patch, whole_feature, embedding, kmeans = patch_sampling(rootdir, patch_savedir, subgraph_df, label_df, patch_cluster_num, length_df, pt_rootdir, IG_type)
    patch_visualization(sampled_patch, patch_cluster_num, patch_savedir,rootdir, pt_rootdir, csv_dir)
    calculate_label_distribution(sampled_patch, graph_cluster_num, patch_cluster_num, graph_save_dir)
    visualization(sampled_subgraph, sampled_patch, whole_feature, embedding, kmeans, graph_cluster_num, patch_cluster_num, graph_save_dir, graph_cluster_label_list, rootdir, pt_rootdir, csv_dir, IG_type)

if __name__ == "__main__":
    main()
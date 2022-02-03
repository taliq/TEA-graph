import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
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
from scipy.signal import savgol_filter
import argparse

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')

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
    length_array = np.array(length_list)
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

    # label_subgraph_dict = subgraph_list per label
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
    label_csv = os.path.join(Figure_dir, 'label_list.csv')
    Label_count = np.zeros((len(Patient_ID_list), cluster_num))

    #Count the label per patient for survival analysis
    for c, item in enumerate(Patient_ID_list):
        ID_match = np.isin(ID_list, item)
        #ID_match = np.isin(High_ID_list, item)
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

def cluster_analysis(rootdir, pt_rootdir, threshold):
    threshold = int(threshold)
    CCRCC_dataset = pd.read_excel('/home/taliq_lee/DSA/1995-2008_CCRCC.xlsx')

    Patient_dir_list = os.listdir(rootdir)

    Patient_ID_list = [('_').join(item.split('_')[0:5]) for item in Patient_dir_list]
    Patient_dir_list = [os.path.join(rootdir, item) for item in Patient_dir_list]

    Surv = np.zeros(len(Patient_dir_list))
    Event = np.zeros(len(Patient_dir_list))

    High_IG_whole_feature = []
    Mid_IG_whole_feature = []
    Low_IG_whole_feature = []

    High_ID_list = []
    Mid_ID_list = []
    Low_ID_list = []

    High_length_list = []
    Mid_length_list = []
    Low_length_list = []

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
                    if 'Top' in IG_item:
                        High_ID_list.extend([Patient_ID_list[c]] * Subgraph_whole_list.shape[0])
                        High_length_list.extend(length_list)
                        if len(High_IG_whole_feature) == 0:
                            High_IG_whole_feature = Subgraph_whole_list
                        else:
                            High_IG_whole_feature = torch.cat((High_IG_whole_feature, Subgraph_whole_list), 0)

                    elif 'Mid' in IG_item:
                        Mid_ID_list.extend([Patient_ID_list[c]] * Subgraph_whole_list.shape[0])
                        Mid_length_list.extend(length_list)
                        if len(Mid_IG_whole_feature) == 0:
                            Mid_IG_whole_feature = Subgraph_whole_list
                        else:
                            Mid_IG_whole_feature = torch.cat((Mid_IG_whole_feature, Subgraph_whole_list), 0)
                    elif 'Low' in IG_item:
                        Low_ID_list.extend([Patient_ID_list[c]] * Subgraph_whole_list.shape[0])
                        Low_length_list.extend(length_list)
                        if len(Low_IG_whole_feature) == 0:
                            Low_IG_whole_feature = Subgraph_whole_list
                        else:
                            Low_IG_whole_feature = torch.cat((Low_IG_whole_feature, Subgraph_whole_list), 0)
            pbar.update()

    Scaler = StandardScaler()
    High_IG_whole_feature = Scaler.fit_transform(High_IG_whole_feature.detach().cpu().numpy())
    Mid_IG_whole_feature = Scaler.fit_transform(Mid_IG_whole_feature.detach().cpu().numpy())
    Low_IG_whole_feature = Scaler.fit_transform(Low_IG_whole_feature.detach().cpu().numpy())

    High_ID_list = np.array(High_ID_list)
    Mid_ID_list = np.array(Mid_ID_list)
    Low_ID_list = np.array(Low_ID_list)

    return High_IG_whole_feature, Mid_IG_whole_feature, Low_IG_whole_feature, Patient_ID_list, High_ID_list, Mid_ID_list, Low_ID_list, High_length_list,\
           Mid_length_list, Low_length_list, Surv, Event, Patient_item
           
def inter_intra_variance(final_df, cluster_num):

    intra_feature = np.array(final_df.iloc[:,3:])
    index_list = []

    for i in range(cluster_num):
        subdf = final_df[final_df['label'] == i]
        index_num = list(subdf.index)
        sampling = sample(index_num, 50)
        index_list.append(sampling)
    inter_list = []
    intra_list = []
    for i in range(cluster_num):
        index_1 = index_list[i]
        for j in range(cluster_num):
            diff_list = []
            index_2 = index_list[j]
            for node1 in index_1:
                for node2 in index_2:
                    feature1 = final_df.iloc[node1, 3:]
                    feature2 = final_df.iloc[node2, 3:]
                    difference = feature1 - feature2
                    diff_list.append(np.linalg.norm(difference))
            difference_mean = sum(diff_list) / len(diff_list)
            if i == j:
                inter_list.append(difference_mean)
            else:
                intra_list.append(difference_mean)

    inter_mean = sum(inter_list) / len(inter_list)
    intra_mean = sum(intra_list) / len(intra_list)
    final = inter_mean / intra_mean
    
    return final

def patch_sampling(rootdir, savedir, subgraph_df, label_df, patch_cluster_num_list, length_df, pt_rootdir):
    
    savedir = os.path.join(savedir,'patch_validation')
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

        Patient_item = [item for item in Patient_item if 'Top' in item]
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
    whole_feature_df = pd.DataFrame(Subgraph_whole_feature)

    graph_label_df = pd.DataFrame(patch_ID_list, columns=["column"])
    inter_variance_dict = {}
    for cluster_num in patch_cluster_num_list:
        graph_label_df_new = graph_label_df.copy()
        embeddings = run_tsne(Subgraph_whole_feature)
        kmeans = KMeans(n_clusters=cluster_num, random_state=0, verbose=1).fit(embeddings)
        Visualize_tsne(embeddings, str(cluster_num), cluster_num, savedir, kmeans.labels_, c1 = "tab20b")
        graph_label_df_new['label'] = kmeans.labels_
        graph_label_df_new['graph_label'] = patch_label_list
        final = pd.concat([graph_label_df_new, whole_feature_df], axis = 1)
        inter_list = inter_intra_variance(final, cluster_num)
        inter_variance_dict[cluster_num] = inter_list
    
    print(inter_variance_dict.values())
    plt.plot(inter_variance_dict.keys(), inter_variance_dict.values())
    plt.savefig(os.path.join(savedir, 'inter_variance.png'))
    plt.clf()
    plt.close()
    
    yhat = savgol_filter(inter_variance_dict.values(), 5, 3)
    plt.plot(patch_cluster_num_list, inter_variance_dict.values(), alpha = 0.3)
    plt.plot(patch_cluster_num_list ,yhat, color='green')
    plt.xticks(range(min(patch_cluster_num_list),max(patch_cluster_num_list)+1,2))
    plt.savefig(os.path.join(savedir, "inter_intra_validation.pdf"), transparent = True)
    plt.clf()

    
    return patch_df, graph_label_df, Subgraph_whole_feature, embeddings, kmeans

def graph_data_processing(high_diff_df, mid_diff_df, low_diff_df, high_pvalue_df, mid_pvalue_df, low_pvalue_df,
                          graph_cluster_diff_dict, graph_cluster_p_dict, graph_cluster_diff_sum_dict, graph_cluster_p_num_dict,
                          graph_cluster_p_num_dict_strict,graph_cluster_average_p_dict, graph_cluster_num, graph_savedir):
           
    for key in high_diff_df.keys():
        if high_diff_df[key] < 0:
            high_pvalue_df[key] = 1
        if low_diff_df[key] > 0:
            low_pvalue_df[key] = 1
    
    graph_cluster_diff_dict[graph_cluster_num] = [high_diff_df, mid_diff_df, low_diff_df]
    graph_cluster_p_dict[graph_cluster_num] = [high_pvalue_df, mid_pvalue_df, low_pvalue_df]
    graph_cluster_diff_sum_dict[graph_cluster_num] = [sum(high_diff_df.values())/graph_cluster_num, sum(mid_diff_df.values())/graph_cluster_num, sum(low_diff_df.values())/graph_cluster_num]
    graph_cluster_p_num_dict[graph_cluster_num] = [(high_pvalue_df < 0.05).sum(axis = 1).item(), (mid_pvalue_df < 0.05).sum(axis = 1).item(), (low_pvalue_df<0.05).sum(axis=1).item()]
    graph_cluster_p_num_dict_strict[graph_cluster_num] = [(high_pvalue_df < 0.01).sum(axis = 1).item(), (mid_pvalue_df < 0.01).sum(axis = 1).item(), (low_pvalue_df<0.01).sum(axis=1).item()]
    graph_cluster_average_p_dict[graph_cluster_num] = ((high_pvalue_df < 0.01).sum(axis = 1).item() + (low_pvalue_df<0.01).sum(axis=1).item()) / (2* graph_cluster_num)

    return graph_cluster_diff_dict, graph_cluster_p_dict, graph_cluster_diff_sum_dict, graph_cluster_p_num_dict, graph_cluster_p_num_dict_strict,graph_cluster_average_p_dict

def Parser_main():
    
    parser = argparse.ArgumentParser(description="TEA-graph superpatch generation")
    parser.add_argument("--rootdir", default="/mnt/sdb/supernode_WSI/IG_analysis_BORAME_final/2022-01-27_08:58:10/0/IG_analysis/", help="patient_IG_analysis_result_dir", type = str)
    parser.add_argument("--pt_dir",default='/mnt/sdb/supernode_WSI/0.75/' ,help="patient_graph_torch_dir",type=str)
    parser.add_argument("--save_dir",default="/home/seob/DSA/figure_six/220202_figure_six_revision_",help="save prefix",type=str)
    parser.add_argument("--seed",default=1234567,help="random seed",type=int)
    parser.add_argument("--imagesize", default = 256, help ="crop image size", type = int)
    parser.add_argument("--threshold", default = 0.75, help = "cosine similarity threshold", type = float)
    parser.add_argument("--spatial_threshold", default = 5.5, help = "spatial threshold", type = float)
    parser.add_argument("--gpu", default = '0' , help = "gpu device number", type = str)
    return parser.parse_args()

def main():
    Argument = Parser_main()
    rootdir = Argument.rootdir
    pt_rootdir = Argument.pt_dir
    seed_num = Argument.seed
    threshold_list = ['0', '10', '15', '20', '25', '30', '35', '40']
    patch_cluster_num_list = [3, 5, 7, 8, 10, 13, 15, 17, 20, 23, 25, 27, 30]
    graph_cluster_num_list = [10,12,15,18,20,23,25,28,30]
    
    graph_cluster_diff_dict = {}
    graph_cluster_p_dict = {}
    graph_cluster_diff_sum_dict = {}
    graph_cluster_p_num_dict = {}
    graph_cluster_p_num_dict_strict = {}
    graph_cluster_average_p_dict = {}
    
    with tqdm(total = len(threshold_list)) as pbar_threshold:
        for threshold in threshold_list:
            savedir = Argument.save_dir + threshold
            if os.path.exists(savedir) == False:
                os.mkdir(savedir)
            graph_savedir = os.path.join(savedir, 'graph')
            if os.path.exists(graph_savedir) == False:
                os.mkdir(graph_savedir)
            patch_savedir = os.path.join(savedir, 'patch')
            if os.path.exists(patch_savedir) == False:
                os.mkdir(patch_savedir)
            
            seed(seed_num)
            
            with tqdm(total = len(graph_cluster_num_list)) as pbar_graph:
                for graph_cluster_num in graph_cluster_num_list:
                    High_IG_whole_feature, Mid_IG_whole_feature, Low_IG_whole_feature, Patient_ID_list, High_ID_list, Mid_ID_list, Low_ID_list, \
                    High_length_list, Mid_length_list, Low_length_list, Surv, Event, Patient_item = cluster_analysis(rootdir, pt_rootdir, threshold)
                    
                    graph_save_dir = os.path.join(graph_savedir, str(graph_cluster_num))
                    if os.path.exists(graph_save_dir) == False:
                        os.mkdir(graph_save_dir)
                        
                    high_subgraph_df, high_label_df, high_length_df, high_diff_df, high_pvalue_df = clustering(High_IG_whole_feature, Patient_ID_list, graph_cluster_num, High_ID_list, High_length_list, Surv, Event,'High', graph_save_dir)
                    mid_subgraph_df, mid_label_df, mid_length_df, mid_diff_df, mid_pvalue_df = clustering(Mid_IG_whole_feature, Patient_ID_list, graph_cluster_num, Mid_ID_list, Mid_length_list, Surv, Event,'Mid', graph_save_dir)
                    low_subgraph_df, low_label_df, low_length_df, low_diff_df, low_pvalue_df = clustering(Low_IG_whole_feature, Patient_ID_list, graph_cluster_num, Low_ID_list, Low_length_list, Surv, Event,'Low', graph_save_dir)
                    
                    graph_cluster_diff_dict, graph_cluster_p_dict, graph_cluster_diff_sum_dict, 
                    graph_cluster_p_num_dict, graph_cluster_p_num_dict_strict,graph_cluster_average_p_dict = graph_data_processing(high_diff_df, mid_diff_df, low_diff_df, high_pvalue_df, mid_pvalue_df, low_pvalue_df,
                                                                                                                                  graph_cluster_diff_dict, graph_cluster_p_dict, graph_cluster_diff_sum_dict, graph_cluster_p_num_dict,
                                                                                                                                  graph_cluster_p_num_dict_strict, graph_cluster_average_p_dict,graph_cluster_num, graph_savedir)
            
                    patch_sampling(rootdir, patch_savedir, high_subgraph_df, high_label_df, patch_cluster_num_list, high_length_df,pt_rootdir)
                    patch_sampling(rootdir, patch_savedir, mid_subgraph_df, mid_label_df, patch_cluster_num_list, mid_length_df,pt_rootdir)
                    patch_sampling(rootdir, patch_savedir, low_subgraph_df, low_label_df, patch_cluster_num_list, low_length_df,pt_rootdir)
                    pbar_graph.update()
            
            graph_cluster_diff_sum_df = pd.DataFrame.from_dict(graph_cluster_diff_sum_dict)
            graph_cluster_diff_sum_df.to_csv(os.path.join(graph_savedir, 'graph_cluster_diff_sum_dict.csv'))
            graph_cluster_p_num_df = pd.DataFrame.from_dict(graph_cluster_p_num_dict)
            graph_cluster_p_num_df.to_csv(os.path.join(graph_savedir, 'graph_cluster_p_num_dict.csv'))
            graph_cluster_p_num_strict_df = pd.DataFrame.from_dict(graph_cluster_p_num_dict_strict)
            graph_cluster_p_num_strict_df.to_csv(os.path.join(graph_savedir, 'graph_cluster_p_num_dict_strict.csv'))
            graph_cluster_average_p_df = pd.DataFrame.from_dict(graph_cluster_average_p_dict)
            graph_cluster_average_p_df.to_csv(os.path.join(graph_savedir, 'graph_cluster_p_avereage_dict.csv'))
            
            plt.plot(graph_cluster_num_list, graph_cluster_average_p_dict)
            tick = range(min(graph_cluster_num_list) , max(graph_cluster_num_list)+1)
            plt.xticks(tick)
            plt.savefig("/home/seob/DSA/for_revision/p_strict_avg_new.pdf", transparent = True)
            plt.clf()

            pbar_threshold.update()
    
if __name__ == "__main__":
    main()

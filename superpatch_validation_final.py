#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:11:59 2021

@author: kyungsub
"""

import torch
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from math import log2
import os
import numpy as np
from tqdm import tqdm
import statistics

def kl_divergence(p, q):
    kl_list = []
    for i in q.keys():
        if i not in p.keys():
            kl_list.append(0)
        else:
            kl_list.append(p[i] * log2(p[i]/q[i]))
                
    return sum(kl_list)

def entropy(p):
    return -sum(p[i] * log2(p[i]) for i in range(len(p)))

def NMI(matrix, cluster_num = 10):
    denom = 0
    numerator = 0
    N = int(sum(sum(matrix)))
    for i in range(cluster_num):
        for j in range(cluster_num):
            if matrix[i][j] == 0:
                numerator += 0
            else:
                numerator += -2 * matrix[i][j]*log2((matrix[i][j]*N)/(sum(matrix[i])*sum(matrix[:,j])))
    
    for i in range(cluster_num):
        denom += sum(matrix[i])*log2(sum(matrix[i])/N)
    for j in range(cluster_num):
        denom += sum(matrix[:,j])*log2(sum(matrix[:,j])/N)
    
    NMI = numerator/denom
    
    return NMI

def kld_calculation(feature1, feature2):
    feature1_len = feature1.shape[0]
    feature2_len = feature2.shape[0]

    cluster_num = 10
    
    final_feature = np.concatenate((feature1, feature2), axis = 0)
    
    km = KMeans(
        n_clusters=cluster_num, init='random',
        n_init=10, max_iter=300, 
        tol=1e-04, random_state=0
    )
    percent = []
    y_km = km.fit_predict(final_feature)
    feature_label1 = y_km[:feature1_len]
    feature_label2 = y_km[feature1_len:]
    feature_dist1 = {}
    feature_dist2 = {}
    label = [feature_label1, feature_label2]
    dist = [feature_dist1, feature_dist2]
    for i,label_list  in enumerate(label):
        unique, count = np.unique(label_list, return_counts = True)
        for index_num, j in enumerate(list(unique)):    
            dist[i][j] = list(count)[index_num] / sum(count)
    
    kl_ws = kl_divergence(dist[0], dist[1])
    
    return kl_ws

sample_list = os.listdir('/disk/sdd1/test/12345/')
sample_list = list(set([('_').join(sample.split('_')[0:5]) for sample in sample_list]))
metadata = pd.read_excel("/disk/sdd1/CCRCC_data.xlsx")
stage_mean_dict = {}
stage_var_dict = {}
stage_dict = {}

for i in range(4):
    stage_mean_dict.setdefault(i+1,[])
    
for i in range(4):
    stage_var_dict.setdefault(i+1, [])
for sample in sample_list:
    aid = sample
    sample = ('_').join(sample.split('_')[0:4]) + '_01'
    sample_meta = metadata[metadata['deidentify']==sample]
    stage = int(sample_meta['WHO/ISUP grade'].item())
    stage_dict[aid] = stage
    
final_corr_mean_list = []
final_corr_var_list = []
final_kl_list = []
diff_list = []

with tqdm(total = len(sample_list)) as pbar:
    for sample in sample_list:
        final_corr_mean_list = []
        final_corr_var_list = []
        final_kl_list = []
        stage = stage_dict[sample]
        
        loc_csv= pd.read_csv("/disk/sdd1/test/1234567/" + sample +'_node_location_list.csv')
        
        superpatch_file_list = ["/disk/sdd1/test/1234567/"+sample+"_new_0.csv",
                                "/disk/sdd1/test/123456/"+sample+"_new_0.csv",
                                "/disk/sdd1/test/12345/"+sample+"_new_0.csv",
                                "/disk/sdd1/test/1234/"+sample+"_new_0.csv",
                                "/disk/sdd1/test/123/"+sample+"_new_0.csv"]
        
        graph_file_list = ["/disk/sdd1/test/1234567/"+sample+"_new_0_graph_torch.pt",
                           "/disk/sdd1/test/123456/"+sample+"_new_0_graph_torch.pt",
                           "/disk/sdd1/test/12345/"+sample+"_new_0_graph_torch.pt",
                           "/disk/sdd1/test/1234/"+sample+"_new_0_graph_torch.pt",
                           "/disk/sdd1/test/123/"+sample+"_new_0_graph_torch.pt"]
        
        
        #superpatch_num_list = []
        superpatch_index_list = []
        superpatch_list = []
        feature_list = []
        
        for index in range(len(superpatch_file_list)):
            superpatch = pd.read_csv(superpatch_file_list[index])
            superpatch_list.append(superpatch)
            
            graph = torch.load(graph_file_list[index])
            feature_list.append(graph.x.detach().cpu().numpy())
            
            superpatch_num_perfile = []
            superpatch_index_perfile = []
            
            for i, row in superpatch.iterrows():
                index_num = int(row.iloc[0])
                X = loc_csv['X'][index_num] - min(loc_csv['X'])
                Y = loc_csv['Y'][index_num] - min(loc_csv['Y'])
                number = row.dropna().tolist()
                superpatch_num_perfile.append(len(number)-1)
                superpatch_index_perfile.append(number[0])
            
            superpatch_index_list.append(superpatch_index_perfile)
        
        kld_list = []
        for index in range(len(superpatch_file_list)):
            for subindex in range(index+1, len(superpatch_file_list)):
                kld_value = kld_calculation(feature_list[index], feature_list[subindex])
                kld_list.append(kld_value)
        
            
        #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        #plt.boxplot(kld_list, positions =[1])
        #fig.suptitle('additional_information')
        #plt.savefig('/disk/sdd1/test/fig/' + sample+ '_kl.pdf', transparent = True)
        final_kl_list.append(kld_list)
        
        final_corr_list = []
        final_corr_dict_list = {}
        
        for index in range(len(superpatch_file_list)-1):
            
            difference = set(superpatch_index_list[0]) - set(superpatch_index_list[index + 1])
            diff_list.append(len(difference)/len(superpatch_index_list[0]))
            orig_superpatch_list = superpatch_list[0]
            target_superpatch_list = superpatch_list[index+1]
            
            final_corr_list_persample = []
            final_corr_dict_persample = {}
            final_corr_mean_dict_persample = {}
            final_corr_var_dict_persample = {}
            
            for orig_node in difference:
                X = loc_csv['X'][orig_node]
                Y = loc_csv['Y'][orig_node]
                target_node_candidate = loc_csv[(X-2 <= loc_csv['X'])&(loc_csv['X'] <= X+2)&
                                               (Y-2 <= loc_csv['Y'])&(loc_csv['Y'] <= Y+2)]
                orig_superpatch_index = int(orig_superpatch_list[orig_superpatch_list['Unnamed: 0'] == orig_node].index.item())
                
                corr_list = []
                corr_dict = {}
                for i, row in target_node_candidate.iterrows():
                    target_node_cand = row['Unnamed: 0'].item()
                    target_row = target_superpatch_list[target_superpatch_list['Unnamed: 0'] == target_node_cand]
                    superpatch_cand_list = []
    
                    if len(target_row) != 0:
                        patch_list = target_row.iloc[0,].dropna().iloc[1:].tolist()
                        if orig_node in patch_list:
                            target_node = target_row.iloc[0,].dropna().iloc[0].item()
                            target_superpatch_index = int(target_row.index.item())
                            
                            orig_feature = feature_list[0][orig_superpatch_index]
                            target_feature = feature_list[index+1][target_superpatch_index]
                            corr = np.corrcoef(orig_feature, target_feature)[0][1]
                            corr_list.append(corr)
                            node_link = str(int(orig_node)) + '_' + str(int(target_node))
                            corr_dict[node_link] = corr
                        
                if len(corr_list) != 0 :
                    final_corr_list_persample.append(max(corr_list))
                    final_corr_dict_persample[max(corr_dict)] = corr_dict[max(corr_dict)]
                    final_corr_mean_dict_persample[int(orig_node)] = np.mean(corr_list)
                    final_corr_var_dict_persample[int(orig_node)] = np.var(corr_list)
                
            mean_df = pd.DataFrame(final_corr_mean_dict_persample, index = [0]).transpose()
            var_df = pd.DataFrame(final_corr_var_dict_persample, index = [0]).transpose()
            mean_df.to_csv('/disk/sdd1/test/fig/' + sample + '_' + str(index + 1) + '_mean_dictionary.csv')
            var_df.to_csv('/disk/sdd1/test/fig/' + sample + '_' + str(index+1) + '_var_dictionary.csv')
            
            #plt.boxplot(final_corr_mean_dict_persample.values())
            #plt.savefig('/disk/sdd1/test/fig/' + sample + '_' + str(index+1) + '_mean_boxplot.pdf', transparent = True)
            #plt.cla()
            #plt.clf()
            
            final_corr_mean_list.append(final_corr_mean_dict_persample.values())
            
            #plt.boxplot(final_corr_var_dict_persample.values())
            #plt.savefig('/disk/sdd1/test/fig/' + sample + '_' + str(index+1) + '_var_boxplot.pdf' , transparent = True)
            #plt.cla()
            #plt.clf()
            
            final_corr_var_list.append(final_corr_var_dict_persample.values())
            
            final_corr_list.append(final_corr_list_persample)
            final_corr_dict_list[str(sample) + '_' + str(index)] = final_corr_dict_persample
            df = pd.DataFrame.from_dict(final_corr_dict_list)
            df.to_csv("/disk/sdd1/test/fig/" + sample + '_' + str(index) + '_dictionary.csv')
        
        for mean_list in final_corr_mean_list:
            median = statistics.median(mean_list)
            stage_mean_dict[stage].append(median)
        for var_list in final_corr_var_list:
            median = statistics.median(var_list)
            stage_var_dict[stage].append(median)
        
        pbar.update()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    fig.suptitle('Superpatch randomization')
    
    ax[0].boxplot(stage_mean_dict[1], positions = [1])
    ax[0].boxplot(stage_mean_dict[2], positions = [2])
    ax[0].boxplot(stage_mean_dict[3], positions = [3])
    ax[0].boxplot(stage_mean_dict[4], positions = [4])
    ax[0].set_title('Correlation mean')
    
    ax[1].boxplot(stage_var_dict[1], positions = [1])
    ax[1].boxplot(stage_var_dict[2], positions = [2])
    ax[1].boxplot(stage_var_dict[3], positions = [3])
    ax[1].boxplot(stage_var_dict[4], positions = [4])
    ax[1].set_title('Corrleation variance')
    plt.savefig('/disk/sdd1/test/fig/final_variance.pdf', transparent = True)
    plt.cla()
    plt.clf()
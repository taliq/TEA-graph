#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:51:17 2021

@author: kyungsub
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from math import log2
import os 
import pandas as pd
import math
from tqdm import tqdm
import pickle



def kl_divergence(p, q):
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

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

def calculate_stat(sample):
    
    sample_dir = os.path.join(samples_dir, sample)
    match_patient = matching_excel[matching_excel['deidentify'] == sample]
    file_list = os.listdir(sample_dir)
    feature_list = [file for file in file_list if 'feature.npy' in file]
    feature_dict = {}
    for feature in feature_list:
        feature_name = feature.split('_')
        if len(feature_name) == 2:
            feature_name = feature_name[0]
        else:
            feature_name = ('_').join( feature_name[0:2])
    
        feature_dict[feature_name] = np.load(os.path.join(sample_dir, feature))
    
    len_dict = {}
    for key in feature_dict.keys():
        len_dict[key] = len(feature_dict[key])

    cluster_num = 10
    
    supernode_dict = {}
    csv_list = [file for file in file_list if ('.csv' in file) & ('node_location' not in file)]
    for file in csv_list:
        file_name = file.split('_')
        if len(file_name) == 7:
            number = file_name[6].split('.')[0]
            typing = file_name[5]
            key = number + '_' + typing
            supernode_dict[key] = pd.read_csv(os.path.join(sample_dir, file))
        else:
            key = file_name[-1].split('.')[0]
            supernode_dict[key] = pd.read_csv(os.path.join(sample_dir, file))
    
    compress_dict = {}
    for key in len_dict.keys():
        compress_dict[key] = len_dict[key] / len_dict['whole']
        
    whole_dict = {}
    for key in len_dict.keys():
        whole_dict[key] = np.concatenate((feature_dict['whole'], feature_dict[key]), axis = 0)

    km = KMeans(
        n_clusters=cluster_num, init='random',
        n_init=10, max_iter=300, 
        tol=1e-04, random_state=0
    )
    accuracy_dict = {}
    diff_dict = {}
    matrix_dict = {}
    #diff_proportion_list =[]
    for key in whole_dict.keys():
        if 'whole' not in key:
            total_feature = whole_dict[key]
            y_km = km.fit_predict(total_feature)
            whole_label = y_km[:len_dict['whole']]
            super_label = y_km[len_dict['whole']:]
    
            whole_dist = {}
            super_dist = {}
            label = [whole_label, super_label]
            dist = [whole_dist, super_dist]
            for i,label_list  in enumerate(label):
                unique, count = np.unique(label_list, return_counts = True)
                dist[i] = count / sum(count)
            
            #kl_ws = kl_divergence(dist[0], dist[1])
            #ent_w = entropy(dist[0])
            #percent[c].append(kl_ws/ent_w)
            
            matrix = np.zeros((cluster_num, cluster_num))
            label_acc = []
            diff_proportion_list = []
            for idx, row in supernode_dict[key].iterrows():
                row = row.tolist()
                row = [int(row[i]) for i in range(len(row)) if math.isnan(row[i]) is False]
                whole_patch_label_list = whole_label[row]
                superpatch_label = super_label[idx]
                unique, count = np.unique(whole_patch_label_list, return_counts = True)
                occurences = np.where(count == count.max())[0]
                if len(occurences) != 1:
                    if superpatch_label in unique:
                        label_acc.append(True)
                        wholepatch_label = superpatch_label
                    else:
                        label_acc.append(False)
                        wholepatch_label = unique[0]
                else:
                    wholepatch_label = unique[np.argmax(np.array(count))]
                    label_acc.append(superpatch_label == wholepatch_label)
                
                superpatch_label_count = np.where(unique == superpatch_label)[0]
                if len(superpatch_label_count) == 0:
                    superpatch_label_count = 0
                else:
                    superpatch_label_count = count[superpatch_label_count[0]]
                diff_proportion_list.append((sum(count)- superpatch_label_count)/sum(count))
                matrix[superpatch_label, wholepatch_label] += 1
                
            acc = sum(label_acc) / len(label_acc)
            difference = sum(diff_proportion_list)/len(diff_proportion_list)
            result = NMI(matrix, cluster_num)
            accuracy_dict[key] = acc
            matrix_dict[key] = result
            diff_dict[key] = difference
    
    return compress_dict, accuracy_dict, matrix_dict, diff_dict
        



samples_dir = "/disk/sdd1/test/supernode_validation/"
supernode_dir = "/disk/sdd1/supernode_WSI/"
WSI_match = "/disk/sdd1/WSI_CCRCC_Match.xlsx"
matching_excel = pd.read_excel(WSI_match)
sample_list = os.listdir(samples_dir)
sample_list = [sample for sample in sample_list if (sample != 'result')]
seed = 1234567
#keys = ['25', '25_freq', '25_length', '50', '50_freq', '50_length', '55', '55_freq', '55_length', '60', '60_freq', '60_length', '65', '65_freq', '65_length', '70', '70_freq', '70_length', '75', '75_freq', '75_length', 'random']
keys = ['random', '25_freq','25_length', '25', '50_freq','50_length',  '50', '55_freq','55_length', '55', '60_freq','60_length', '60', '65_freq','65_length', '65', '70_freq','70_length', '70', '75_freq','75_length', '75']
'''
acc_dict = {}
diff_dict = {}
comp_dict = {}
mat_dict = {}

dict_list = [acc_dict, diff_dict, comp_dict, mat_dict]

for item in dict_list:
    for i in keys:
        item.setdefault(i,[])
                
with tqdm(total = len(sample_list)) as pbar:
    for sample in sample_list:
        compress_dict, accuracy_dict, matrix_dict, difference_dict = calculate_stat(sample)
        for key in keys:
            acc_dict[key].append(accuracy_dict[key])
            diff_dict[key].append(difference_dict[key])
            mat_dict[key].append(matrix_dict[key])
            comp_dict[key].append(compress_dict[key])
        pbar.update()
'''
with open("/disk/sdd1/test/supernode_validation/acc_dict.pickle", 'rb') as f:
    acc_dict = pickle.load(f)
    
with open("/disk/sdd1/test/supernode_validation/mat_dict.pickle", 'rb') as f:
    mat_dict = pickle.load(f)
    
with open("/disk/sdd1/test/supernode_validation/diff_dict.pickle", 'rb') as f:
    diff_dict = pickle.load(f)
    
with open("/disk/sdd1/test/supernode_validation/comp_dict.pickle", 'rb') as f:
    comp_dict = pickle.load(f)
    

fig, ax = plt.subplots(figsize = (30, 10))
comp_data = [comp_dict[key] for key in keys]
ax.boxplot(comp_data)
ax.set_xticklabels(keys)
plt.savefig('/disk/sdd1/test/supernode_validation/result/comp_data_new.pdf', transparent = True)
plt.clf()

fig, ax = plt.subplots(figsize = (30, 10))
acc_data = [acc_dict[key] for key in keys]
ax.boxplot(acc_data)
ax.set_xticklabels(keys)

plt.savefig('/disk/sdd1/test/supernode_validation/result/acc_data_new.pdf', transparent = True)
plt.clf()


fig, ax = plt.subplots(figsize = (30, 10))
diff_data = [diff_dict[key] for key in keys]
ax.boxplot(diff_data)
ax.set_xticklabels(keys)
plt.savefig('/disk/sdd1/test/supernode_validation/result/diff_data_new.pdf', transparent = True)
plt.clf()


fig, ax = plt.subplots(figsize = (30, 10))
mat_data = [mat_dict[key] for key in keys]
ax.boxplot(mat_data)
ax.set_xticklabels(keys)
plt.savefig('/disk/sdd1/test/supernode_validation/result/mat_data_new.pdf', transparent = True)
plt.clf()

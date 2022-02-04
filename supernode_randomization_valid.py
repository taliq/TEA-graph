#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 20:47:33 2021

@author: kyungsub
"""

import numpy as np
import torch
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from math import log2
import os 
import math
from collections import Counter

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

loc_csv = pd.read_csv("/disk/sdb1/test/1234567/AA0328_2020_04_04_01_new_node_location_list.csv")

superpatch_file_list = ["/disk/sdb1/test/1234567/AA0328_2020_04_04_01_new_0.csv",
                        "/disk/sdb1/test/123456/AA0328_2020_04_04_01_new_0.csv",
                        "/disk/sdb1/test/12345/AA0328_2020_04_04_01_new_0.csv",
                        "/disk/sdb1/test/1234/AA0328_2020_04_04_01_new_0.csv",
                        "/disk/sdb1/test/123/AA0328_2020_04_04_01_new_0.csv"]

graph_file_list = ["/disk/sdb1/test/1234567/AA0328_2020_04_04_01_new_0_graph_torch.pt",
                   "/disk/sdb1/test/123456/AA0328_2020_04_04_01_new_0_graph_torch.pt",
                   "/disk/sdb1/test/12345/AA0328_2020_04_04_01_new_0_graph_torch.pt",
                   "/disk/sdb1/test/1234/AA0328_2020_04_04_01_new_0_graph_torch.pt",
                   "/disk/sdb1/test/123/AA0328_2020_04_04_01_new_0_graph_torch.pt"]

superpatch_node_list = []
total = []
superpatch_num = []
superpatch_index = []

for index in range(len(superpatch_file_list)):
    superpatch = pd.read_csv(superpatch_file_list[index])
    superpatch_num_perfile = []
    superpatch_index_perfile = []
    for index, row in superpatch.iterrows():
        number = row.dropna().tolist()
        superpatch_num_perfile.append(len(number)-1)
        superpatch_index_perfile.append(number[0])
    superpatch_num.append(superpatch_num_perfile)
    superpatch_index.append(superpatch_index_perfile)

difference_list = []
for index in range(len(superpatch_file_list)-1):
    difference = set(superpatch_index[0]) - set(superpatch_index[index + 1])
    difference_list.append(difference)
    
for index in range(len(superpatch_file_list)):
    superpatch = pd.read_csv(superpatch_file_list[index])
    graph = torch.load(graph_file_list[index])
    superpatch_node_list.append(superpatch)
    if len(total) == 0:
        total = graph.x
    else:
        total = torch.cat((total, graph.x), dim = 0)
        
total = total.detach().cpu().numpy()
     
cluster_num = 10
km = KMeans(
        n_clusters=cluster_num, init='random',
        n_init=10, max_iter=300, 
        tol=1e-04, random_state=0
    )    

y_km = km.fit_predict(total)
new_list = []
start = 0
for superpatch in superpatch_node_list:
    superpatch['label'] = y_km[start:start + len(superpatch)]
    start += len(superpatch)
    new_list.append(superpatch)
    
X_max = max(list(loc_csv['X']))
Y_max = max(list(loc_csv['Y']))
X_min = min(list(loc_csv['X']))
Y_min = min(list(loc_csv['Y']))
tile_num = 6
X_quartile = int((X_max - X_min)/ tile_num)
Y_quartile = int((Y_max - Y_min) / tile_num)

X_quatile = []
Y_quatile = []
for i in range(tile_num):
    X_quatile.append(X_quartile * i)
    Y_quatile.append(Y_quartile * i)  

X_quatile.append(X_max + 1)
Y_quatile.append(Y_max + 1)

new_loc = []

for superpatch in new_list:
    superpatch = superpatch[['Unnamed: 0', 'label']].copy()
    loc_csv_new = loc_csv.copy()
    loc_csv_new['label'] = 9999
    for idx, row in superpatch.iterrows():
        index = row['Unnamed: 0']
        label = row['label']
        
        loc_csv_new.loc[index, 'label'] = label
    new_loc.append(loc_csv_new)
    
final_label_list = []
final_feature_list = []
final_loc_quartile = []
for loc_csv_new in new_loc:
    new_label_list = []
    feature_list = []
    loc_quartile_list = []
    start = 0
    for i in range(tile_num):
        for j in range(tile_num):
            loc_quartile = loc_csv_new[(X_quatile[i] <= loc_csv_new['X'])&(loc_csv_new['X'] < X_quatile[i+1])&
                                       (Y_quatile[j] <= loc_csv_new['Y'])&(loc_csv_new['Y'] < Y_quatile[j+1])]

            label_df = loc_quartile[loc_quartile['label'] != 9999 ]
            loc_quartile_list.append(label_df)
            feature = graph.x[start:start+len(label_df)].detach().cpu().numpy()
            start += len(label_df)
            label_list = list(label_df['label'])
            new_label_list.append(label_list)
            feature_list.append(feature)
            
    final_label_list.append(new_label_list)
    final_feature_list.append(feature_list)
    final_loc_quartile.append(loc_quartile_list)
    
final_count_list = []
for k in range(5):
    count_list = []
    for i in range(tile_num * tile_num):
        count = dict(Counter(final_label_list[k][i]))
        total = sum(count.values())
        for key in count.keys():
            count[key] = count[key]/ total
        count_list.append(count)
    final_count_list.append(count_list)

kl_divergence_list = []

for i in range(4):
    new_list = []
    for j in range(tile_num * tile_num):
        result = np.corrcoef(final_feature_list[0][j], final_feature_list[i][j])
        new_list.append(result)
    kl_divergence_list.append(new_list)
        

#plt.boxplot(kl_divergence_list)
#plt.show()
a = 1
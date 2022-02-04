#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:10:37 2021

@author: kyungsub
"""

import os
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch

import openslide as osd
from EfficientNet import EfficientNet
from torchvision import transforms
import random

from skimage.filters import threshold_multiotsu
import argparse

class SurvivalImageDataset():

    """
    Target dataset has the list of images such as
    _patientID_SurvDay_Censor_TumorStage_WSIPos.tif
    """

    def __init__(self, image, x, y, transform):

        self.image = image
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len((self.image))

    def __getitem__(self, idx):

        """
        patientID, SurvivalDuration, SurvivalCensor, Stage,
        ProgressionDuration, ProgressionCensor, MetaDuration, MetaCensor
        """
        transform = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
                ])
        #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        image = self.image[idx]
        x = self.x[idx]
        y = self.y[idx]
        image = image.convert('RGB')
        R = transform(image)

        sample = { 'image' : R,'X' : torch.tensor(x), 'Y' : torch.tensor(y) }
    
        return sample



def supernode_generation(image, model_ft, device, seed, save_dir, imagesize):

    transform = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
                ])
    sample = image.split('.')[0].split('/')[-1]
    print(sample)

    sample_dir = os.path.join(save_dir, sample)
    if os.path.exists(sample_dir) is False:
        os.mkdir(sample_dir)

    image_path = image
    slideimage = osd.OpenSlide(image_path)

    best_downsampling_level = 2

    # Get the image at the requested scale
    svs_native_levelimg = slideimage.read_region((0, 0), best_downsampling_level, slideimage.level_dimensions[best_downsampling_level])
    svs_native_levelimg = svs_native_levelimg.convert('L')
    img = np.array(svs_native_levelimg)
    
    thresholds = threshold_multiotsu(img)
    regions = np.digitize(img, bins=thresholds)
    regions[regions == 1] = 0
    regions[regions == 2] = 1
    thresh_otsu = regions

    Width = slideimage.dimensions[0]
    Height = slideimage.dimensions[1]
    num_row = int(Height/imagesize) + 1
    num_col = int(Width/imagesize) + 1
    x_list = []
    y_list = []
    feature_list = []
    x_y_list = []
    counter = 0
    inside_counter = 0
    temp_patch_list = []
    temp_x = []
    temp_y = []
    
    with tqdm(total = num_row * num_col) as pbar_image:
        for i in range(0, num_col):
            for j in range(0, num_row):
                
                if thresh_otsu.shape[1] >= (i+1)*16:
                    if thresh_otsu.shape[0] >= (j+1)*16:
                        cut_thresh = thresh_otsu[j*16:(j+1)*16, i*16:(i+1)*16]
                    else:
                        cut_thresh = thresh_otsu[(j)*16:thresh_otsu.shape[0], i*16:(i+1)*16]
                else:
                    if thresh_otsu.shape[0] >= (j+1)*16:
                        cut_thresh = thresh_otsu[j*16:(j+1)*16, (i)*16:thresh_otsu.shape[1]]
                    else:
                        cut_thresh = thresh_otsu[(j)*16:thresh_otsu.shape[0], (i)*16:thresh_otsu.shape[1]]
                         
                if np.mean(cut_thresh) > 0.75:
                    pbar_image.update()
                    pass
                else:

                    filter_location = (i*imagesize, j*imagesize)
                    level = 0
                    patch_size = (imagesize, imagesize)
                    location = (filter_location[0], filter_location[1])
                    
                    CutImage = slideimage.read_region(location, level, patch_size)
                    
                    temp_patch_list.append(CutImage)
                    x_list.append(i)
                    y_list.append(j)
                    temp_x.append(i)
                    temp_y.append(j)
                    counter += 1
                    batchsize = 64
                    
                    if counter == batchsize:
                    
                        Dataset = SurvivalImageDataset(temp_patch_list, temp_x, temp_y, transform)
                        dataloader = torch.utils.data.DataLoader(Dataset,batch_size=batchsize,num_workers=0,drop_last=False)
                        for sample_img in dataloader:
                            images = sample_img['image']
                            X = sample_img['X']
                            Y = sample_img['Y']
                            images = images.to(device)
                            with torch.set_grad_enabled(False):
                                 classifier, features = model_ft(images)
             
                        if inside_counter == 0:
                            feature_list = np.concatenate((features.cpu().detach().numpy(),
                                                           classifier.cpu().detach().numpy()), axis=1)
                            temp_x = np.reshape(np.array(temp_x), (len(temp_x),1))
                            temp_y = np.reshape(np.array(temp_y), (len(temp_x),1))
                            
                            x_y_list = np.concatenate((temp_x,temp_y),axis=1)
                        else:
                            feature_list = np.concatenate((feature_list, 
                                                           np.concatenate((features.cpu().detach().numpy(),
                                                                         classifier.cpu().detach().numpy()),axis=1)), axis=0)
                            temp_x = np.reshape(np.array(temp_x), (len(temp_x),1))
                            temp_y = np.reshape(np.array(temp_y), (len(temp_x),1))
                                         
                            x_y_list = np.concatenate((x_y_list, 
                                                       np.concatenate((temp_x,temp_y),axis=1)), axis=0)
                        inside_counter += 1
                        temp_patch_list = []
                        temp_x = []
                        temp_y = []
                        counter = 0
                        
                    pbar_image.update()
                    
        if counter < batchsize and counter >0:
            Dataset = SurvivalImageDataset(temp_patch_list, temp_x, temp_y, transform)
            dataloader = torch.utils.data.DataLoader(Dataset,batch_size=batchsize,num_workers=0,drop_last=False)
            for sample_img in dataloader:
                images = sample_img['image']
                X = sample_img['X']
                Y = sample_img['Y']
                images = images.to(device)
                with torch.set_grad_enabled(False):
                     classifier, features = model_ft(images)
 
                feature_list = np.concatenate((feature_list, 
                                               np.concatenate((features.cpu().detach().numpy(),
                                                             classifier.cpu().detach().numpy()),axis=1)), axis=0)
                temp_x = np.reshape(np.array(temp_x), (len(temp_x),1))
                temp_y = np.reshape(np.array(temp_y), (len(temp_x),1))
                             
                x_y_list = np.concatenate((x_y_list, 
                                           np.concatenate((temp_x,temp_y),axis=1)), axis=0)
            temp_patch_list = []
            temp_x = []
            temp_y = []
            counter = 0
            
    feature_df = pd.DataFrame.from_dict(feature_list)
    feature_arr = np.array(feature_list)
    coordinate_df = pd.DataFrame({'X': x_y_list[:,0],'Y': x_y_list[:,1]})
    graph_dataframe = pd.concat([coordinate_df, feature_df], axis = 1)
    graph_dataframe = graph_dataframe.sort_values(by = ['Y', 'X'])
    graph_dataframe = graph_dataframe.reset_index(drop = True)
    coordinate_df = graph_dataframe.iloc[:,0:2]
    coordinate_df.to_csv(os.path.join(sample_dir, sample+'_node_location_list.csv'))
    
    index = list(graph_dataframe.index)
    feature_arr = np.array(graph_dataframe.iloc[:,2:])
    np.save(os.path.join(sample_dir,sample + '_whole_feature.npy'), feature_arr)
    graph_dataframe.insert(0,'index_orig', index)
    
    node_dict_25 = {}
    node_dict_50 = {}
    node_dict_55 = {}
    node_dict_60 = {}
    node_dict_65 = {}
    node_dict_70 = {}
    node_dict_75 = {}
    
    node_dict_random_all = {}
    
    node_dict_random_only_25 = {}
    node_dict_random_only_50 = {}
    node_dict_random_only_55 = {}
    node_dict_random_only_60 = {}
    node_dict_random_only_65 = {}
    node_dict_random_only_70 = {}
    node_dict_random_only_75 = {}
    
    node_dict_random_freq_25 = {}
    node_dict_random_freq_50 = {}
    node_dict_random_freq_55 = {}
    node_dict_random_freq_60 = {}
    node_dict_random_freq_65 = {}
    node_dict_random_freq_70 = {}
    node_dict_random_freq_75 = {}

    node_dict = [node_dict_25, node_dict_50, node_dict_55, node_dict_60, node_dict_65, node_dict_70, node_dict_75]
    node_dict_random_only = [node_dict_random_only_25, node_dict_random_only_50, node_dict_random_only_55, node_dict_random_only_60,
                             node_dict_random_only_65, node_dict_random_only_70, node_dict_random_only_75]
    
    node_dict_random_freq = [node_dict_random_freq_25, node_dict_random_freq_50, node_dict_random_freq_55, node_dict_random_freq_60,
                             node_dict_random_freq_65, node_dict_random_freq_70, node_dict_random_freq_75]
    
    node_dict_total = [node_dict, [node_dict_random_all], node_dict_random_only, node_dict_random_freq]
    
    for node_dict_list in node_dict_total:
        for item in node_dict_list:
            for i in range(len(coordinate_df)):
                item.setdefault(i,[])
        
    X = max(set(np.squeeze(graph_dataframe.loc[:, ['X']].values,axis = 1)))
    Y = max(set(np.squeeze(graph_dataframe.loc[:, ['Y']].values, axis = 1)))
    
    gridNum = 4
    X_size = int(X / gridNum)
    Y_size = int(Y / gridNum)
            
    with tqdm(total=(gridNum+2)*(gridNum+2)) as pbar:
        for p in range(gridNum+2):
            for q in range(gridNum+2):
                if p == 0 :
                    if q == 0:
                        is_X = graph_dataframe['X'] <= X_size * (p+1)
                        is_X2 = graph_dataframe['X'] >= 0
                        is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                        is_Y2 = graph_dataframe['Y'] >= 0
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        
                    elif q == (gridNum+1):
                        is_X = graph_dataframe['X'] <= X_size * (p+1)
                        is_X2 = graph_dataframe['X'] >= 0
                        is_Y = graph_dataframe['Y'] <= Y
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        
                    else:
                        is_X = graph_dataframe['X'] <= X_size * (p+1)
                        is_X2 = graph_dataframe['X'] >= 0
                        is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                elif p == (gridNum+1) :
                    if q == 0:
                        is_X = graph_dataframe['X'] <= X
                        is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                        is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                        is_Y2 = graph_dataframe['Y'] >= 0
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    elif q == (gridNum+1):
                        is_X = graph_dataframe['X'] <= X
                        is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                        is_Y = graph_dataframe['Y'] <= Y
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    else:
                        is_X = graph_dataframe['X'] <= X
                        is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                        is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                else :
                    if q == 0:
                        is_X = graph_dataframe['X'] <= X_size * (p+1)
                        is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                        is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                        is_Y2 = graph_dataframe['Y'] >= 0
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    elif q == (gridNum+1):
                        is_X = graph_dataframe['X'] <= X_size * (p+1)
                        is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                        is_Y = graph_dataframe['Y'] <= Y
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    else:
                        is_X = graph_dataframe['X'] <= X_size * (p+1)
                        is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                        is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                
                if len(X_10) == 0:
                    pbar.update()
                    continue
                
                coordinate_dataframe = X_10.loc[:, ['X','Y']]
                X_10 = X_10.reset_index(drop = True)
                coordinate_list = coordinate_dataframe.values.tolist()
                index_list = coordinate_dataframe.index.tolist()
                
                feature_dataframe = X_10[X_10.columns.difference(['index_orig','X','Y'])]
                feature_list = feature_dataframe.values.tolist()
                coordinate_matrix = euclidean_distances(coordinate_list, coordinate_list)
                coordinate_matrix = np.where(coordinate_matrix > 2.9, 0 , 1)
                cosine_matrix = cosine_similarity(feature_list, feature_list)
                
                Adj_list25= (coordinate_matrix == 1).astype(int) * (cosine_matrix >= 0.25).astype(int)
                Adj_list50 = (coordinate_matrix == 1).astype(int) * (cosine_matrix >= 0.50).astype(int)
                Adj_list55 = (coordinate_matrix == 1).astype(int) * (cosine_matrix >= 0.55).astype(int)
                Adj_list60 = (coordinate_matrix == 1).astype(int) * (cosine_matrix >= 0.60).astype(int)
                Adj_list65 = (coordinate_matrix == 1).astype(int) * (cosine_matrix >= 0.65).astype(int)
                Adj_list70 = (coordinate_matrix == 1).astype(int) * (cosine_matrix >= 0.70).astype(int)
                Adj_list75 = (coordinate_matrix == 1).astype(int) * (cosine_matrix >= 0.75).astype(int)
                
                Adj_list = [Adj_list25, Adj_list50, Adj_list55, Adj_list60, Adj_list65, Adj_list70, Adj_list75]
                
                
                Adj_list_for_random = (coordinate_matrix == 1).astype(int)
                
                np.random.seed(seed)
                len_dict_25 = {}
                len_dict_50 = {}
                len_dict_55 = {}
                len_dict_60 = {}
                len_dict_65 = {}
                len_dict_70 = {}
                len_dict_75 = {}
                len_dict = [len_dict_25, len_dict_50, len_dict_55, len_dict_60, len_dict_65, len_dict_70, len_dict_75]
                
                index_len_dict_25 = {}
                index_len_dict_50 = {}
                index_len_dict_55 = {}
                index_len_dict_60 = {}
                index_len_dict_65 = {}
                index_len_dict_70 = {}
                index_len_dict_75 = {}
                index_len_dict = [index_len_dict_25, index_len_dict_50, index_len_dict_55, index_len_dict_60, index_len_dict_65, index_len_dict_70, index_len_dict_75]
                
                for c, item in enumerate(len_dict):
                    for i in range(25):
                        len_dict[c][i] = 0 
                    
                for c, item in enumerate(Adj_list):
                    for index, node_item in enumerate(item):
                        node_list = np.array(index_list)[node_item.astype('bool')]
                        length = len(node_list) -1
                        len_dict[c][length] += 1
                        index_len_dict[c][index_list[index]] = length
                        for node_index in node_list:
                            if node_index == index_list[index]:
                                pass
                            else:
                                node_dict[c][index_list[index]].append(node_index)
                
                for c, item in enumerate(len_dict):
                    total = sum(item.values())
                    for key in item.keys():
                        item[key] /= total
                    len_dict[c] = list(item.values())
                    
                for c, item in enumerate(Adj_list_for_random):
                    node_list = np.array(index_list)[item.astype('bool')]
                    node_list = node_list[np.random.choice(len(node_list), np.random.choice(len(node_list)))]
                    for node_index in node_list:
                        if node_index == index_list[c]:
                            pass
                        else:
                            node_dict_random_all[index_list[c]].append(node_index)
                
                for c, item in enumerate(len_dict):
                    for index,node_item in enumerate(Adj_list_for_random):
                        node_list = np.array(index_list)[node_item.astype('bool')]
                        frequency = item[0:len(node_list)]
                        frequency = [frequent/sum(frequency) for frequent in frequency]
                        node_list_freq = node_list[np.random.choice(len(node_list), np.random.choice(len(node_list), p = frequency))]
                        if index_len_dict[c][index_list[index]] == 0:
                            node_list_length = []
                        else:
                            node_list_length = node_list[np.random.choice(len(node_list), np.random.choice(index_len_dict[c][index_list[index]]))] 
                    
                        for node_index in node_list_freq:
                            if node_index == index_list[index]:
                                pass
                            else:
                                node_dict_random_freq[c][index_list[index]].append(node_index)
                        
                        for node_index in node_list_length:
                            if node_index == index_list[index]:
                                pass
                            else:
                                node_dict_random_only[c][index_list[index]].append(node_index)
                pbar.update()
                
    
    arglist = []
    for node_dict_list in node_dict_total:
        arglist_temp = []
        for c, item in enumerate(node_dict_list):
            new_list = []
            for i in range(0, len(item)):
                new_list.append(len(list(set(item[i]))))
            arglist_new = np.argsort(np.array(new_list))
            arglist_new = arglist_new[::-1]
            arglist_temp.append(arglist_new.copy())
        arglist.append(arglist_temp.copy())
    
    for i, arglist_temp in enumerate(arglist):
        for c, item in enumerate(arglist_temp):
            for arg_value in item:
                if arg_value in node_dict_total[i][c].keys():
                    for adj_item in node_dict_total[i][c][arg_value]:
                        if adj_item in node_dict_total[i][c].keys():
                            node_dict_total[i][c].pop(adj_item, None)
                            item=np.delete(item, np.argwhere(item == adj_item))
        
    for node_dict_list in node_dict_total:
        for item in node_dict_list:
            for key_value in item.keys():
                item[key_value] = list(set(item[key_value]))

    supernode_relate_value_total = []
    whole_feature = graph_dataframe[graph_dataframe.columns.difference(['index_orig','X','Y'])]
    for node_dict_list in node_dict_total:
        supernode_relate_value_list = []
        for item in node_dict_list:
            supernode_coordinate_x = []
            supernode_coordinate_y = []
            supernode_feature = []
            supernode_relate_value_temp = [supernode_coordinate_x, supernode_coordinate_y, supernode_feature]
            for key_value in item.keys():
                supernode_relate_value_temp[0].append(graph_dataframe['X'][key_value])
                supernode_relate_value_temp[1].append(graph_dataframe['Y'][key_value])
                if len(item[key_value]) == 0:
                    select_feature = whole_feature.iloc[key_value]
                else:
                    select_feature = whole_feature.iloc[item[key_value] + [key_value]]
                    select_feature = select_feature.mean()

                if len(supernode_relate_value_temp[2]) == 0:
                    temp_select = np.array(select_feature)
                    supernode_relate_value_temp[2] = np.reshape(temp_select, (1,1794))
                else:
                    temp_select = np.array(select_feature)
                    supernode_relate_value_temp[2] = np.concatenate((supernode_relate_value_temp[2], np.reshape(temp_select, (1,1794))), axis=0)
            supernode_relate_value_list.append(supernode_relate_value_temp.copy())
        supernode_relate_value_total.append(supernode_relate_value_list.copy())
        pbar.update()
            
    
    final_threshold = ['25', '50', '55','60', '65', '70', '75']
    for c, supernode_relate_value in enumerate(supernode_relate_value_total[0]):
        np.save(os.path.join(sample_dir, sample + '_' + final_threshold[c] + '_feature.npy'), supernode_relate_value[2])
    
    np.save(os.path.join(sample_dir, sample + '_' + 'random_feature.npy'), supernode_relate_value_total[1][0][2])
    
    for c, supernode_relate_value in enumerate(supernode_relate_value_total[2]):
        np.save(os.path.join(sample_dir, sample + '_' + final_threshold[c] + '_length_random_feature.npy') , supernode_relate_value[2])
    
    for c, supernode_relate_value in enumerate(supernode_relate_value_total[3]):
        np.save(os.path.join(sample_dir, sample + '_' + final_threshold[c] + '_freq_random_feature.npy') , supernode_relate_value[2])

    for c, node_dict_temp in enumerate(node_dict_total[0]):
        node_dict_temp = pd.DataFrame.from_dict(node_dict_temp, orient = 'index')
        node_dict_temp.to_csv(os.path.join(sample_dir, sample + '_' + sample + '_' + final_threshold[c] + '.csv'))
    
    node_dict_temp = pd.DataFrame.from_dict(node_dict_total[1][0], orient= 'index')
    node_dict_temp.to_csv(os.path.join(sample_dir, sample + '_' + sample + '_random.csv'))
    
    for c, node_dict_temp in enumerate(node_dict_total[2]):
        node_dict_temp = pd.DataFrame.from_dict(node_dict_temp, orient = 'index')
        node_dict_temp.to_csv(os.path.join(sample_dir, sample + '_' + sample + '_length_' + final_threshold[c] + '.csv'))
    
    for c, node_dict_temp in enumerate(node_dict_total[3]):
        node_dict_temp = pd.DataFrame.from_dict(node_dict_temp, orient = 'index')
        node_dict_temp.to_csv(os.path.join(sample_dir, sample + '_' + sample + '_freq_' + final_threshold[c] + '.csv'))


def Parser_main():
    parser = argparse.ArgumentParser(description="TEA-graph superpatch_random_concatenation")
    parser.add_argument("--database", default='BORAMAE', help="Use in the savedir", type=str)
    parser.add_argument("--cancertype", default='CCRCC', help="cancer type", type=str)
    parser.add_argument("--graphdir", default="/home/seob/DSA/code_validation/graph_dir/", help="graph save dir",type=str)
    parser.add_argument("--imagedir", default="/home/seob/DSA/code_validation/svs_dir/", help="svs file location",type=str)
    parser.add_argument("--savetype", default="random_number")
    parser.add_argument("--seed", default=12345, help="randomize seed", type=int)
    parser.add_argument("--weight_path", default=None, help="pretrained weight path", type=str)
    parser.add_argument("--imagesize", default=256, help="crop image size", type=int)
    parser.add_argument("--threshold", default=0.75, help="cosine similarity threshold", type=float)
    parser.add_argument("--spatial_threshold", default=5.5, help="spatial threshold", type=float)
    parser.add_argument("--gpu", default='0', help="gpu device number", type=str)
    return parser.parse_args()


def main():
    Argument = Parser_main()
    image_dir = Argument.imagedir
    save_dir = Argument.graphdir
    database = Argument.database
    cancer_type = Argument.cancertype
    savetype = Argument.savetype
    gpu = Argument.gpu
    seed = Argument.seed
    image_size = Argument.imagesize
    threshold = Argument.threshold
    spatial_threshold = Argument.spatial_threshold

    image_list = os.listdir(image_dir)
    image_list = [os.path.join(image_dir, image) for image in image_list]
    image_list.sort(key=lambda f: os.stat(f).st_size, reverse=False)
    random.seed(seed)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    save_dir = os.path.join(save_dir, database)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    save_dir = os.path.join(save_dir, cancer_type)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    save_dir = os.path.join(save_dir, savetype)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    device = torch.device(int(gpu) if torch.cuda.is_available() else "cpu")
    model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    if Argument.weight_path is not None:
        weight_path = Argument.weight_path
        load_weight = torch.load(weight_path, map_location=device)
        model_ft.load_state_dict(load_weight)

    model_ft = model_ft.to(device)
    model_ft.eval()

    with tqdm(total=len(image_list)) as pbar_tot:
        for image in image_list:
            supernode_generation(image, model_ft, device, seed, save_dir, image_size)
            pbar_tot.update()

if __name__ == "__main__":
    main()
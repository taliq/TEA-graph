#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:04:37 2020

@author: taliq
"""

import random
import os

import torch
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
import sys
import torch.nn.functional as F

from torch import optim
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import DataListLoader

from torch_geometric.nn import DataParallel
from torch_geometric.utils import dropout_adj
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from models import GAT
from utils import train_test_split
from utils import makecheckpoint_dir_graph as mcd
from utils import TrainValid_path
from utils import non_decay_filter
from utils import coxph_loss
from utils import cox_sort
from utils import accuracytest


class CoxGraphDataset(Dataset):

    def __init__(self, filelist, survlist, stagelist, censorlist, Metadata, mode, model,
                 transform=None, pre_transform=None):
        super(CoxGraphDataset, self).__init__()
        self.filelist = filelist
        self.survlist = survlist
        self.stagelist = stagelist
        self.censorlist = censorlist
        self.Metadata = Metadata
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

        data_re = Data(x=data_origin.x[:, :1792], edge_index=data_origin.edge_index)
        data = transfer(data_re)
        data.survival = torch.tensor(survival)
        data.phase = torch.tensor(phase)
        data.mets_class = torch.tensor(mets_class)
        data.stage = torch.tensor(stage)
        data.item = item
        data.edge_attr = data_origin.edge_attr
        data.pos = data_origin.pos

        return data


def Analyze(Argument, best_model, checkpoint_dir, Figure_dir, bestepoch, best_select=True):

    batch_num = int(Argument.batch_size)
    # batch_num = Argument.batch_size
    device = torch.device(int(Argument.gpu))
    Metadata = pd.read_csv('../Sample_data_for_demo/Metadata/KIRC_clinical.tsv', sep='\t')

    TrainRoot = TrainValid_path(Argument.DatasetType)
    Trainlist = os.listdir(TrainRoot)
    Trainlist = [item for c, item in enumerate(Trainlist) if '0_graph_torch_4.3_artifact_sophis_final.pt' in item]
    Trainlist = Trainlist[:100]
    Fi = Argument.FF_number

    Test_set = train_test_split(Trainlist, Metadata, Argument.DatasetType, TrainRoot, Fi, Analyze_flag=True)

    TestDataset = CoxGraphDataset(filelist=Test_set[0], survlist=Test_set[1],
                                  stagelist=Test_set[3], censorlist=Test_set[2],
                                  Metadata=Metadata, mode=Argument.DatasetType,
                                  model=Argument.model)
    test_loader = DataListLoader(TestDataset, batch_size=batch_num, shuffle=True, num_workers=8, pin_memory=True,
                                 drop_last=False)

    model = best_model
    if best_select == "Y":
        epoch = bestepoch
    else:
        weightpathlist = os.listdir(checkpoint_dir)
        weightpath = 0
        for item in weightpathlist:
            if ('epoch-' + str(best_select) + ',') in item:
                weightpath = item
        if weightpath == 0:
            return print("Can't find the epoch weight")
        epoch = best_select

        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, weightpath)), map_location="cuda:0")

    # model = DataParallel(model, device_ids=[0, 1], output_device=0)
    model = model.to(device)
    Cox_loss = coxph_loss()
    Cox_loss = Cox_loss.to(device)

    EpochSurv = []
    EpochPhase = []
    EpochRisk = []
    EpochFeature = []
    EpochID = []
    EpochStage = []

    Epochloss = 0
    batchcounter = 1

    model.eval()
    count = 0

    with tqdm(total=len(test_loader)) as pbar:
        with torch.set_grad_enabled(False):
            for c, d in enumerate(test_loader, 1):
                tempsurvival = torch.tensor([data.survival for data in d])
                tempphase = torch.tensor([data.phase for data in d])
                tempID = np.asarray([data.item for data in d])
                tempstage = torch.tensor([data.stage for data in d])
                tempmeta = torch.tensor([data.mets_class for data in d])

                out = model(d)
                out = F.normalize(out, p=2, dim=0)

                # final_updated_feature_list = updated_feature_list
                sort_idx = torch.argsort(tempsurvival, descending=True)

                risklist = out[sort_idx]
                tempsurvival = tempsurvival[sort_idx]
                tempphase = tempphase[sort_idx]
                for idx in sort_idx.cpu().detach().tolist():
                    EpochID.append(tempID[idx])
                tempstage = tempstage[sort_idx]
                tempmeta = tempmeta.to(out.device)

                risklist = risklist.to(out.device)
                tempsurvival = tempsurvival.to(out.device)
                tempphase = tempphase.to(out.device)

                if len(risklist) == 0:
                    temp = 0

                for riskval, survivalval, phaseval, stageval in zip(risklist, tempsurvival, tempphase, tempstage):
                    count = count + 1
                    EpochSurv.append(survivalval.cpu().detach().item())
                    EpochPhase.append(phaseval.cpu().detach().item())
                    if Argument.DatasetType == "BORAME_Meta":
                        EpochRisk.append(riskval[0].cpu().detach().item())
                    else:
                        EpochRisk.append(riskval.cpu().detach().item())
                    EpochStage.append(stageval.cpu().detach().item())
                    # print(count)

                batchcounter += 1
                if Argument.DatasetType == "BORAME_Meta":
                    Batchacc = accuracytest(tempsurvival, risklist[:, 0], tempphase)
                else:
                    Batchacc = accuracytest(tempsurvival, risklist, tempphase)

                risklist = []
                tempsurvival = []
                tempphase = []
                tempstage = []

                pbar.update()
            pbar.close()

    Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk), torch.tensor(EpochPhase))

    #print(" acc:" + str(Epochacc))
    #statistical_vis(Figure_dir, (EpochSurv, EpochRisk, EpochStage, EpochPhase, EpochID), epoch)

    return 0
# -*- coding: utf-8 -*-

import random
import os
import copy
import torch
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
import sys
import torch.nn.functional as F
import torch.nn as nn

from torch import optim
from torch_geometric.transforms import Polar
from torch_geometric.data import DataListLoader
from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.utils import dropout_adj
from torch_geometric.utils import add_self_loops
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, OneCycleLR
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv as GATConv_v1
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.nn import MessageNorm, PairNorm, GraphSizeNorm
from torch_ema import ExponentialMovingAverage as EMA
from torch.nn import BatchNorm1d, LayerNorm

from tqdm import tqdm

from model_selection import model_selection
from utils import train_test_split
from utils import makecheckpoint_dir_graph as mcd
from utils import TrainValid_path
from utils import non_decay_filter
from utils import coxph_loss
from utils import cox_sort
from utils import accuracytest

from torch.utils.data.sampler import Sampler


class Sampler_custom(Sampler):

    def __init__(self, event_list, censor_list, batch_size):
        self.event_list = event_list
        self.censor_list = censor_list
        self.batch_size = batch_size

    def __iter__(self):

        train_batch_sampler = []
        Event_idx = copy.deepcopy(self.event_list)
        Censored_idx = copy.deepcopy(self.censor_list)
        np.random.shuffle(Event_idx)
        np.random.shuffle(Censored_idx)

        Int_event_batch_num = Event_idx.shape[0] // 2
        Int_event_batch_num = Int_event_batch_num * 2
        Event_idx_batch_select = np.random.choice(Event_idx.shape[0], Int_event_batch_num, replace=False)
        Event_idx = Event_idx[Event_idx_batch_select]

        Int_censor_batch_num = Censored_idx.shape[0] // (self.batch_size - 2)
        Int_censor_batch_num = Int_censor_batch_num * (self.batch_size - 2)
        Censored_idx_batch_select = np.random.choice(Censored_idx.shape[0], Int_censor_batch_num, replace=False)
        Censored_idx = Censored_idx[Censored_idx_batch_select]

        Event_idx_selected = np.random.choice(Event_idx, size=(len(Event_idx) // 2, 2), replace=False)
        Censored_idx_selected = np.random.choice(Censored_idx, size=(
            (Censored_idx.shape[0] // (self.batch_size - 2)), (self.batch_size - 2)), replace=False)

        if Event_idx_selected.shape[0] > Censored_idx_selected.shape[0]:
            Event_idx_selected = Event_idx_selected[:Censored_idx_selected.shape[0],:]
        else:
            Censored_idx_selected = Censored_idx_selected[:Event_idx_selected.shape[0],:]

        for c in range(Event_idx_selected.shape[0]):
            train_batch_sampler.append(
                Event_idx_selected[c, :].flatten().tolist() + Censored_idx_selected[c, :].flatten().tolist())

        return iter(train_batch_sampler)

    def __len__(self):
        return len(self.event_list) // 2

class CoxGraphDataset(Dataset):

    def __init__(self, filelist, survlist, stagelist, censorlist, Metadata, mode, model, transform=None, pre_transform=None):
        super(CoxGraphDataset, self).__init__()
        self.filelist = filelist
        self.survlist = survlist
        self.stagelist = stagelist
        self.censorlist = censorlist
        self.Metadata = Metadata
        self.mode = mode
        self.model = model
        self.polar_transform = Polar()

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

        mock_data = Data(x=data_origin.x[:,:1792], edge_index=data_origin.edge_index, pos=data_origin.pos)

        data_re.pos = data_origin.pos
        data_re_polar = self.polar_transform(mock_data)
        polar_edge_attr = data_re_polar.edge_attr

        if (data_re.edge_index.shape[1] != data_origin.edge_attr.shape[0]):
            print('error!')
            print(self.filelist[idx].split('/')[-1])
        else:
            data = transfer(data_re)
            data.survival = torch.tensor(survival)
            data.phase = torch.tensor(phase)
            data.mets_class = torch.tensor(mets_class)
            data.stage = torch.tensor(stage)
            data.item = item
            data.edge_attr = polar_edge_attr
            data.pos = data_origin.pos

        return data

def Train(Argument):

    checkpoint_dir, Figure_dir = mcd(Argument)

    batch_num = int(Argument.batch_size)
    device = torch.device(int(Argument.gpu))
    Metadata = pd.read_csv('./Sample_data_for_demo/Metadata/KIRC_clinical.tsv', sep='\t')
    TrainRoot = TrainValid_path(Argument.DatasetType)
    Trainlist = os.listdir(TrainRoot)
    Trainlist = [item for c, item in enumerate(Trainlist) if '0_graph_torch_4.3_artifact_sophis_final.pt' in item]
    Fi = Argument.FF_number

    TrainFF_set, ValidFF_set, Test_set = train_test_split(Trainlist, Metadata, Argument.DatasetType, TrainRoot, Fi)

    TestDataset = CoxGraphDataset(filelist=Test_set[0], survlist=Test_set[1],
                                  stagelist=Test_set[3], censorlist=Test_set[2],
                                  Metadata=Metadata, mode=Argument.DatasetType,
                                  model=Argument.model)
    TrainDataset = CoxGraphDataset(filelist=TrainFF_set[0], survlist=TrainFF_set[1],
                                   stagelist=TrainFF_set[3], censorlist=TrainFF_set[2],
                                   Metadata=Metadata, mode=Argument.DatasetType,
                                   model=Argument.model)
    ValidDataset = CoxGraphDataset(filelist=ValidFF_set[0], survlist=ValidFF_set[1],
                                   stagelist=ValidFF_set[3], censorlist=ValidFF_set[2],
                                   Metadata=Metadata, mode=Argument.DatasetType,
                                   model=Argument.model)

    Event_idx = np.where(np.array(TrainFF_set[2]) == 1)[0]
    Censored_idx = np.where(np.array(TrainFF_set[2]) == 0)[0]
    train_batch_sampler = Sampler_custom(Event_idx, Censored_idx, batch_num)

    torch.manual_seed(12345)
    test_loader = DataListLoader(TestDataset, batch_size=batch_num, shuffle=True, num_workers=8, pin_memory=True,
                             drop_last=False)
    train_loader = DataListLoader(TrainDataset, batch_sampler=train_batch_sampler, num_workers=8, pin_memory=True)
    val_loader = DataListLoader(ValidDataset, batch_size=batch_num, shuffle=True, num_workers=8, pin_memory=True,
                            drop_last=False)

    model = model_selection(Argument)
    model_parameter_groups = non_decay_filter(model)

    model = DataParallel(model, device_ids=[0, 1], output_device=0)
    model = model.to(device)
    Cox_loss = coxph_loss()
    Cox_loss = Cox_loss.to(device)
    risklist = []
    optimizer_ft = optim.AdamW(model_parameter_groups, lr=Argument.learning_rate, weight_decay=Argument.weight_decay)
    scheduler = OneCycleLR(optimizer_ft, max_lr=Argument.learning_rate, steps_per_epoch=len(train_loader),
                           epochs=Argument.num_epochs)

    tempsurvival = []
    tempphase = []
    transfer = T.ToSparseTensor()
    bestloss = 100000
    bestacc = 0
    bestepoch = 0

    loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    BestAccDict = {'train': 0, 'val': 0, 'test':0}
    AccHistory = {'train': [], 'val': [], 'test': []}
    LossHistory = {'train': [], 'val': [], 'test': []}
    RiskAccHistory = {'train': [], 'val': [], 'test': []}
    RiskLossHistory = {'train': [], 'val': [], 'test': []}
    ClassAccHistory = {'train': [], 'val': [], 'test': []}
    ClassLossHistory = {'train': [], 'val': [], 'test': []}

    global_batch_counter = 0

    FFCV_accuracy = []
    FFCV_best_epoch = []

    with tqdm(total=Argument.num_epochs) as pbar:
        for epoch in range(0, int(Argument.num_epochs)):
            phaselist = ['train', 'val', 'test']

            for mode in phaselist:

                if mode == 'train':
                    model.train()
                    grad_flag = True
                else:
                    model.eval()
                    grad_flag = False

                with torch.set_grad_enabled(grad_flag):
                    loss = 0
                    risk_loss = 0
                    class_loss = 0
                    EpochSurv = []
                    EpochPhase = []
                    EpochRisk = []
                    EpochTrueMeta = []
                    EpochPredMeta = []
                    EpochFeature = []
                    EpochID = []
                    EpochStage = []
                    Epochloss = 0
                    Aux_node_loss = 0
                    Aux_edge_loss = 0
                    Risk_loss = 0
                    Epochriskloss = 0
                    Epochclassloss = 0
                    batchcounter = 1
                    pass_count = 0

                    # with tqdm(total=len(loader[mode])) as pbar:
                    for c, d in enumerate(loader[mode], 1):
                        optimizer_ft.zero_grad()

                        tempsurvival = torch.tensor([data.survival for data in d])
                        tempphase = torch.tensor([data.phase for data in d])
                        tempID = np.asarray([data.item for data in d])
                        tempstage = torch.tensor([data.stage for data in d])
                        tempmeta = torch.tensor([data.mets_class for data in d])

                        out = model(d)

                        risklist, tempsurvival, tempphase, tempmeta, EpochSurv, EpochPhase, EpochRisk, EpochStage = \
                            cox_sort(out, tempsurvival, tempphase, tempmeta, tempstage, tempID,
                                     EpochSurv, EpochPhase, EpochRisk, EpochStage, EpochID)

                        if torch.sum(tempphase).cpu().detach().item() < 1:
                            pass_count += 1
                        else:
                            risk_loss = Cox_loss(risklist, tempsurvival, tempphase)
                            loss = risk_loss

                            if mode == 'train':
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model_parameter_groups[0]['params'], max_norm=Argument.clip_grad_norm_value, error_if_nonfinite=True)
                                torch.nn.utils.clip_grad_norm_(model_parameter_groups[1]['params'], max_norm=Argument.clip_grad_norm_value, error_if_nonfinite=True)
                                optimizer_ft.step()
                                scheduler.step()

                            Epochloss += loss.cpu().detach().item()
                            Batchacc = accuracytest(tempsurvival, risklist, tempphase)

                            batchcounter += 1
                            risklist = []
                            tempsurvival = []
                            tempphase = []
                            tempstage = []
                            final_updated_feature_list = []
                            updated_feature_list = []


                    Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk),
                                            torch.tensor(EpochPhase))
                    Epochloss = Epochloss / batchcounter


                    if mode == 'train':
                        if Epochacc > BestAccDict['train']:
                            BestAccDict['train'] = Epochacc

                    elif mode == 'val':
                        if Epochacc > BestAccDict['val']:
                            BestAccDict['val'] = Epochacc

                    elif mode == 'test':
                        if Epochacc > BestAccDict['test']:
                            BestAccDict['test'] = Epochacc

                    print()
                    print('epoch:' + str(epoch))
                    print(" mode:" + mode)
                    print(" loss:" + str(Epochloss) + " acc:" + str(Epochacc) + " pass count:" + str(pass_count))

                    checkpointinfo = 'epoch-{},acc-{:4f},loss-{:4f}.pt'

                    if mode == 'test':
                        if epoch == 0:
                            torch.save(model.state_dict(), os.path.join(checkpoint_dir,
                                                                        checkpointinfo.format(epoch, Epochacc,
                                                                                              Epochloss)))
                        else:
                            if Epochacc > bestacc or Epochloss < bestloss:
                                bestepoch = epoch
                                torch.save(model.state_dict(), os.path.join(checkpoint_dir,
                                                                            checkpointinfo.format(epoch, Epochacc,
                                                                                                  Epochloss)))

                            if Epochacc > bestacc:
                                bestacc = Epochacc

                            if Epochloss < bestloss:
                                bestloss = Epochloss

        pbar.update()

    FFCV_accuracy.append(bestacc)
    FFCV_best_epoch.append(bestepoch)

    bestFi = np.argmax(FFCV_accuracy)
    best_checkpoint_dir = os.path.join(checkpoint_dir, str(bestFi))
    best_figure_dir = os.path.join(checkpoint_dir, str(bestFi))

    Argument.checkpoint_dir = best_checkpoint_dir

    return model, best_checkpoint_dir, best_figure_dir, FFCV_best_epoch[bestFi]

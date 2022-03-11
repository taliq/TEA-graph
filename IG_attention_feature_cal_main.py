#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:42:06 2020

@author: taliq
"""
import os

import sys
import argparse

from IG_attention_feature_cal import Vis_main
from Subgraph_visualization import Subgraph_analysis_vis

def Parser_main():
    parser = argparse.ArgumentParser(description="Deep cox analysis model")
    parser.add_argument("--Description", default="BorameProgression_decreasefeature100ReFinal",
                        help="Describe the objective of the running")
    parser.add_argument("--Pretrain", default="N", help="Cox pretrained feature or not", type=str)
    parser.add_argument("--DatasetType", default="TCGA", help="TCGA_BRCA or BORAME or BORAME_Meta or BORAME_Prog",
                        type=str)
    parser.add_argument("--Logging", default=True, help="record to excel file or not", type=bool)
    parser.add_argument("--learning_rate", default=0.0001, help="Learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.001, help="Weight decay rate", type=float)
    parser.add_argument("--batch_size", default=4, help="batch size", type=int)
    parser.add_argument("--gpu", default=0, help="select gpu device", type=str)
    parser.add_argument("--Sampling", default=1, help="Sampling percentage of large train set", type=int)
    parser.add_argument("--load_state_dict",
                        default="/home/taliq_lee/2021_DSA/script_for_submission/results/TCGA/GAT_custom/2022-03-11_08:45:00/epoch-1,acc-0.626506,loss-0.964262.pt",
                        help="Checkpoint saved directory",type=str)
    #parser.add_argument("--load_state_dict",
    #                    default="/mnt/sda/DSA_Archive/BORAME_Meta/2021-07-04_20:09:06/0/epoch-52,acc-0.817423,loss-1.413782.pt",
    #                    help="Checkpoint saved directory",type=str)
    parser.add_argument("--num_epochs", default=1, help="Number of epochs", type=int)
    parser.add_argument("--dropedge_rate", default=0.0, help="Dropout rate of block", type=float)
    parser.add_argument("--dropout_rate", default=0.0, help="Dropout rate of block", type=float)
    parser.add_argument("--graph_dropout_rate", default=0.0, help="Dropout rate of the graph node", type=float)
    parser.add_argument("--Run_mode", default="Train", help="Train or Extract", type=str)
    parser.add_argument("--Stat_vis_mode", default=False, help="Visualize stat interpretation", type=bool)
    parser.add_argument("--initial_dim", default=50, help="Initial dimension for the GNN", type=int)
    parser.add_argument("--attention_head_num", default=2, help="Initial dimension for the GNN", type=int)
    parser.add_argument("--clip_grad_norm_value", default=5.0, help="Initial dimension for the GNN", type=float)
    parser.add_argument("--number_of_layers", default=3, help="Whole number of layer in the GNN", type=int)
    parser.add_argument("--noise_var_value", default=0.3, help="Noise variation value", type=float)
    parser.add_argument("--noise_node_portion", default=0.1, help="Noise variation value", type=float)
    parser.add_argument("--FF_number", default=0, help="Five fold cross validation", type=int)
    parser.add_argument("--parameter_sweep", default="N", help="", type=str)
    parser.add_argument("--parameter_folder", default="mode_test", help="", type=str)
    parser.add_argument("--parameter_set", default="Set15", help="", type=str)
    parser.add_argument("--model", default="GAT_custom", help="GIN or GCN or GAT or MLP or AttMLP", type=str)
    parser.add_argument("--prelayernum", default=2, type=int)
    parser.add_argument("--postlayernum", default=2, type=int)
    parser.add_argument("--with_distance", default="Y", type=str)
    parser.add_argument("--simple_distance", default="N", type=str)
    parser.add_argument("--with_noise", default="N", type=str)
    parser.add_argument("--loss_type", default="PRELU", type=str)
    parser.add_argument("--residual_connection", default="Y", type=str)
    parser.add_argument("--Corr_threshold", default="0.75", help="0.5 // 0.65 // 0.75", type=str)
    parser.add_argument("--norm_type", default="layer", help="layer // batch", type=str)
    parser.add_argument("--angle", default="0", help="trimming_angle", type=str)

    return parser.parse_args()

def main():
    Argument = Parser_main()
    Vis_main(Argument)
    Subgraph_analysis_vis(Argument)

if __name__ == "__main__":
    main()

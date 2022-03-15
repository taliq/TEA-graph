#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:42:06 2020

@author: taliq
"""

import argparse

from IG_attention_feature_cal import Vis_main
from Subgraph_visualization import Subgraph_analysis_vis

def Parser_main():
    parser = argparse.ArgumentParser(description="Deep cox analysis model")
    parser.add_argument("--DatasetType", default="TCGA", help="TCGA_BRCA or BORAME or BORAME_Meta or BORAME_Prog",
                        type=str)
    parser.add_argument("--learning_rate", default=0.0001, help="Learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.00005, help="Weight decay rate", type=float)
    parser.add_argument("--batch_size", default=6, help="batch size", type=int)
    parser.add_argument("--num_epochs", default=50, help="Number of epochs", type=int)
    parser.add_argument("--dropedge_rate", default=0.25, help="Dropout rate of block", type=float)
    parser.add_argument("--dropout_rate", default=0.25, help="Dropout rate of block", type=float)
    parser.add_argument("--graph_dropout_rate", default=0.25, help="Dropout rate of the graph node", type=float)
    parser.add_argument("--initial_dim", default=100, help="Initial dimension for the GNN", type=int)
    parser.add_argument("--attention_head_num", default=2, help="Initial dimension for the GNN", type=int)
    parser.add_argument("--clip_grad_norm_value", default=2.0, help="Initial dimension for the GNN", type=float)
    parser.add_argument("--number_of_layers", default=3, help="Whole number of layer in the GNN", type=int)
    parser.add_argument("--FF_number", default=0, help="Five fold cross validation", type=int)
    parser.add_argument("--model", default="GAT_custom", help="GIN or GCN or GAT or MLP or AttMLP", type=str)
    parser.add_argument("--gpu", default=0, help="GIN or GCN or GAT or MLP or AttMLP", type=int)
    parser.add_argument("--norm_type", default="layer", type=str)
    parser.add_argument("--prelayernum", default=3, type=int)
    parser.add_argument("--postlayernum", default=2, type=int)
    parser.add_argument("--with_distance", default="Y", type=str)
    parser.add_argument("--simple_distance", default="N", type=str)
    parser.add_argument("--loss_type", default="PRELU", type=str)
    parser.add_argument("--residual_connection", default="Y", type=str)
    parser.add_argument("--Corr_threshold", default="0.75", help="0.5 // 0.65 // 0.75", type=str)
    parser.add_argument("--load_state_dict",
                        default="./results/TCGA/GAT_custom/2022-03-11_08:45:00/epoch-1,acc-0.626506,loss-0.964262.pt",
                        help="Checkpoint saved directory",type=str)

    return parser.parse_args()

def main():
    Argument = Parser_main()
    Vis_main(Argument)
    Subgraph_analysis_vis(Argument)

if __name__ == "__main__":
    main()

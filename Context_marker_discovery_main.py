import os
import numpy as np
import pandas as pd

import argparse
from Context_marker_discovery.Clustering_module import Subgraph_feature_prep
from Context_marker_discovery.Clustering_module import Save_dir_create
from Context_marker_discovery.Clustering_module import Subgraph_level_cluster_analysis
from Context_marker_discovery.Clustering_module import Patch_level_cluster_analysis
from Context_marker_discovery.Clustering_module import Final_visualization

def Parser_main():
    parser = argparse.ArgumentParser(description="Deep cox analysis model")
    parser.add_argument("--Subgraph_save_dir",
                        default="./results/TCGA/GAT_custom/",
                        help="Select the directory where IG_analysis results are saved ",
                        type=str)
    parser.add_argument("--original_data_dir",
                        default="./Sample_data_for_demo/Graphdata/KIRC/",
                        help="Select the directory where superpatch networks are saved")
    parser.add_argument("--WSI_rootdir",
                        default="./Sample_data_for_demo/Raw_WSI/",
                        help="Select the directory where WSIs are saved ")
    parser.add_argument("--IG_type",
                        default="Mid", help="Top, Mid, Low")
    parser.add_argument("--save_dir_name", default="test", type=str)
    parser.add_argument("--patch_cluster_num", default=8, type=int)
    parser.add_argument("--graph_cluster_num", default=10, type=int)

    return parser.parse_args()

def main():
    Argument = Parser_main()

    rootdir = Argument.Subgraph_save_dir
    pt_rootdir = Argument.original_data_dir
    IG_rootdir = os.path.join(Argument.Subgraph_save_dir, "IG_analysis")
    csv_rootdir = pt_rootdir

    Metadata = pd.read_csv('./Sample_data_for_demo/Metadata/KIRC_clinical.tsv', sep='\t')

    IG_whole_feature, Patient_ID_list, IG_ID_list, IG_length_list, Surv, Event \
        = Subgraph_feature_prep(IG_rootdir, pt_rootdir, Argument.IG_type, Metadata)

    graph_dir, patch_dir = Save_dir_create(rootdir, Argument.save_dir_name, Argument.IG_type)

    patch_cluster_num = Argument.patch_cluster_num
    graph_cluster_num = Argument.graph_cluster_num

    graph_cluster_label_list = range(graph_cluster_num)

    subgraph_df, label_df, length_df, rmst_difference_dict = \
        Subgraph_level_cluster_analysis(IG_whole_feature,Patient_ID_list,graph_cluster_num,
                                    IG_ID_list, IG_length_list, Surv, Event, Argument.IG_type, graph_dir)

    patch_df, graph_label_df, patch_label_kmean = Patch_level_cluster_analysis(IG_rootdir, patch_dir, label_df,
                                 patch_cluster_num, length_df, pt_rootdir,
                                 Argument.IG_type, Argument.WSI_rootdir)

    label_list = 0

    Final_visualization(graph_label_df, patch_label_kmean, graph_cluster_num, patch_cluster_num,
                         graph_dir, label_list, rootdir,
                         pt_rootdir, csv_rootdir, Argument.WSI_rootdir, Argument.IG_type)

if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import openslide as osd
from tqdm import tqdm
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from MulticoreTSNE import MulticoreTSNE as TSNE
from lifelines.utils import restricted_mean_survival_time
from lifelines.statistics import multivariate_logrank_test
import numpy.linalg
# from spherecluster.spherecluster import SphericalKMeans

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statistics import median
from random import seed
import cv2 as cv
import networkx as nx
import seaborn as sns
from collections import Counter
from random import sample
import pickle


def run_tsne(X):
    tsne = TSNE(n_jobs=8)
    embeddings = tsne.fit_transform(X)

    return embeddings

def Visualize_tsne(embeddings, feat, cluster_num, save_dir, label, c1="Set1"):
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    cmap = matplotlib.cm.get_cmap(c1, cluster_num)
    plt.figure(figsize=(10, 10))
    plt.scatter(vis_x, vis_y, c=label, cmap=cmap, alpha=0.5, s=10)
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, feat + '_tsne.png'), transparent=True)
    plt.savefig(os.path.join(save_dir, feat + '_tsne.pdf'), transparent=True)
    plt.cla()
    plt.clf()
    plt.close()


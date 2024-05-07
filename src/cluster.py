from models.cluster_machine import ClusterMachine
from utils.preprocess import init_datasets, get_full_data, init_dual_datasets
from utils.plot import plot_map
import math
from models.som_vae import SOMVAE, ConvEncoder, ConvDecoder
import torch
import random

random.seed(10)

# runner
if __name__ == "__main__":
    n_states = init_datasets()
    init_dual_datasets()
    all_timeseries = get_full_data()
    # random.shuffle(all_timeseries)
    print(all_timeseries[0].head())
    print('=====================')
    print('number of states: ', n_states)
    n_clusters = 4
    print('number of clusters: ', n_clusters)

    # standard K means clustering
    cm1 = ClusterMachine(all_timeseries, iter=50000)
    print('=====================')
    print('performing kmeans clustering...')
    new_data = cm1.cluster(algorithm='kmeans', n_clusters=n_clusters, verbose=True)
    county_cluster_map1 = cm1.cluster_data_map
    print('plotting kmeans map')
    plot_map(county_cluster_map1, algorithm='kmeans')
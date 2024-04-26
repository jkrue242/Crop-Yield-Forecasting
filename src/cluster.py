from models.cluster_machine import ClusterMachine
from utils.preprocess import init_datasets, get_full_data
from utils.plot import plot_map
import math
from models.som_vae import SOMVAE, ConvEncoder, ConvDecoder
import torch

if __name__ == "__main__":
    n_states = init_datasets()
    all_timeseries = get_full_data()
    print('=====================')
    print('number of states: ', n_states)
    n_clusters = 9
    print('number of clusters: ', n_clusters)

    # # standard K means clustering
    # cm1 = ClusterMachine(all_timeseries, iter=10000)
    # print('=====================')
    # print('performing kmeans clustering...')
    # new_data = cm1.cluster(algorithm='kmeans', n_clusters=n_clusters, verbose=True)
    # county_cluster_map1 = cm1.cluster_data_map
    # print('plotting kmeans map')
    # plot_map(county_cluster_map1, algorithm='kmeans')

    # miniSOM clustering
    cm2 = ClusterMachine(all_timeseries, iter=50000)
    print('=====================')
    print('training SOM...')

    num_series = len(all_timeseries)
    print('num_series:', num_series)
    new_data = cm2.cluster(algorithm='som', verbose=True, som_x=3, som_y=3, n_clusters=n_clusters)
    county_cluster_map2 = cm2.cluster_data_map
    print('plotting MiniSOM map')
    plot_map(county_cluster_map2, algorithm='minisom')
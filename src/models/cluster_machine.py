import pandas as pd
import torch
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
import collections
import os
import math
import sys
from utils.minisom import MiniSom 
import tslearn 
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class ClusterMachine:
    def __init__(self, data, iter=50000):
        self.data = data
        self.iter = iter
        self.centroids = None
        self.clusters = None
        self.scaled_data = data
        self.data_dict = collections.defaultdict(dict)
        self.scaled_data_dict = collections.defaultdict(dict)
        self.normalized_data = []
        self.cluster_count = 0
        self.state_county_index = collections.defaultdict(dict)
        self.cluster_data_map = collections.defaultdict(dict)

    
    def cluster(self, algorithm='kmeans', som_x=None, som_y=None, verbose=True, n_clusters=9):
        new_data = self.normalize()
        print('CHECKPOINT')
        if algorithm == 'kmeans':
            labels, score = self.kmeans(new_data, verbose=verbose, n_clusters=n_clusters)
            print("silhouette score: ", score)
        
        if algorithm == 'som':
            if som_x is not None and som_y is not None:
                self.cluster_count = som_x * som_y
            
            som = self.som_algorithm(new_data, verbose=verbose, som_x=som_x, som_y=som_y)
            new_data = []
            labels = []
            label = 0
            for i in range(som_x):
                for j in range(som_y):
                    cluster = (i, j)
                    if cluster in som.keys():
                        for series in som[cluster]:
                            new_data.append(series)
                            labels.append(label)
                        label += 1

        # determine the cluster for each state-county pair
        for i in range(len(labels)):
            for state in self.state_county_index.keys():
                for county in self.state_county_index[state].keys():
                    if self.state_county_index[state][county] == i:
                        self.cluster_data_map[state][county] = labels[i]

        self.plot_clusters(new_data, labels, algorithm=algorithm)
        return self.scaled_data_dict

    def normalize(self):
        normalized_data = []
        index = 0
        for i in range(len(self.data)):
            if not self.data[i].empty:
                curr_ts = self.data[i]
                state = curr_ts['State'].unique()[0]
                county = curr_ts['County'].unique()[0]
                if type(state) == str and type(county) == str:
                    self.data_dict[state][county] = curr_ts[["CORN, GRAIN - YIELD, MEASURED IN BU / ACRE"]]
                    scaler = MinMaxScaler()
                    self.scaled_data_dict[state][county] = MinMaxScaler().fit_transform(self.data_dict[state][county])
                    self.scaled_data_dict[state][county]= self.scaled_data_dict[state][county].reshape(len(self.data_dict[state][county]))
                    normalized_data.append(self.scaled_data_dict[state][county])
                    self.state_county_index[state][county] = index
                    index += 1
        self.normalized_data = normalized_data
        return self.normalized_data


    def som_algorithm(self, data, verbose=False, som_x=3, som_y=3):
        som = MiniSom(som_x, som_y, len(data[0]), sigma=0.3, learning_rate = 0.1, random_seed=10)
        # som.random_weights_init(data)
        som.random_weights_init(data)
        som.train(data, self.iter, verbose=verbose)
        win_map = som.win_map(data)
        return win_map

    def kmeans(self, data, verbose=True, n_clusters=9):
        km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=verbose, max_iter=self.iter)
        self.cluster_count = n_clusters
        data = tslearn.utils.to_time_series_dataset(data)
        print('kmeans input: ', data)
    
        labels = km.fit_predict(data)  
        print('=======================================')
        print('labels type: ', type(labels))
        print('labels[0]', labels[0])
        print('---------------------------------------')
        print('data type: ', type(data))
        # score = 0
        score = tslearn.clustering.silhouette_score(data, labels)
        
        print('KMeans completed')

        # determine the cluster for each state-county pair
        for i in range(len(labels)):
            for state in self.state_county_index.keys():
                for county in self.state_county_index[state].keys():
                    if self.state_county_index[state][county] == i:
                        self.cluster_data_map[state][county] = labels[i]
        return labels, score

    def plot_clusters(self, data, labels,algorithm='kmeans'):
        len_rows = 4
        len_cols = 4
        fig, axs = plt.subplots(len_rows,len_cols, figsize=(25,10))
        row_i=0 
        column_j=0
        # For each label there is,
        # plots every series with that label
        for label in set(labels):
            cluster = []
            for i in range(len(labels)):
                    if(labels[i]==label):
                        axs[row_i, column_j].plot(data[i],c="gray",alpha=0.4)
                        cluster.append(data[i])
            if len(cluster) > 0:
                axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
            axs[row_i, column_j].set_title("Cluster "+str(row_i*(self.cluster_count)//2 + column_j))
            column_j+=1
            if column_j%(len_cols) == 0:
                row_i+=1
                column_j=0
        fig.tight_layout()
        print(f'saving image to images/{algorithm}.png')
        plt.savefig(f"images/{algorithm}_clustersA.png")
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import collections
import os
import math
from utils.minisom import MiniSom   
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class ClusterMachine:
    def __init__(self, data:[pd.DataFrame()], k, iter=1000):
        self.data = data
        self.k = k
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

    
    def cluster(self):
        new_data = self.normalize()
        # som = self.som_algorithm(new_data, verbose=True)
        #self.plot_som(5,5,som)
        labels = self.kmeans(new_data)
        # print(labels)
        self.plot_kmeans(new_data, labels)
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


    def som_algorithm(self, data, verbose=False):
        # data_list = []
        # for state in data.keys():
        #     for county in data[state].keys():
        #         data_list.append(data[state][county])

        # som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(data))))
        som_x = som_y = 2
        som = MiniSom(som_x, som_y, len(data[0]), sigma=0.3, learning_rate = 0.1)

        som.random_weights_init(data)
        som.train(data, 50000, verbose=False)
        win_map = som.win_map(data)
        if verbose:
            print('Total Clusters: ', len(win_map.keys()))
            for i in range(len(win_map.keys())): 
                print(f'Cluster {i} has {len(win_map[i])} elements')
        return win_map

    
    def plot_som(self, win_map, som_x=5, som_y=5,):
        fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
        fig.suptitle('Clusters')
        for x in range(som_x):
            for y in range(som_y):
                if cluster in win_map.keys():
                    for series in win_map[cluster]:
                        axs[cluster].plot(series,c="gray",alpha=0.5) 
                    axs[cluster].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
                cluster_number = x*som_y+y+1
                axs[cluster].set_title(f"Cluster {cluster_number}")
        plt.show()
        plt.savefig("/images/som_clusters.png")

    def kmeans(self, data):
        cluster_count = math.ceil(10) 
        km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw")
        self.cluster_count = cluster_count
        labels = km.fit_predict(data)
        print('number of labels: ', len(labels))
        # determine the cluster for each state-county pair
        for i in range(len(labels)):
            for state in self.state_county_index.keys():
                for county in self.state_county_index[state].keys():
                    # print(f'{state}, {county}: ', self.state_county_index[state][county])
                    if self.state_county_index[state][county] == i:
                        self.cluster_data_map[state][county] = labels[i]
        return labels

    def plot_kmeans(self, data, labels, cluster_count=10):
        plot_count = self.cluster_count
        fig, axs = plt.subplots(2,plot_count//2,figsize=(25,10))
        fig.suptitle('Clusters')
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
            if column_j%(plot_count//2) == 0:
                row_i+=1
                column_j=0
                
        plt.savefig("images/kmeans.png")
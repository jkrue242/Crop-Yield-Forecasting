from models.cluster_machine import ClusterMachine
from utils.preprocess import get_midwest_data
from utils.plot import plot_map


if __name__ == "__main__":
    all_timeseries = get_midwest_data()
    cm = ClusterMachine(all_timeseries, 3)
    new_data = cm.cluster()
    
    county_cluster_map = cm.cluster_data_map
    # for state in county_cluster_map.keys():
    #     for county in county_cluster_map[state].keys():
    #         print(f"{state}, {county}: {county_cluster_map[state][county]}")
    plot_map(county_cluster_map)

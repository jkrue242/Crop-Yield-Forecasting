from models.cluster_machine import ClusterMachine
from utils.preprocess import get_midwest_data



if __name__ == "__main__":
    all_timeseries = get_midwest_data()
    cm = ClusterMachine(all_timeseries, 3)
    new_data = cm.cluster()

from models import kmeans
from utils.preprocess import get_midwest_data



if __name__ == "__main__":
    df = get_midwest_data()
    print(df.head())
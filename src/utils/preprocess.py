import pandas as pd
import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import yfinance as yf
sys.path.append('data')
warnings.filterwarnings('ignore')
data_path = 'data/'

# get harvest data for polk county
def _get_county_data(path=data_path +'polk_county_data_corn.csv', start_year=1972, end_year=2021):
    raw = pd.read_csv(path)[['Year', 'Data Item', 'Value']]
    timeseries = raw.pivot(index='Year', columns='Data Item', values='Value').reset_index().rename(
        columns={
            'Year': 'Year', 
            'CORN - ACRES PLANTED': 'Acres Planted', 
            'CORN, GRAIN - ACRES HARVESTED': 'Acres Harvested', 
            'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE': 
            'Yield (bu/ac)'
        }
    )

    timeseries['Acres Planted'] = timeseries['Acres Planted'].str.replace(',', '').astype(float)
    timeseries['Acres Harvested'] = timeseries['Acres Harvested'].str.replace(',', '').astype(float)
    timeseries['Yield (bu/ac)'] = timeseries['Yield (bu/ac)'].astype(float)

    timeseries = timeseries[(timeseries['Year'] >= start_year) & (timeseries['Year'] <= end_year)]
    return timeseries

# get john deere stock data
def _get_deere_stock_data(start_year=1972, end_year=2021):
    # grab data from yahoo finance for stock John Deere and plot
    deere = yf.Ticker('DE')
    deere_stock = deere.history(start=str(start_year)+'-01-01', end=str(end_year)+'-12-31')

    # convert index to column
    deere_stock.reset_index(inplace=True)

    # convert datetime to year
    deere_stock['Year'] = deere_stock['Date'].dt.year

    deere_stock_by_year = deere_stock.groupby('Year').agg({'Close': 'mean'}).reset_index().rename({'Close': 'DE Avg Stock Price'}, axis=1)  
    return deere_stock_by_year

# get weather data for the des moines area
def _get_weather_data(path=data_path + 'dsm_weather_data.csv', start_year=1972, end_year=2021):
    weather_dsm = pd.read_csv(path, delimiter=',')
    weather_dsm.replace('M',np.NaN, inplace=True)
    weather_dsm.ffill(inplace=True)
    weather_dsm = weather_dsm[(weather_dsm['Year'] >= start_year) & (weather_dsm['Year'] <= end_year)]
    weather_dsm.drop(['Max Temp', 'Min Temp', 'HDD', 'CDD'], axis=1, inplace=True)

    weather_dsm['Year'] = weather_dsm['Year'].astype(int)
    weather_dsm['Precip'] = weather_dsm['Precip'].astype(float)
    weather_dsm['Snow'] = weather_dsm['Snow'].astype(float)
    weather_dsm['Mean Temp'] = weather_dsm['Mean Temp'].astype(float)
    weather_dsm['GDD'] = weather_dsm['GDD'].astype(float)
    weather_dsm['Precip'] = weather_dsm['Precip'].astype(float)
    weather_dsm['Precip'] = weather_dsm['Precip'].astype(float)
    return weather_dsm

# get all data, merge it, and return it
def get_data(start_year=1972, end_year=2021):
    county_timeseries = _get_county_data(start_year=start_year)
    deere_stock_by_year = _get_deere_stock_data(start_year=start_year)
    weather_dsm = _get_weather_data(start_year=start_year)
    timeseries = county_timeseries.merge(deere_stock_by_year, on='Year', how='left')
    timeseries = timeseries.merge(weather_dsm, on='Year', how='left')
    return timeseries


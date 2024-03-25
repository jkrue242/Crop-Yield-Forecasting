import pandas as pd
import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import yfinance as yf
import datetime
sys.path.append('data')
warnings.filterwarnings('ignore')
data_path = 'data/'

# get harvest data for polk county
def _get_county_data(path=data_path +'polk_county_harvest_corn.csv'):
    timeseries = _process_usda_data(path)
    return timeseries

def _process_usda_data(path):
    raw = pd.read_csv(path, parse_dates=["Year"])[['Year', 'Data Item', 'Value']]
    raw["Year"] = raw["Year"].dt.year
    timeseries = raw.pivot(index='Year', columns='Data Item', values='Value').reset_index()
    numeric_cols = timeseries.columns[1:]
    for col in numeric_cols:
        timeseries[col] = timeseries[col].str.replace(',', '').astype(float)
    return timeseries

def _get_pricing_data(path=data_path + 'corn-prices.csv'):
    raw = pd.read_csv(path, parse_dates=["date"])
    raw["Year"] = raw["date"].dt.year
    timeseries = raw.groupby('Year').agg({'price': 'mean'}).reset_index().rename({'price': 'Avg Price'}, axis=1)
    return timeseries

# get weather data for the des moines area
def _get_weather_data(path=data_path + 'dsm_climate_data_yoy.csv'):
    weather_dsm = pd.read_csv(path, delimiter=',', parse_dates=["Date"])[["Date", "Max Temp (Degrees Fahrenheit)", "Min Temp (Degrees Fahrenheit)", "Precip (Inches)", "Snow (Inches)"]]
    weather_dsm['Max Temp (Degrees Fahrenheit)'] = weather_dsm['Max Temp (Degrees Fahrenheit)'].astype(float)
    weather_dsm['Min Temp (Degrees Fahrenheit)'] = weather_dsm['Min Temp (Degrees Fahrenheit)'].astype(float)
    weather_dsm["Precip (Inches)"] = weather_dsm['Precip (Inches)'].astype(float)
    weather_dsm["Snow (Inches)"] = weather_dsm['Snow (Inches)'].astype(float)
    weather_dsm["Avg Temp"] = (weather_dsm["Max Temp (Degrees Fahrenheit)"] + weather_dsm["Min Temp (Degrees Fahrenheit)"]) / 2
    weather_dsm["Year"] = weather_dsm["Date"].dt.year
    weather_dsm.drop(["Date", "Max Temp (Degrees Fahrenheit)", "Min Temp (Degrees Fahrenheit)"], axis=1, inplace=True)
    weather_dsm = weather_dsm.groupby(["Year"]).agg(
        {
            'Avg Temp': 'mean',
            'Precip (Inches)': 'sum',
            'Snow (Inches)': 'sum'
        }
    ).reset_index()
    return weather_dsm

# get john deere stock data
def _get_deere_stock_data(start_year=1945, end_year=2022):
    # grab data from yahoo finance for stock John Deere and plot
    deere = yf.Ticker('DE')
    deere_stock = deere.history(start=str(start_year)+'-01-01', end=str(end_year)+'-12-31')

    # convert index to column
    deere_stock.reset_index(inplace=True)

    # convert datetime to year
    deere_stock['Year'] = deere_stock['Date'].dt.year

    deere_stock_by_year = deere_stock.groupby('Year').agg({'Close': 'mean'}).reset_index().rename({'Close': 'DE Avg Stock Price'}, axis=1)  
    return deere_stock_by_year
    
# get all data, merge it, and return it
def get_data():
    county_timeseries = _get_county_data()
    weather_dsm = _get_weather_data()
    deere_stock = _get_deere_stock_data()
    price_data = _get_pricing_data()
    timeseries = county_timeseries.merge(weather_dsm, on='Year', how='left')
    timeseries = timeseries.merge(deere_stock, on='Year', how='left')
    timeseries = timeseries.merge(price_data, on='Year', how='left')
    timeseries["Avg Price"] = timeseries["Avg Price"].fillna(1.0)
    timeseries["DE Avg Stock Price"] = timeseries["DE Avg Stock Price"].fillna(0.0)
    return timeseries


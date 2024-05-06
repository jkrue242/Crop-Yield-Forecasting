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

def init_datasets():
    full_data = pd.DataFrame()
    for i in range(1,6):
        df = _process_full_data(data_path + f'county_yields{i}.csv')
        full_data = pd.concat([full_data, df])
    full_data = full_data.drop_duplicates()
    n_states = len(full_data["State"].unique())
    full_data.to_csv(data_path + 'full_data.csv', index=False)
    return n_states

def init_dual_datasets():
    full_data = pd.DataFrame()
    for i in range(1,9):
        df = _process_full_data2(data_path + f'yield_and_prod{i}.csv')
        full_data = pd.concat([full_data, df])
    full_data = full_data.drop_duplicates()
    n_states = len(full_data["State"].unique())
    full_data.to_csv(data_path + 'full_data_dual.csv', index=False)
    return n_states

"""
Get data from USDA for corn yield (saved as a csv file)
"""
def _get_county_data(path=data_path +'polk_county_harvest_corn.csv'):
    timeseries = _process_usda_data(path)
    return timeseries

"""
Process the USDA data to make it usable 
"""
def _process_usda_data(path):
    raw = pd.read_csv(path, parse_dates=["Year"])[['Year', 'Data Item', 'Value']]
    
    # convert to datetime
    raw["Year"] = raw["Year"].dt.year
    # pivot to create useful columns
    timeseries = raw.pivot(index='Year', columns='Data Item', values='Value').reset_index()
    numeric_cols = timeseries.columns[1:]
    for col in numeric_cols:
        timeseries[col] = timeseries[col].str.replace(',', '').astype(float)
    return timeseries

def _process_full_data(path):
    raw = pd.read_csv(path, parse_dates=["Year"])[['Year', 'State', 'County', 'Data Item', 'Value']]
    raw = raw[raw['County'] != "OTHER COUNTIES"]
    # convert to datetime
    raw["Year"] = raw["Year"].dt.year
    timeseries = raw.pivot_table(index=['Year', 'State', 'County'], columns='Data Item', values='Value').reset_index()
    # pivot to create useful columns
    numeric_cols = timeseries.columns[4:]
    for col in numeric_cols:
        timeseries[col] = timeseries[col].str.replace(',', '').astype(float)
    return timeseries


def _clean(val):
    return float(val.replace(',', ''))


def _process_full_data2(path):
    raw = pd.read_csv(path, parse_dates=["Year"])[['Year', 'State', 'County', 'Data Item', 'Value']]
    raw = raw[raw['County'] != "OTHER COUNTIES"]

    raw['Value'] = raw['Value'].apply(_clean)    

    # convert to datetime
    raw["Year"] = raw["Year"].dt.year
    timeseries = raw.pivot_table(index=['Year', 'State', 'County'], columns='Data Item', values='Value').reset_index()
    return timeseries

"""
Get weather data from Des Moines, IA (saved as a csv file)
"""
def _get_weather_data(path=data_path + 'dsm_climate_data_yoy.csv'):
    # get data, convert columns to float, and calculate sums and averages over each year
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

def interpolate_full_data(data, upsample=True):
    series_lengths = [len(series) for series in data]
    problems_index = []
    max_len = max(series_lengths)
    longest_series = None
    for series in data:
        if len(series) == max_len:
            longest_series = series

    for i in range(len(data)):
        if len(data[i])!= max_len:
            problems_index.append(i)
            data[i] = data[i].reindex(longest_series.index)

    for i in problems_index:
        data[i].interpolate(limit_direction="both",inplace=True)
    
    for i in range(len(data)):
        data[i].reset_index(inplace=True)
        data[i].fillna(method='bfill', inplace=True)
        data[i].fillna(method='ffill', inplace=True)
        data[i].fillna(method='backfill', inplace=True)
    
    return data


def get_full_data(path=data_path +'full_data.csv', interpolate=True):
    print('fetching data...')
    full = pd.read_csv(path, delimiter=',')
    print('splitting data...')
    series = breakup_timeseries(full)
    if interpolate:
        print('interpolating data...')
        return  interpolate_full_data(data=series)
    return series

def breakup_timeseries(data):
    ts = []
    data = data[(data["County"] != "OTHER COUNTIES") & (data["County"] != "OTHER (COMBINED) COUNTIES")]
    data.set_index('Year', inplace=True)
    pairs = data[['State', 'County']].drop_duplicates().values.tolist()

    for p in pairs:
        state = p[0]
        county = p[1]
        ts.append(data[(data['State'] == state) & (data['County'] == county)])
    return ts

"""
Upsample the data to a given frequency (default to monthly)
"""
def resample(data, freq='M'):
    date_range = pd.date_range(start='1945-01-01', end='2022-01-01', freq='AS')
    data.index = date_range

    # resample, fill missing values using linear interpolation
    data = data.resample('MS').interpolate(method='linear')
    return data

"""
Get the data for the model
"""
def get_data(freq="M"):
    # merge the different datasets to a unified timeseries
    county_timeseries = _get_county_data()
    weather_dsm = _get_weather_data()
    timeseries = county_timeseries.merge(weather_dsm, on='Year', how='left')
    timeseries = timeseries[['CORN, GRAIN - YIELD, MEASURED IN BU / ACRE', 'Avg Temp', 'Precip (Inches)']]

    # upsample
    timeseries = resample(timeseries, freq=freq)
    return timeseries


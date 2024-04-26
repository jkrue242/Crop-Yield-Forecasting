import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff
import plotly.io as pio

"""
Plotting function to visualize the dataset
"""
def plot_dataset(data):
    df = data.copy()
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    df.rename(columns={
        "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE": "Yield (bu/ac)",
        "Avg Temp": "Avg Temp (F)",
        "Precip (Inches)": "Precip (in)",
        }, inplace=True)
    plot = df.plot(x='Year')
    # save the plot
    plot.get_figure().savefig("images/dataset.png")
    

def plot_map(data, algorithm):
    state_county_fips = pd.read_csv("data/fips2county.tsv", sep="\t")
    state_names = state_county_fips["StateName"].tolist()
    county_names = state_county_fips["CountyName"].tolist()
    fips = state_county_fips["CountyFIPS"].tolist()

    state_names = [name.upper() for name in state_names]
    county_names = [name.upper() for name in county_names]
    
    keys = zip(state_names, county_names)
    county_fips = dict(zip(keys, fips))

    labels = []
    values = []

    for state1 in data.keys():
        for county1 in data[state1]:
            for pair in county_fips.keys():
                if state1 in pair[0] and county1 in pair[1]:
                    labels.append(str(county_fips[pair]))
                    values.append(int(data[state1][county1]))
    print(labels)
    print('=========')
    print(values)
    fig = ff.create_choropleth(
        fips=labels, values=values, scope=["IL", "IA", "MN", "MO", "NE", "ND", "SD", "WI"],
        county_outline={'color': 'rgb(0,0,0)', 'width': 0.5}, 
        state_outline={'color': 'rgb(0,0,0)', 'width': 1.0}, round_legend_values=True,
        legend_title='Cluster', title='Midwest Clustered by Yield'
    )
    pio.write_image(fig, f"images/{algorithm}_map.png")

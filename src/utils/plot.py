import matplotlib.pyplot as plt

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
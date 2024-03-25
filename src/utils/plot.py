import matplotlib.pyplot as plt

def plot_dataset(data):
    df = data.copy()
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    df["CORN, GRAIN - PRODUCTION, MEASURED IN BU"] = df["CORN, GRAIN - PRODUCTION, MEASURED IN BU"] * 10e-6
    df["CORN, GRAIN - ACRES HARVESTED"] = df["CORN, GRAIN - ACRES HARVESTED"] * 10e-4
    df.rename(columns={
        "CORN, GRAIN - PRODUCTION, MEASURED IN BU": "Production (x10e-6)",
        "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE": "Yield (bu/ac)",
        "CORN, GRAIN - ACRES HARVESTED": "Acres Harvested (x10e-4)",
        "Avg Price": "Avg Corn Price ($)",
        "Avg Temp": "Avg Temp (F)",
        }, inplace=True)
    plot = df.plot(x='Year')
    plot.get_figure().savefig("images/dataset.png")
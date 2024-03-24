import matplotlib.pyplot as plt

def plot_dataset(data):
    fig, axs = plt.subplots(3, 3, figsize=(16, 8))
    # first row
    axs[0][0].plot(data['Year'], data['Acres Planted'])
    axs[0][0].set_title('Acres Planted')

    axs[0][1].plot(data['Year'], data['Acres Harvested'])
    axs[0][1].set_title('Acres Harvested')

    axs[0][2].plot(data['Year'], data['Yield (bu/ac)'])
    axs[0][2].set_title('Yield (bu/ac)')

    # second row
    axs[1][0].plot(data['Year'], data['DE Avg Stock Price'])
    axs[1][0].set_title('DE Avg Stock Price')

    axs[1][1].plot(data['Year'], data['Precip'])
    axs[1][1].set_title('Precip')

    axs[1][2].plot(data['Year'], data['Snow'])
    axs[1][2].set_title('Snow')

    # third row
    axs[2][0].plot(data['Year'], data['Mean Temp'])
    axs[2][0].set_title('Mean Temp')

    axs[2][1].plot(data['Year'], data['GDD'])
    axs[2][1].set_title('GDD')

    plt.tight_layout()
    plt.savefig("images/dataset.png")
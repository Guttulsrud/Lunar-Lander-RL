import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import seaborn as sns


def plot_results(file_name):
    window_size = 5
    sns.set_style('white')
    sns.set_context("paper", font_scale=1)

    with open(f'../results/{file_name}', 'r') as f:
        r = json.load(f)

    dev_note = r['note']
    r = pd.json_normalize(r['results'])
    r['rolling_average'] = r['average_return'].rolling(window=window_size).mean()

    plt.plot(r['average_return'], label='Score')
    ax = sns.lineplot(r['average_return'])
    ax.axhline(200, color='red')
    ax.axhline(0, color='green')
    # plt.ylim(-2000, 400)
    plt.legend(loc='upper right', labels=['Score', 'Success'])
    ax.set(title=dev_note)
    ax.set(xlabel='Episodes', ylabel='Score')
    leg = ax.get_legend()
    leg.legendHandles[1].set_color('red')
    plt.show()
    fig = ax.get_figure()

    if not os.path.isdir('plots'):
        os.mkdir('plots')

    fig.savefig(f'plots/{dev_note}.png')


if __name__ == '__main__':
    file_name = os.listdir('../results')[-1]  # Plot the latest result
    plot_results(file_name)

import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import seaborn as sns


def plot_results(file_name):
    plt.style.use('ggplot')
    window_size = 5

    with open(f'../results/{file_name}', 'r') as f:
        r = json.load(f)

    r = pd.json_normalize(r['results'])

    r['rolling_average'] = r['average_return'].rolling(window=window_size).mean()

    plt.plot(r['average_return'], label='Average score')
    plt.plot(r['rolling_average'], label='Rolling average 5')
    ax = sns.lineplot(r['rolling_average'])
    # ax.set_ylim(-500, 300)

    ax.set(xlabel='Episodes', ylabel='Score')
    # Size of replay buffer? Since its gets full. 2% exploration?
    plt.show()
    fig = ax.get_figure()

    if not os.path.isdir('../img/'):
        os.mkdir('../img/')

    fig.savefig(f'../img/{file_name.split(".")[0]}.png')


if __name__ == '__main__':
    file_name = os.listdir('../results')[-2]

    plot_results(file_name)

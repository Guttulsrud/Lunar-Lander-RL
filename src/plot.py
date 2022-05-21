import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import seaborn as sns


def plot_results(file_name):
    # plt.style.use('ggplot')
    window_size = 10
    sns.set_style('white')
    sns.set_context("paper", font_scale=1)

    with open(f'../results/{file_name}', 'r') as f:
        r = json.load(f)

    # with open(f'../results/{file_name2}', 'r') as f:
    #     r2 = json.load(f)
    #
    # with open(f'../results/{file_name3}', 'r') as f:
    #     r3 = json.load(f)
    #



    dev_note = r['note']
    # dev_note = 'Naive approach - Rolling average of 10'

    r = pd.json_normalize(r['results'])
    # r2 = pd.json_normalize(r2['results'])
    # r3 = pd.json_normalize(r3['results'])
    r['rolling_average'] = r['average_return'].rolling(window=window_size).mean()
    # r2['rolling_average'] = r2['average_return'].rolling(window=window_size).mean()
    # r3['rolling_average'] = r3['average_return'].rolling(window=window_size).mean()

    plt.plot(r['average_return'], label='Score')
    # plt.plot(r2['average_return'], label='Position', color='red')
    # plt.plot(r3['average_return'], label='Wind')
    plt.plot(r['rolling_average'], label=f'Rolling {window_size}')
    # plt.plot(r2['rolling_average'], label='Position', color='orange')
    # plt.plot(r3['rolling_average'], label='Wind', color='green')
    ax = sns.lineplot(r['average_return'])
    # ax.set_ylim(-400, 400)
    ax.axhline(200, color='red')
    ax.axhline(0, color='pink')
    ax.axhline(100, color='green')
    # ax.axhline(-350, color='red')
    # ax.axhline(0, color='red')
    # plt.legend(loc='upper right', labels=['Gravity', 'Position', 'Wind', 'Success'])

    # ax.axhline(100, color='green')
    # ax.axhline(0, color='pink')
    ax.set(title=dev_note)
    ax.set(xlabel='Episodes', ylabel='Score')
    leg = ax.get_legend()
    leg.legendHandles[1].set_color('red')
    # Size of replay buffer? Since its gets full. 2% exploration?
    plt.show()
    fig = ax.get_figure()

    if not os.path.isdir('../img/'):
        os.mkdir('../img/')

    fig.savefig(f'../img/{dev_note}.png')
    fig.savefig(f'../img/eps/{dev_note}.eps')


if __name__ == '__main__':
    # file_name = os.listdir('../results')[-4]
    file_name = os.listdir('../results')[-1]
    # file_name2 = os.listdir('../results')[1]
    # file_name3 = os.listdir('../results')[2]

    plot_results(file_name)

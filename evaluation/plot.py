import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import seaborn as sns
import numpy as np


def plot_results(file_names):
    window_size = 5
    sns.set_style('white')
    sns.set_context("paper", font_scale=1)

    for i, file_name in enumerate(file_names):
        if 'THREAD2' not in file_name:
            continue
        with open(f'../results/{file_name}', 'r') as f:
            r = json.load(f)

        results = [np.average(x['simple_eval_scores']) for x in r['results']]
        print(i, r['hparams'], max(results))
        plt.plot(results, label=i)

    ax = plt.gca()
    ax.axhline(200, color='red', label='Success')
    ax.axhline(0, color='green', label='Failure')
    plt.legend(loc='upper right')
    ax.set(xlabel='Episodes', ylabel='Score')
    plt.title("Multiple Line Plot")
    plt.show()
    if not os.path.isdir('plots'):
        os.mkdir('plots')

    fig = ax.get_figure()
    fig.savefig(f'plots/multiple_lines.png')


if __name__ == '__main__':
    file_names = os.listdir('../results')
    plot_results(file_names)

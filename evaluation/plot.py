import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
import numpy as np


def gaussian_smoothing(data, sigma):
    kernel = [1. / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / (2 * sigma ** 2)) for x in range(-3, 4)]
    kernel /= np.sum(kernel)
    smoothed_data = np.convolve(data, kernel, mode='same')
    return smoothed_data


def plot_results(file_names):
    window_size = 5
    sns.set_style('white')
    sns.set_context("paper", font_scale=1)

    for i, file_name in enumerate(file_names):
        # if 'THREAD1' not in file_name:
        #     continue

        with open(f'{data_path}{file_name}', 'r') as f:
            r = json.load(f)

        if r['results'][0].get('episode_scores'):
            results = [np.median(x['episode_scores']) for x in r['results']]
        if r['results'][0].get('simple_eval_scores'):
            results = [np.median(x['simple_eval_scores']) for x in r['results']]

        results = [x if i < 200 else x if x > -100 else -100 for i, x in enumerate(results)]

        results = [np.max(results[i:i + window_size]) for i in range(len(results) - window_size)]
        results = gaussian_smoothing(results, 1)




        results = results[:322]
        if not len(results):
            continue


        print(i, max(results))
        # print(i, r['hparams'], max(results))
        label = {0: 'Proposed Agent', 1: 'Naive Agent'}[i]
        plt.plot(results, label=label)

    ax = plt.gca()
    ax.axhline(200, color='red')
    # ax.axhline(0, color='green', label='Failure')
    plt.legend(loc='upper left')
    ax.set(xlabel='Episodes', ylabel='Score')
    # set y range
    plt.ylim(-400, 300)
    # plt.title("Multiple Line Plot")
    plt.show()
    if not os.path.isdir('plots'):
        os.mkdir('plots')

    fig = ax.get_figure()

    fig.savefig(f'plots/multiple_lines.svg', )


data_path = '../results/'

if __name__ == '__main__':
    file_names = os.listdir(data_path)
    plot_results(file_names)

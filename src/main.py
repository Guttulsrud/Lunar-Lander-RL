from TrainingHandler import TrainingHandler
import random
import os
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_thread(thread, trials):
    hparams = {
        'learning_rate': random.choice([0.001]),
        'batch_size': random.choice([64]),
        'hidden_size': random.choice([128]),
        'layers': random.choice([2]),
        'activation': random.choice(['relu']),
        'timesteps': random.choice([2, 3, 4, 5, 6])
    }

    for trial in range(trials):
        print(f'Running thread: {thread + 1}, trial {trial + 1}')

        dev_note = f'Single thread {thread + 1} trial {trial + 1}'

        h = TrainingHandler(dev_note=dev_note, thread=thread, trial=trial, hparams=hparams)

        h.run(hparams)


if __name__ == '__main__':
    # n_threads = 3
    # n_trials = 2
    # threads = []

    dev_note = f'Single trial'

    hparams = {
        'learning_rate': random.choice([0.001]),
        'batch_size': random.choice([64]),
        'hidden_size': random.choice([128]),
        'layers': random.choice([2]),
        'activation': random.choice(['relu']),
        'timesteps': random.choice([5, 6])
    }

    h = TrainingHandler(dev_note=dev_note, hparams=hparams)

    h.run(hparams)

    # for thread in range(n_threads):
    #     t = threading.Thread(target=run_thread, args=(thread, n_trials))
    #     threads.append(t)
    #     t.start()
    #
    # # Wait for all threads to complete
    # for t in threads:
    #     t.join()

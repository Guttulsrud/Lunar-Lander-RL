from TrainingHandler import TrainingHandler
import random
import os
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_thread(thread, trials):
    hparams = {
        'learning_rate': random.choice([0.0001, 0.0003, 0.001, 0.003, 0.01]),
        'batch_size': random.choice([32, 64, 128]),
        'hidden_size': random.choice([64, 128, 256, 512]),
        'layers': random.choice([2, 3, 4]),
        'activation': random.choice(['relu', 'swish'])
    }

    for trial in range(trials):
        print(f'Running thread: {thread + 1}, trial {trial + 1}')

        dev_note = f'Single thread {thread + 1} trial {trial + 1}'

        h = TrainingHandler(dev_note=dev_note, thread=thread, trial=trial, hparams=hparams)

        h.run(hparams)


if __name__ == '__main__':
    n_threads = 4
    n_trials = 3
    threads = []

    for thread in range(n_threads):
        t = threading.Thread(target=run_thread, args=(thread, n_trials))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

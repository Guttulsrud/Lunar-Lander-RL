from TrainingHandler import TrainingHandler

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    dev_note = f'Single All'
    h = TrainingHandler(dev_note=dev_note)
    h.run_hpo()


from TrainingHandler import TrainingHandler
from utils import load_model

if __name__ == '__main__':
    path_to_model = ''
    dev_note = f'Testing {path_to_model}'

    h = TrainingHandler(dev_note=dev_note, pre_trained_model=load_model(path=path_to_model))
    h.run()


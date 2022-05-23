from TrainingHandler import TrainingHandler
from utils import load_model

if __name__ == '__main__':
    path_to_model = 'naive_all_uncertainties'
    model = load_model(path=path_to_model)

    h = TrainingHandler(pre_trained_model=model, testing=True)
    h.run()

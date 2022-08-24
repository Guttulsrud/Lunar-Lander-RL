from TrainingHandler import TrainingHandler
from src.utils import load_model

if __name__ == '__main__':
    dev_note = f'6t 128, 192 '
    h = TrainingHandler(dev_note=dev_note)
    h.run()


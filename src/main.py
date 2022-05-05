from Handler import Handler
from utils import load_model

if __name__ == '__main__':
    model_name = '22-05-04_19-47_SCORE_285'
    model = load_model(path=model_name)

    dev_note = f'Trained naive model: {model_name}. With gravity -5 and -15. Iterations: '
    h = Handler(dev_note=dev_note, pre_trained_model=model)
    h.run()

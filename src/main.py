from Handler import Handler
from utils import load_model

if __name__ == '__main__':
    # model_name = '22-05-07_02-35_SCORE_303'
    # model = load_model(path=model_name)
    model = None
    dev_note = f'Double timestep 256 no uncertainty. Iterations: '
    h = Handler(dev_note=dev_note, pre_trained_model=model)
    h.run()


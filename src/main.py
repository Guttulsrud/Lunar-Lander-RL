from Handler import Handler
from utils import load_model

if __name__ == '__main__':
    # model_name = '22-05-06_21-56_SCORE_-28'
    # model = load_model(path=model_name)
    model = None
    dev_note = f'Double timestep 128 topology. Iterations: '
    h = Handler(dev_note=dev_note, pre_trained_model=model)
    h.run()


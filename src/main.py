import yaml
from yaml import SafeLoader

from Agent import Agent
from Handler import Handler
from new_custom_lunar_lander import LunarLander
from utils import load_model

if __name__ == '__main__':
    # model_name = '22-05-21_05-05_SCORE_145'
    # model = load_model(path=model_name)
    model = None
    dev_note = f'Multi (256) - None'
    h = Handler(dev_note=dev_note, pre_trained_model=model)
    h.run()


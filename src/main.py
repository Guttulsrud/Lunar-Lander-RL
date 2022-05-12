from Handler import Handler
from utils import load_model

if __name__ == '__main__':
    # model_name = '22-05-07_02-35_SCORE_303'
    # model = load_model(path=model_name)
    model = None
    dev_note = f'Single(64) variable gravity.'
    h = Handler(dev_note=dev_note, pre_trained_model=model)
    h.run()

# 1. Single(64) no uncertainty.
# 2. Single(64) gravity
# 3. Single(64) position
# 4. Single(64) gravity + position
# 5. Single(64) wind
# 6. Single(64) gravity + position + wind


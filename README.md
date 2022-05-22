### Solving the Lunar Lander environment with multiple uncertainties using Reinforcement Learningand Deep Q Learning

## Installation

1. Make sure you have conda installed.
2. Run `conda create lunar-lander` to create new conda environment.
3. Run `conda activate lunar-lander` to active environment.
4. Run `conda install swig`.
5. Run `pip install -r requirements.txt`.

* Make sure you use Python 3.7
* Make sure Tensorflow version is 2.8.0 (Important to load pre-trained models)
--------------------



## How to run

* Make sure to check out the config file if you wish to customize the training, model, etc.
* Rendering of training/evaluation is turned off by default. May be turned on in config file.

### Pre-trained model

1. Open `src/test.py`
2. Change the `path_to_model` variable to the name of a model in the `saved_models` folder.
3. Make sure `model_type` is configured correctly in the config file.
4. Run the file.

### Train new model

1. Open`src/main.py`.
2. (Optional) Change the `dev_note` variable to describe the run.
3. Run the file.


--------------------

## Plotting

- It is possible to plot the results from training and evaluation. In the `evaluation` folder, run `plot.py` to
  visualize the latest model run. A PNG figure gets saved to the plots folder. The filename is the same as the dev note
  from the results.

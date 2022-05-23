### Solving the Lunar Lander environment with multiple uncertainties using Reinforcement Learningand Deep Q Learning

## Installation

1. Make sure you have conda installed.
2. Run `conda create --name lunar-lander` to create new conda environment.
3. Run `conda activate lunar-lander` to active environment.
4. Run `conda install swig`.
5. Run `pip install -r requirements.txt`.


Important to load pre-trained models:
* Make sure you use Python version 3.7 
* Make sure you use Tensorflow version 2.8.0 
--------------------


## How to run

* Make sure to check out the config file if you wish to customize the training, model, etc.
* Rendering of training is turned off by default. May be turned on in config file.

### Train new model

1. Open`src/main.py`.
2. (Optional) Change the `dev_note` variable to describe the run.
3. Run the file.


### Pre-trained model

1. Open `src/test.py`
2. Change the `path_to_model` variable to the name of a model in the `saved_models` folder.
3. Make sure `model_type` is configured correctly in the config file.
4. Run the file.
5. If an error appears, validate that you have Python version 3.7 and Tensorflow version 2.8.0 installed. 


--------------------

## Plotting

- It is possible to plot results. In the `evaluation` folder, run `plot.py` to
  visualize the latest model run. A PNG figure gets saved to the plots folder. The filename is the same as the dev note
  from the results.

### Solving the Lunar Lander environment with multiple uncertainties using Reinforcement Learning and Deep Q Learning

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

1. Navigate to `src` working directory. 
2. Open`main.py`.
3. (Optional) Change the `dev_note` variable to describe the run.
4. Run the file. Make sure you are in the `src` working directory.


### Pre-trained model

1. Navigate to `src` working directory. 
2. Open`test.py`.
3. Change the `path_to_model` variable to the name of a model in the `saved_models` folder.
4. Make sure `model_type` is configured correctly in the config file.
5. Run the file. Make sure you are in the `src` working directory.
6. If an error appears, validate that you have Python version 3.7 and Tensorflow version 2.8.0 installed. 


--------------------

## Plotting

- It is possible to plot results. In the `evaluation` folder, run `plot.py` to
  visualize the latest model run. A PNG figure gets saved to the plots folder. The filename is the same as the dev note
  from the results.

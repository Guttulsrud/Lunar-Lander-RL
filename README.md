### Solving the Lunar Lander environment with multiple uncertainties using Reinforcement Learningand Deep Q Learning





HUSK å få sjekke TF versjon slik at modelller kan lastes!

## Installation
1. Create and activate new conda environment
2. Run `conda install swig`
3. Run `pip install -r requirements.txt`
--------------------

## How to run
4. (Optional) Change parameters in config.yml
5. Open `test.py`, change the variable `path_to_model` to the 
   name of a model in the models folder. Run the file
   
6. Or run `main.py` to train a new model. Rendering of training/evaluation is turned off by default. May be turned on in config file.

--------------------

## Plotting

- It is possible to plot the results from training and evaluation. In the `evaluation` folder, run `plot.py` to visualize the latest model run. A PNG figure gets saved to the plots folder. The filename is the same as the dev note from the results.

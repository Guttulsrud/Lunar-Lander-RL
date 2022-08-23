import os

for file in os.listdir('saved_models'):
    if int(file.split('_')[-1]) > 150:
        print(file)
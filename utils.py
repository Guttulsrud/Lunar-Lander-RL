import numpy as np
import json


def reverse_one_hot(value, length):
    output = np.zeros(length)
    output[value] = 1.0
    return output


def get_config():
    f = open('../config.json')
    config = json.load(f)
    return config

import matplotlib.pyplot as plt
import json
import os
import pandas as pd

file_name = os.listdir('results')[-1]

plt.style.use('ggplot')
window_size = 5

with open(f'results/{file_name}', 'r') as f:
    r = json.load(f)
r = pd.json_normalize(r['results'])
r['rolling_average'] = r['average_return'].rolling(window=window_size).mean()

plt.plot(r['average_return'], label='Score')
plt.plot(r['rolling_average'], label='Mean 5')
plt.show()

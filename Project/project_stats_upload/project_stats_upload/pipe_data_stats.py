# This is pipeline script. Run pipelines in order:
# Order : 2

import pandas as pd
import numpy as np
from scipy.fft import fft, fftshift, rfft, fftfreq

seg_ids = "../../predict-volcanic-eruptions-ingv-oe/train.csv"
raw_data_path = "../../predict-volcanic-eruptions-ingv-oe/fft_train/"


#  Create new input data
df = pd.read_csv(seg_ids)
data_path = raw_data_path
index = 0
for seg_id in df['segment_id']:
    sensor_reading = pd.read_csv(data_path + str(seg_id) + ".csv", dtype=float, index_col=0)
    sensor_reading = sensor_reading[:1000]
    x = sensor_reading.describe()
    for i in range(1, 11):
        df.loc[index, 'sensor_' + str(i) + '_mean'] = x['sensor_' + str(i)]['mean']
        df.loc[index, 'sensor_' + str(i) + '_std'] = x['sensor_' + str(i)]['std']
    print(index)
    index += 1

df.to_csv('../../predict-volcanic-eruptions-ingv-oe/fft_stats/dataset2.csv')
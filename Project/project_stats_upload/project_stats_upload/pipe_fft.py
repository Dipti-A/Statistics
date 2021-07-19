# This is pipeline script. Run pipelines in order:
# Order : 1

import pandas as pd
import numpy as np
from scipy.fft import fft, fftshift, rfft, fftfreq

seg_ids = "../../predict-volcanic-eruptions-ingv-oe/train.csv"
raw_data_path = "../../predict-volcanic-eruptions-ingv-oe/train/"


def raw_data(index_data, data_path):
    """
    Description: Fills na values with zero.
    :param index_data: Meta data file for signal data.
    :param data_path: Signal data location.
    :return: Dict containing dataframes and seg id as keys.
    """
    raw_sensor_data = {}
    for seg_id in index_data['segment_id']:  # For testing run index_data['segment_id'][:1]
        sensor_reading = pd.read_csv(data_path + str(seg_id) + ".csv", dtype=float)
        sensor_reading.fillna(0, inplace=True)
        raw_sensor_data[str(seg_id)] = sensor_reading
    return raw_sensor_data


def fft_raw_data(data):
    """
    Transforms signal data using fft of time to frequency.
    :param data:
    :return: Dict containing dataframes of fft data and seg id as keys.
    """
    fft_raw_data = {}
    for sname, sdata in data.items():
        fft_sdata = {c_names: np.abs(fft(np.asarray(sdata[c_names]))) for c_names in sdata.columns}
        fft_raw_data[sname] = fft_sdata
    return fft_raw_data


#  PRE PRE PROCESS DATA
train = pd.read_csv(seg_ids)

data = raw_data(index_data=train, data_path=raw_data_path)
data = fft_raw_data(data)

for sname, sdata in data.items():  # fft data
    df = pd.DataFrame(sdata)
    df.to_csv('../../predict-volcanic-eruptions-ingv-oe/fft_train/'+sname+'.csv')
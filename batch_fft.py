import numpy as np
import matplotlib.pyplot as plt
import os 
import tqdm
from corrections import pattern_correction, lowpass


def data_reduction(directory, number_data:int=None):
    ''' 
    Function that handles the data reduction of the correlation data. This includes the pattern correction, lowpass filter, normalisation and the weighted average.
    This function assumes the data is named 'measurement_XXXXX.fcorr' where XXXXX is the number of the data.
    Takes:
        - directory: str, the directory where the files are stored.
        - number_data: int, all files from 0 to number_data will be used. Default is None, which means all files in the directory will be used.
    Returns:
        - average_g2: np.array, the corrected weighted and nomralised average of all the data. 
    '''
    if number_data is None:
        number_data = count_files_in_directory(directory)
    sum = np.zeros(10000)
    weight_sum = 0
    for n in tqdm(range(number_data), desc='Do data reduction'): 
        path = directory + '\measurement_' + str(n).zfill(5) + '.fcorr'
        data = np.loadtxt(path)
        data = pattern_correction(data)
        weight = np.std(data)**-2
        sum += data*weight
        weight_sum += weight
    average_g2 = sum/weight_sum
    return lowpass(average_g2)

def count_files_in_directory(directory):
    ''' 
    Function that counts the number of files in a given directory. 
    '''
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

t_width=1.6

measurement_name = '20240605_10x10test'
average = data_reduction('D:\\' + measurement_name)
fourier = np.fft.rfft(average)
frequencies = np.fft.rfftfreq(len(average), t_width*1e-9)
fig,ax = plt.subplots(figsize=(10, 5))
ax.plot(frequencies, np.abs(fourier))
ax.set(xlabel='Frequency (GHz)', ylabel='Amplitude', yscale='log')
plt.show()
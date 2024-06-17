import numpy as np
import os
from tqdm import tqdm
#from corrections import pattern_correction, lowpass


from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
import scipy.signal as scs
def lowpass(data, cutoff = 0.2): # cutoff in GHz    
    fs = 1./1.6
    nyq = 0.5*fs
    order = 2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    data = filtfilt(b, a, data)
    return data
def pattern_correction(data): # remove 8 bin periodicity in raw G2 functions
	corrector = np.zeros(8) # intialize 8 bin patter to be filled
	for i in range(0, 4000):
		corrector[i % 8] += data[i] * 1. / 500. # Overlay 8 bin patterns and average over them
	
	# Apply correction to data
	datacor = []
	for i in range(len(data)):  # apply 8 bin correction
		datacor.append(data[i] / corrector[i % 8]) # datacor is also normalized by this step
	datacor = np.array(datacor) # For some reason
	return datacor

class Data_reduction:
    
    def __init__(self, raw_data_directory:str, reduced_data_directory:str, number_data:int=None, sampling_time:float=1.6e-9):
        self.raw_dir = raw_data_directory
        self.reduced_dir = reduced_data_directory
        self.n_data = number_data
        self.sampling_time = sampling_time
        
        self.time_arr = np.arange(-5000*self.sampling_time, 5000*self.sampling_time, self.sampling_time)
        self.av_g2 = None   
            
    def _data_reduction(self):
        ''' 
        Helper function that handles the data reduction of the correlation data. This includes the pattern correction, lowpass filter, normalisation and the weighted average.
        This function assumes the data is named 'measurement_XXXXX.fcorr' where XXXXX is the number of the data. 
        '''
        if self.n_data is None:
            self.n_data = self._count_files_in_directory(self.raw_dir)
        sum = np.zeros(10000)
        weight_sum = 0
        for n in tqdm(range(self.n_data), desc='Do data reduction'): 
            path = self.raw_dir + '\measurement_' + str(n).zfill(5) + '.fcorr'
            data = np.loadtxt(path)
            data = pattern_correction(data)
            weight = np.std(data)**-2
            sum += data*weight
            weight_sum += weight
        average_g2 = sum/weight_sum
        self.av_g2 = lowpass(average_g2)

    def _count_files_in_directory(self, directory):
        ''' 
        Helper function that counts the number of files in a given directory. 
        '''
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

    def get_average_g2(self, repeat_calculation:bool=False):
        '''
        Main function that should be called. It gets the average g2 data from the raw data and saves the result to the specified reduced_dir.
        Takes: 
            - repeat_calculation: bool, optional; if True the data reduction will be done even if the reduced data already exists. Default is False.
        '''
        suffix = 'average' +'_'+ str(self.n_data) # This is the suffix of the reduced data file. It is used to distinguish between different data reduction methods.
        # Load time intensive data reduction from saved data if already done and the user does not want to repeat the calculation:
        if os.path.exists(self.reduced_dir + '\\' + suffix + '.txt') and not repeat_calculation:
            self.av_g2 = np.loadtxt(self.reduced_dir + '\\' + suffix + '.txt')
        # Otherwise do the data reduction:        
        else:  
            self._data_reduction()
            # Make the reduced data directory if it does not exist and save the data:
            if not os.path.exists(self.reduced_dir):
                os.makedirs(self.reduced_dir)
            np.savetxt(self.reduced_dir + '\\' + suffix + '.txt', self.av_g2)
         
    def batch_fft(self):
        '''
        Function that calculates the Fourier transform of the average g2 data. 
        Returns:
            - freq: np.array, the frequency array of the Fourier transform.
            - fft_amplitude: np.array, the amplitude of the Fourier transform of the average g2 data.
        '''
        if self.av_g2 is None:
            self.get_average_g2()
        fft = np.abs(np.fft.rfft(self.av_g2))
        freq = np.fft.rfftfreq(len(self.av_g2), self.sampling_time)
        return freq, fft
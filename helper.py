import numpy as np
import os
from tqdm import tqdm
from corrections import pattern_correction, lowpass


class Data_reduction:
    
    def __init__(self, raw_data_directory:str, reduced_data_directory:str, number_data:int=None, sampling_time:float=1.6e-9) -> None:
        self._raw_dir = raw_data_directory
        self._reduced_dir = reduced_data_directory
        self._n_data = number_data
        self._sampling_time = sampling_time
        
        self.time_arr = np.arange(-5000*self._sampling_time, 5000*self._sampling_time, self._sampling_time)
        self.av_g2 = None   
            
    def data_reduction(self, repeat_calculation:bool=False) -> None:
        '''
        Method that calculated the average g2 from the raw data and saves the result to the specified _reduced_dir.
        
        Takes: 
            - repeat_calculation: bool, optional; if True the data reduction will be done even if the reduced data already exists. Default is False.
        '''
        # If the user did not specify the number of data to reduce, use all data in the directory:
        if self._n_data is None:
            self._n_data = self._count_files_in_directory(self._raw_dir)
        filename = 'average' +'_'+ str(self._n_data) # This is the name of the reduced data file. 
        # Load data reduction from saved data if already done and the user does not want to repeat the calculation:
        if os.path.exists(self._reduced_dir + '\\' + filename + '.txt') and not repeat_calculation:
            self.av_g2 = np.loadtxt(self._reduced_dir + '\\' + filename + '.txt')
        # Otherwise do the data reduction:        
        else:  
            self._set_average_g2()
            # Create the reduced data directory if it does not exist and save the data:
            if not os.path.exists(self._reduced_dir):
                os.makedirs(self._reduced_dir)
            np.savetxt(self._reduced_dir + '\\' + filename + '.txt', self.av_g2)
    
    # (This should probably be in its own utils class that inherits from Data_reduction:)     
    def batch_fft(self, repeat_calculation:bool=False) -> tuple:
        '''
        Method that calculates the Fourier transform of the average g2 data. 
        
        Takes:
            - repeat_calculation: bool, optional; if True the preliminary data reduction will be done even if the reduced data already exists. Default is False.
            
        Returns:
            - freq: np.array, the frequency array of the Fourier transform.
            - fft_amplitude: np.array, the amplitude of the Fourier transform of the average g2 data.
        '''
        if (self.av_g2 is None) or repeat_calculation:
            # The calculation of the average g2 is not done yet or the user wants it to be repeated.
            self.data_reduction(True)     
        fft_amplitude = np.abs(np.fft.rfft(self.av_g2))
        freq = np.fft.rfftfreq(len(self.av_g2), self._sampling_time)
        return freq, fft_amplitude
    
    def _set_average_g2(self) -> None:
        ''' 
        Helper method that handles the data reduction of the correlation data. This includes the pattern correction, lowpass filter, normalisation and the weighted average.
        This method assumes the data is named 'measurement_XXXXX.fcorr' where XXXXX is the number of the data. 
        '''
        sum = np.zeros(10000)
        weight_sum = 0
        for n in tqdm(range(self._n_data), desc='Do data reduction'): 
            path = self._raw_dir + '\measurement_' + str(n).zfill(5) + '.fcorr'
            # Check if the file is missing due to errors in the correlation. If so, skip the file:
            if not os.path.exists(path):
                continue
            data = np.loadtxt(path)
            data = pattern_correction(data)
            weight = np.std(data)**-2
            sum += data*weight
            weight_sum += weight
        average_g2 = sum/weight_sum
        self.av_g2 = lowpass(average_g2)

    def _count_files_in_directory(self, directory) -> int:
        ''' 
        Helper function that counts the number of files in a given directory. 
        '''
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

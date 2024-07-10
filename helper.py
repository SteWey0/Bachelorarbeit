import numpy as np
import os
from tqdm import tqdm
from corrections import pattern_correction, lowpass


class Data_reduction:
    
    def __init__(self, raw_data_directory:str, reduced_data_directory:str=None, number_data:int=None, sampling_time:float=1.6e-9) -> None:
        '''
        Initialises the Data_reduction object. The object is used to reduce the raw data to an average g2 and to calculate the Fourier transform of the average g2.
        
        Takes:
            - raw_data_directory: str, the path to the directory containing the raw data.
            - reduced_data_directory: str, optional; the path to the directory where the reduced data will be saved. Default is None, which means the reduced data will be saved at location 'reduced_data/raw_data_dir'.
            - number_data: int, optional; the number of data to reduce. Default is None, which means all data in the directory will be reduced.
            - sampling_time: float, optional; the sampling time of the data. Default is 1.6ns.
        '''
        self._raw_dir = os.path.normpath(raw_data_directory)
        if reduced_data_directory is None:
            self._reduced_dir = os.path.join('reduced_data', os.path.basename(self._raw_dir))
        else:
            self._reduced_dir = reduced_data_directory    
        if number_data is None:
            self._n_data = len(self._get_file_path_list(self._raw_dir))
        else:
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
        # Set the name of the file where the average g2 will be saved:
        filename = 'average' +'_'+ str(self._n_data)  + '.txt'  
        # Load data reduction from saved data if already done and the user does not want to repeat the calculation:
        if os.path.exists(os.path.join(self._reduced_dir, filename)) and not repeat_calculation:
            self.av_g2 = np.loadtxt(os.path.join(self._reduced_dir, filename))
        # Otherwise do the data reduction:        
        else:  
            self._set_average_g2()
            # Create the reduced data directory if it does not exist and save the data:
            if not os.path.exists(self._reduced_dir):
                os.makedirs(self._reduced_dir)
            np.savetxt(os.path.join(self._reduced_dir, filename), self.av_g2)
    
    # (This should probably be in its own utils class that inherits from Data_reduction, together with later integration)     
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
        # Take the first n_data files in the raw data directory (or all if n_data is None):
        file_path_list = self._get_file_path_list(self._raw_dir)[:self._n_data]
        if os.path.splitext(file_path_list[0])[1] == '.npy':
            use_npy = True
        for path in tqdm(file_path_list, desc='Do data reduction'): 
            if use_npy:
                data = np.load(path) # Use faster loading of .npy files if possible
            else:
                data = np.loadtxt(path) # Kept there for compatibility with old .fcorr txt-data
            data = pattern_correction(data)
            weight = np.std(data)**-2
            sum += data*weight
            weight_sum += weight
        average_g2 = sum/weight_sum
        self.av_g2 = lowpass(average_g2)

    def _get_file_path_list(self, directory) -> int:
        ''' 
        Helper mehtod that returns a list of all paths to files in a given directory. 
        '''
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    def _return_mean_pulseshapes(self) -> tuple:
        '''
        Helper method that returns the mean PMT-pulseshapes of the data. Expects them to be saved in raw_data_directory/chX/calib.shape1 where X is either 0  or 1.
        
        Returns:
            - ch0, ch1: np.array, the mean PMT-pulseshapes of channel 0 and 1 respectively.
        '''
        ch0 = np.loadtxt(os.path.join(self._raw_dir, 'ch0', 'calib.shape1'))
        ch1 = np.loadtxt(os.path.join(self._raw_dir, 'ch1', 'calib.shape1'))
        return ch0, ch1
        

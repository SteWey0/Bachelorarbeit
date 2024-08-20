import numpy as np
import os
from tqdm import tqdm
from corrections import *
from scipy.optimize import curve_fit
from scipy.signal import correlate
from scipy.interpolate import interp1d

class Data_reduction:
    
    def __init__(self, raw_data_directory:str, reduced_data_directory:str=None, number_data:int=None, sampling_time:float=1.6e-9) -> None:
        '''
        Initialises the Data_reduction object. The object is used to reduce the raw data to an average g2 and to calculate the Fourier transform of the average g2. Support for .npy files is included, 
        while maintaining compatibility with the txt-data format (such as .fcorr).
        
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
            self._n_data = len(self._return_file_path_list(self._raw_dir))
        else:
            self._n_data = number_data
        self._sampling_time = sampling_time
        
        self.time_arr = np.arange(-5000*self._sampling_time, 5000*self._sampling_time, self._sampling_time)
        self.av_g2 = None   
            
    def data_reduction(self, repeat_calculation:bool=False) -> None:
        '''
        Method that calculates the average g2 from the raw data and saves the result to the specified _reduced_dir.
        
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
    
    def integrate(self, return_infodict:bool=False, plot_fits:bool=False, external_plotting:bool=False):
        '''
        Method that calculates the integral of the bunching peak in the average g2 data and its error. The error on the integral is calculated by shifting the bunching peak to different positions, fitting and integrating the fit for each shift. 
        The standard deviation of these integrals is the error on the integral.
        The fitting function is a convolution of the correlated mean pulse shape and a Gaussian smooting kernel that contains information about the time resolution of the measurement.
        
        Takes: 
            - return_infodict: bool, optional; if True additional information will be returned. Default is False. 
            - plot_fits: bool, optional; if True some fits will be plotted for debugging purposes. Default is False.
            - external_plotting: bool, optional; if True the shifted g2 data, fits and integrals and corresponding time shifts will be set as class variables for external plotting. Default is False.
            
        Returns:     
            - If return_infodict, a dictionary containing the following will be returned:
                - integral: tuple, the integral of the bunching peak and its error.
                - fit_params: list, contains integral fit parameters and errors. fit_params[i] = [popt[i], pcov[i,i]**0.5] where i is amplitude, x_shift, sigma.
                - lf_pattern: np.array, the lowpass filter pattern that was subtracted from the average g2 data.
        '''
        # ----------- (Changing these values will need changes in for loop!) -------------
        fitting_width = 100 # fit data in +/- 100 bins (160ns) around peak
        integration_width = 100  # integrate data +/- 25 bins (40ns) around peak
        # --------------------------------------------------------------------------------
        self._set_correlated_pulse_function() # Sets self.corr_func, needed for fitting
        g2, t = self.av_g2, self.time_arr
        lf_pattern = lowpass(g2, 0.60e-3) # 60kHz for lf pattern
        g2 = g2 - lf_pattern
        self.av_g2_corr = g2
        peak_pos = np.argmax(g2)
        # Cut peak from data
        t_fit = t[peak_pos-fitting_width:peak_pos+fitting_width]
        g2_fit = g2[peak_pos-fitting_width:peak_pos+fitting_width]
        
        # ----------------- Fit and integration of bunching peak -------------------------
        p0 = [6e-6, t[peak_pos], 1e-9]
        popt, pcov  = curve_fit(self._fit_func, t_fit, g2_fit, p0=p0)
        fit = self._fit_func(t[peak_pos-integration_width : peak_pos+integration_width], *popt)
        integral = np.trapz(self._fit_func(np.arange(t[peak_pos-integration_width], t[peak_pos+integration_width],0.1e-9), *popt), dx=0.1e-9)

        if plot_fits:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('viridis')
            fig,ax = plt.subplots()  
            ax.plot(t*1e9, g2)
            ax.plot(t_fit*1e9, self._fit_func(t_fit, *popt), color='r')
            ax.set(xlabel='Time [ns]', ylabel='g2')
            
        # -------------------- Get error on integral -------------------------------------
        # Array of all shifts relative to peak_pos in bins. Centers around peak_pos so that shift_range ~ [-3750:peak_pos_bin-2*fitting_width, peak_pos_bin+2*fitting_width:3750]. This way the region around peak_pos is always excluded.
        # peak_pos is the index of the peak in g2 and not its time bin. To get a time bin we must therefore use peak_pos-5000 as len(g2)=10000.
        # Np.flip is used to make the array symmetric around peak_pos.
        shift_range = np.append(np.flip(np.arange(peak_pos-5000-2*fitting_width, -3750, -25)), np.arange(peak_pos-5000+2*fitting_width, 3750,25)).astype(int)
        # Start integration for all shifts in shift_range
        I = np.zeros_like(shift_range, dtype=float)    # Array where I[n] is the integral for n=shift_range[n]
        # Other stuff for external plottin gwhich will end up in infodict
        shifted_g2 = np.zeros((len(shift_range),200), dtype=float)   
        fits = np.zeros((len(shift_range),200), dtype=float)  
        t_shifted = np.zeros((len(shift_range),200), dtype=float)
        for n, shift in enumerate(shift_range):
            # Get start and stop indices for fit for each shift
            start_index = len(t)//2 + shift
            stop_index = start_index + 2*integration_width
            # Make array of size 2*fitting_width, place fit at centre, add to g2 -> shift_g2: 
            fit_arr = np.zeros(200)
            fit_arr[:] = fit
            shift_g2 = np.copy(g2[start_index-0:stop_index+0]) # Needed, else g2 is modified
            shift_g2 += fit_arr
            
            # Fitting + Integration: 
            peak_pos = t[start_index-0:stop_index+0][np.argmax(shift_g2)] # peak position of shifted fit in s, used for fit
            popt_err, _ = curve_fit(self._fit_func, t[start_index-0:stop_index+0], shift_g2, p0=[6e-6, peak_pos, 1e-9])
            I[n] = np.trapz(self._fit_func(np.arange(t[start_index],t[stop_index],0.1e-9), *popt_err), dx=0.1e-9)
            if external_plotting:
                shifted_g2[n,:] = shift_g2
                fits[n,:] = self._fit_func(t[start_index:stop_index], *popt_err)
                t_shifted[n,:] = t[start_index:stop_index]
            if plot_fits and n%5==0:
                ax.plot(t[start_index-0:stop_index+0]*1e9, shift_g2, color=cmap(n/len(shift_range)))
                ax.plot(t[start_index:stop_index]*1e9, self._fit_func(t[start_index:stop_index], *popt_err), color='k', linestyle='--')
        self.integral = np.array([integral, np.std(I)])
        
        if external_plotting:
            self.shift_range = shift_range 
            self.err_integrals = I
            self.shifted_g2 = shifted_g2
            self.shifted_fits = fits
            self.t_shifted = t_shifted
        if return_infodict:
            return {'integral': (integral, np.std(I)), 'fit_params': popt, 'fit_params_error': np.diag(pcov)**0.5, 'lf_pattern': lf_pattern}

            
    def _fit_func(self, x, amplitude, x_shift, sigma):
        '''
        Helper method that contains the fitting function for the bunching peak.
        '''
        samples = np.arange(-50,50,1)*self._sampling_time     # Samples for interpolation
        f = self._corr_func(samples)                          # Interpolated, correlated mean pulse shape
        g = np.exp(-(samples)**2/(2*sigma**2))                # Gaussian smoothing kernel
        conv = np.convolve(f,g,mode='same')                   # Convolve, then interpolate for function thats defined for all x
        interp = interp1d(samples, conv, fill_value=0, kind='linear', bounds_error=False)  
        return amplitude * interp(x-x_shift)/np.max(interp(x-x_shift)) # Normalisation makes amplitude par. of fit more meaningful
    
    def _set_correlated_pulse_function(self) -> None:
        '''
        Helper method that sets the variable self._corr_func, which is the correlated, interpolated mean pulse shape. Used as a part in fitting the g2 data.
        '''
        shape0, shape1 = self._return_mean_pulseshapes()
        pulse = correlate(shape1[:,1], shape0[:,1], mode='same')
        # Normalisation and shifting the peak to t=0 for easier fitting:
        pulse= pulse/np.max(pulse) 
        x = np.arange(0,211)*self._sampling_time
        x = x - x[np.argmax(pulse)]
        # Interpolation:
        self._corr_func = interp1d(x, pulse, fill_value=0.0, kind='linear', bounds_error=False)
        
    def _set_average_g2(self) -> None:
        ''' 
        Helper method that handles the data reduction of the correlation data. This includes the pattern correction, lowpass filter, normalisation and the weighted average.
        '''
        sum = np.zeros(10000)
        weight_sum = 0
        # Take the first n_data files in the raw data directory (or all if n_data is None):
        file_path_list = self._return_file_path_list(self._raw_dir)[:self._n_data]
        if os.path.splitext(file_path_list[0])[1] == '.npy':
            use_npy = True
        else:
            use_npy = False
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
        
    
    def _return_file_path_list(self, directory) -> int:
        ''' 
        Helper method that returns a list of all paths to files in a given directory. 
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

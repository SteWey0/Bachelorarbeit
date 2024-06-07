from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
import scipy.signal as scs
from matplotlib import pyplot as plt

def gauss_filter(data):
	w = 128
	datagaus = gaussian_filter1d(data, w)
	data_i = data / datagaus
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


def gauss(x, a, m, s, d):
	return a * np.exp(-(x - m) ** 2 / 2 / s / s) + d
def gauss_fit_filter(data):
	xdata = np.arange(0,len(data),1)
	popt, pcov = curve_fit(gauss, xdata, data)
	datacor = []
	for i in range (0,len(data)):
		datacor.append(data[i]/gauss(xdata[i], *popt))
	return datacor

def lowpass(data, cutoff = 0.2): # cutoff in GHz    
    fs = 1./1.6
    nyq = 0.5*fs
    order = 2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    data = filtfilt(b, a, data)
    return data

def notch(data, f0, Q): # f0=frequency to be removed from signal (Hz), Q=quality factor
	fsamp = 625e6 # sample frequency = bin sampling 1.6ns (Hz)
	# Design notch filter
	b, a = scs.iirnotch(f0, Q, fsamp)
	#freq, h = scs.freqz(b, a, fs=fsamp)
	#plt.plot(freq*fsamp/(2*np.pi), 20 * np.log10(abs(h)))
	#plt.show()
	#plt.close()
	# apply notch filter
	new_data = scs.filtfilt(b, a, data)
	return new_data
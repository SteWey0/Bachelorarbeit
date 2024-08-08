import numpy as np
import matplotlib.pyplot as plt

# **** Note for future users: if you believe in the code, you just need
# **** to add the spectra you want to calculate the coherence time from
# **** in the section "Adding the stuff"

def gauss(x,a,m,s):
	return a * np.exp(-(x-m)**2/2/s/s)
def column(matrix, i):
    return [row[i] for row in matrix]
def linear_interpolation(x_left, y_left, x_right, y_right, x_between):
	slope = (y_right-y_left)/(x_right-x_left)
	y_between = y_left + slope*(x_between-x_left)
	return y_between
def integral(y,stepsize):
	integral = 0.
	for i in range (0,len(y)):
		integral += stepsize * y[i]
	return integral
def g2_integral(y,stepsize):
	integral = 0.
	for i in range (0,len(y)):
		integral += stepsize * (y[i]-1.)
	return integral

def normalize_y(y_in):
	integral = 0.
	for i in range (0, len(y_in)):
		integral += y_in[i]
	y_out = []
	for i in range (0, len(y_in)):
		y_out.append(y_in[i]/integral)
	return y_out

def normalized_spectrum(x_in, y_in):
	dx = x_in[1]-x_in[0]
	integral = 0.
	for i in range(0,len(x_in)):
		integral += dx * y_in[i]
	y_out = []
	for i in range(0,len(x_in)):
		y_out.append(y_in[i]/integral)
	return x_in, y_out

def densify_spectrum(x_in, y_in, factor=10):
	dx = (x_in[1]-x_in[0])/10
	x_out = []; y_out = []
	for i in range(0,len(x_in)-1):
		x_base = x_in[i]; x_step = x_in[i+1]
		for j in range (0,int(factor)):
			x = x_base + j*dx
			y = linear_interpolation(x_left=x_in[i], y_left=y_in[i], x_right=x_in[i+1], y_right=y_in[i+1], x_between=x)
			x_out.append(x); y_out.append(y)
	return x_out, y_out

def extend_spectrum(x_in, y_in, factor=10):
	dx = x_in[1]-x_in[0]
	x_out = []; y_out = []
	for i in range(0,len(x_in)):
		x_out.append(x_in[i])
		y_out.append(y_in[i])
	# Big wavelength side
	for i in range (0,int(factor)*len(x_in)):
		x_out.append(x_out[-1]+dx)
		y_out.append(0)
	# Small wavelength side
	for i in range (0,int(factor)*len(x_in)):
		if x_in[0]-i*dx > 100:# don't put too small wavelengths
			x_out.insert(0,x_in[0]-i*dx)
			y_out.insert(0,0)
	return x_out, y_out

def angular_frequency_spectrum(lambdas, spec_y):
	x = lambdas; y = spec_y
	spec_freqs = []
	for i in range (0, len(x)):
		omega = 2*np.pi*299792459/(1e-9*x[i]) # in Hz
		spec_freqs.append([omega, y[i]])
	spec_freqs = sorted(spec_freqs, key=lambda a_entry: a_entry[0])
	out_x = column(spec_freqs,0); out_y = column(spec_freqs,1)
	return out_x, out_y

def constant_x_axis_spectrum(x_in, y_in):
	x_out = np.linspace(x_in[0],x_in[-1],len(x_in))
	y_out = []; i_in_recent = 0
	for i_out in range(0,len(x_out)):
		# Find the neighbours		
		i_in = i_in_recent
		while x_in[i_in] < x_out[i_out] and i_in < len(x_in):
			i_in += 1
		if i_in-1 >= 0:
			left_neighbour = i_in-1
			right_neighbour = i_in
		else:
			left_neighbour = i_in
			right_neighbour = i_in+1
		i_in_recent = left_neighbour
		y_out.append(linear_interpolation(x_left=x_in[left_neighbour],y_left=y_in[left_neighbour],x_right=x_in[right_neighbour],y_right=y_in[right_neighbour],x_between=x_out[i_out]))
	return x_out, y_out

def fft_x_axis(x_data):
	return np.linspace(0, 2.*np.pi/(x_data[1]-x_data[0]), len(x_data), endpoint=True)

def make_g2(g1):
	g2 = []
	for i in range (0,len(g1)):
		g2.append(1. + (np.abs(g1[i]))**2)
	return g2

def stepsize(x_data):
	return 2.*np.pi/(x_data[1]-x_data[0])/len(x_data)

def add_spectrum(spectrum_x, spectrum_y, name, pcolor, pmarker="", plinestyle="-", plinewidth=1):
	# Read in wavelength spectrum data
	# sort wavelengths
	xy = sorted(zip(spectrum_x, spectrum_y))
	xy = list(zip(*xy))
	specx = xy[0]
	specy = xy[1]

	if name=="Alluxa 375-10":
		lam = 375
		dlam = 10
	elif name=="Alluxa 470-10":
		lam = 470
		dlam = 10
	elif name=="Alluxa 465-2":
		lam = 465
		dlam = 2
	elif name=="Semrock 655-40":
		lam = 655
		dlam = 47

	plt.figure("Wavelength spectrum")
	# Extend spectrum to have finer sampling for the FFT
	specx, specy = extend_spectrum(specx, specy, factor=50)

	plt.plot(specx, specy, label=name, color=pcolor, marker=pmarker, linestyle=plinestyle, linewidth=plinewidth)

	# Convert to angular frequency spectrum
	specx, specy = angular_frequency_spectrum(lambdas=specx, spec_y=specy)
	# Interpolate frequencies to have a spectrum with constant x axis
	specx, specy = constant_x_axis_spectrum(specx, specy)
	## Densify spectrum to have finer sampling for the FFT
	#specx, specy = densify_spectrum(specx, specy)
	# Normalize spectrum
	specx, specy = normalized_spectrum(specx, specy)
	plt.figure("Normalized frequency spectrum"); plt.plot(specx, specy, label=name, color=pcolor, marker=pmarker, linestyle=plinestyle, linewidth=plinewidth)
	# Make Fourier transform to get g(1)
	g1y = np.fft.fft(normalize_y(specy))
	gx  = fft_x_axis(specx)
	# Apply Siegert relation to get g(2)
	g2y = make_g2(g1y)
	g2stepsize = stepsize(specx)
	# Integral yields coherence time
	int_g2 = g2_integral(g2y, g2stepsize)
	print ("Coherence time {} :\t{:.3f} ps  / {:.3f} ps (unpolarized)".format(name, 1e12*int_g2, 0.5*1e12*int_g2))
	plt.figure("g2 function"); plt.plot(gx, g2y, label="{}    {:.3f} ps".format(name,1e12*int_g2), color=pcolor, marker=pmarker, linestyle=plinestyle, linewidth=plinewidth)
	np.savetxt(f'simulations/{name}_g2.txt', np.stack((gx, g2y),axis=1))
	# to calculate the factor added in the equation of the coherence time at zero baseline, devide the unpolarized time by the equation
	k_T = (0.5*int_g2) / (0.5*(lam*1e-9)**2/ (2.998e8*dlam*1e-9))
	print("k_T {} : {:.2f}".format(name, k_T))
	A = 0.5*k_T*(lam*1e-9)**2/ (2.998e8*dlam*1e-9)
	print("expected ZB coherence time {} : {:.2f} fs".format(name,A*1e15))

####################################
######### Adding the stuff #########
####################################
add_spectrum(spectrum_x=np.loadtxt("simulations/Alluxa_375-10.txt")[:,0],spectrum_y=np.loadtxt("simulations/Alluxa_375-10.txt")[:,1], name="Alluxa 375-10", pcolor="violet", plinewidth=2)
add_spectrum(spectrum_x=np.loadtxt("simulations/Alluxa_470-10.txt")[:,0],spectrum_y=np.loadtxt("simulations/Alluxa_470-10.txt")[:,1], name="Alluxa 470-10", pcolor="#003366", plinewidth=2)
add_spectrum(spectrum_x=np.loadtxt("simulations/Alluxa_465-2.txt")[:,0],spectrum_y=np.loadtxt("simulations/Alluxa_465-2.txt")[:,1], name="Alluxa 465-2", pcolor="blue", plinewidth=2)
add_spectrum(spectrum_x=np.loadtxt("simulations/655-40.txt")[:,0],spectrum_y=np.loadtxt("simulations/655-40.txt")[:,1], name="Semrock 655-40", pcolor="red", plinewidth=2)


#########################
######### Plots #########
#########################
plt.figure("Wavelength spectrum", figsize=(10,6)); plt.title("Wavelength spectrum")
plt.legend(); plt.xlabel(r"$\lambda$ [nm]"); plt.ylabel("Transmission [%]"); plt.xlim(0,1000)
plt.savefig("simulations/img/wavelengths.png")

plt.figure("Normalized frequency spectrum", figsize=(10,6)); plt.title("Normalized frequency spectrum")
plt.xlabel("Frequency [Hz]"); plt.ylabel("Spectrum [1/Hz]"); plt.legend(); plt.xlim(0,1e16)
plt.savefig("simulations/img/frequencies.png")

plt.figure("g2 function", figsize=(10,6)); plt.title("$g^{(2)}$ functions")
plt.xlabel(r"$\tau$ [s]"); plt.ylabel("$g^{(2)}$"); plt.legend(); plt.xlim(0,0.1e-12)
plt.savefig("simulations/img/g2.png")

plt.show()
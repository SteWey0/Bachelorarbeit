import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.special as sp

R_s = 6.96342e8 # Solar radius
Ly  = 9.46e15 	# Light year

##############
# For a star #
##############
#Alle angaben in mm
#wellenlaenge    = 465e-6
#entfernung      = 1e3 * 570*Ly
#pinhole         = 1e3 * 8.8*R_s
#oeffnungsradius = 6*1e3

##############
# In the lab #
##############
##Alle angaben in mm
wellenlaenge    = 470e-6
entfernung      = 1.75e3
pinhole         = 0.030
oeffnungsradius = 25.4/2



def g2(delta_r):
	zaehler = np.pi*pinhole*delta_r
	nenner = (wellenlaenge*entfernung)
	argument = zaehler/nenner
	g2_value = ( 2* sp.j1(argument) / argument)**2
	return g2_value

t=[]

radii = []
phis  = []

statistics = 1000000

d=[]
for i in range (statistics):
	# Erstes Photon: Phi = 0
	r1 = np.random.triangular(left=0, mode=oeffnungsradius, right=oeffnungsradius)
	# Zweites Photon:
	r2 = np.random.triangular(left=0, mode=oeffnungsradius, right=oeffnungsradius)
	phi = np.random.uniform(low =0, high =np.pi)

	distance = float(np.sqrt( r1**2 + r2**2 - 2*r1*r2* np.cos(phi) ))
	d.append(distance)
	t.append(g2(distance))


t=np.array(t)
print ("Distance distribution (avg +/- rms): {:.2f} +/- {:.2f}".format(np.mean(d),np.std(d)))


print("[" + str(wellenlaenge*1e6) + " nm] x (" + str(pinhole * 1e3) + " mum) --------" + str(entfernung * 1e-3) + " m -------- (" + str(oeffnungsradius*2) + " mm)")
print("Spatial coherence loss: " +str(np.mean(t)))

# fig = plt.figure(figsize=(6,3))

# ax1 = plt.subplot(111)

# ax1.hist(d,bins=100, color="#0099ff")
# ax1.set_xlabel("Distance of photons [mm]")
# ax1.set_ylabel("Number of photon pairs", color="#0099ff", alpha=0.7)
# ax1.tick_params(axis='y', labelcolor="#0099ff")

# ax2 = ax1.twinx()
# xplot = np.arange(0.01,2*oeffnungsradius,0.01)
# ax2.plot(xplot, g2(xplot)+1., c="red", linewidth=2)
# ax2.set_ylabel("$g^{(2)}$", color="red")
# ax2.tick_params(axis='y', labelcolor="red")

# plt.tight_layout()
# plt.savefig("simulations/sc_loss.pdf")
# plt.show()

# t=np.array(t)
# print(np.mean(t))

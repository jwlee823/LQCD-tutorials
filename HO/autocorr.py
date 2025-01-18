#!/usr/bin/env python3

# Monte Carlo simulation of the quantum harmonic oscillator as an example of numerical calculations of path integrals on a lattice
# Sample code for LQCD tutorials presented in the 1st nuclear physics turtles lecture series held in Busan, Jan. 20, 2025

import numpy as np
import matplotlib.pyplot as plt

######################################################################################
# defining functions
######################################################################################

# autocorrelation function
def autocorr(rawdata, tmc):
  nc = len(rawdata)
  mean = 0.0
  ao = np.zeros(tmc)

  for k in range(nc):
    mean += rawdata[k]/nc # compute the ensemble average

  for k in range(tmc):
    E1 = np.mean(rawdata[0:nc-k])
    E2 = np.mean(rawdata[k:nc])
    norm = 1.0/(nc-k-1)
    ao[k] = norm*np.sum((rawdata[0:nc-k]-E1)*(rawdata[k:nc]-E2))

  return ao

######################################################################################
# main code
######################################################################################

# read data from csv file

NT = 24
m = 0.5
ntcut = int(NT/2)
bs = 1 # bin size
tmc = 10

outcorr = ['out_corr_NT',str(NT),'m',str(m),'.csv']
outmom = ['out_moments_NT',str(NT),'m',str(m),'.csv']
data = np.loadtxt("".join(outcorr), delimiter=',')
data_moments = np.loadtxt("".join(outmom), delimiter=',')

nc = int(len(data)/NT) # number of configurations
nb = int(nc/bs) # number of resamples

data = np.reshape(data, (nc,NT)) 
data_moments = np.reshape(data_moments, (nc,4))

acf = autocorr(data_moments[:,0], tmc)

print(acf)

iac = np.zeros(tmc)
ao0 = 1.0/acf[0]
for j in range(tmc):
  iac[j] = 0.5 + ao0*np.sum(acf[0:j])

print(iac)


#x = np.arange(0,tmc,1)
#yac = acf[0:tmc]
#yiac = iac[0:tmc]

fig, axs = plt.subplots(1,2)
axs[0].plot(acf)
axs[1].plot(iac)
#plt.ylim([0.0,1.0])
plt.show()


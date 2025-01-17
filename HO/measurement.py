#!/usr/bin/env python3

# Monte Carlo simulation of the quantum harmonic oscillator as an example of numerical calculations of path integrals on a lattice
# Sample code for LQCD tutorials presented in the 1st nuclear physics turtles lecture series held in Busan, Jan. 20, 2025

import numpy as np
import matplotlib.pyplot as plt

######################################################################################
# defining functions
######################################################################################

# resample data with Jackknife method
def jackknife(rawdata, bs):
  nc = len(rawdata)
  nb = int(nc/bs)
  mean = 0.0
  resample = np.zeros(nb)

  for k in range(nc):
    mean += rawdata[k]/nc # compute the ensemble average

  for k in range(nb):
    bsum = 0.0
    for l in range(bs):
      bsum += rawdata[l+k*bs] # sum of rawdata over a given bin
    resample[k] = (nc*mean-bsum)/(nc-bs) # Jackknife resamples 
  return resample

# compute the effective mass - symmetrized version
def effmass_log(corr):
  NT = len(corr)
  meff = np.zeros(int(NT/2))
  for j in range(int(NT/2)):
    corr[(j-1+NT)%NT] += corr[(-j+1+NT)%NT]
  for j in range(int(NT/2)):
    meff[j] = 0.5*(np.log((corr[(j-1+NT)%NT])/(corr[(j+1)%NT])))
  return meff

# compute the effective mass - cosh version
def effmass(corr):
  NT = len(corr)
  meff = np.zeros(int(NT))
  for j in range(int(NT)):
    meff[j] = np.arccosh((corr[(j-1+NT)%NT]+corr[(j+1)%NT])/(2*corr[j]))
  return meff

# compute the effective mass - cosh version with symmetrization
def effmass_cosh(corr):
  NT = len(corr)
  meff = np.zeros(int(NT/2))
  for j in range(int(NT/2)-1):
    corr[j] += corr[(-j+NT)%NT]
    corr[j] *= 0.5
  for j in range(int(NT/2)-2):
    meff[j] = np.arccosh((corr[(j-1+NT)%NT]+corr[(j+1)%NT])/(2*corr[j]))
  return meff


######################################################################################
# main code
######################################################################################

NT = 24
m = 0.5
bs = 1 # bin size

# read data from csv file

outcorr = ['out_corr_NT',str(NT),'m',str(m),'.csv']
outmom = ['out_moments_NT',str(NT),'m',str(m),'.csv']
data = np.loadtxt("".join(outcorr), delimiter=',')
data_moments = np.loadtxt("".join(outmom), delimiter=',')

nc = int(len(data)/NT) # number of configurations
nb = int(nc/bs) # number of resamples

data = np.reshape(data, (nc,NT)) 
data_moments = np.reshape(data_moments, (nc,4))

# compute <x>, <x^2>, <x^3>, <x^4>

rawdata = data_moments[:,0]
resampledata = jackknife(rawdata, bs)
xmean = np.mean(rawdata)
xstd = np.std(resampledata)*np.sqrt(nb-1)
print("<x> =", xmean,xstd)

rawdata = data_moments[:,1]
resampledata = jackknife(rawdata, bs)
xmean = np.mean(rawdata)
xstd = np.std(resampledata)*np.sqrt(nb-1)
print("<x^2> =", xmean,xstd)

rawdata = data_moments[:,2]
resampledata = jackknife(rawdata, bs)
xmean = np.mean(rawdata)
xstd = np.std(resampledata)*np.sqrt(nb-1)
print("<x^3> =", xmean,xstd)

rawdata = data_moments[:,3]
resampledata = jackknife(rawdata, bs)
xmean = np.mean(rawdata)
xstd = np.std(resampledata)*np.sqrt(nb-1)
print("<x^4> =", xmean,xstd)


# compute 2-point correlation function, <x(t)x(0)>

corrmean = np.zeros(NT)
corrvar = np.zeros(NT)
resample = np.zeros((nb,NT))

for j in range(NT):
  for k in range(nc):
    corrmean[j] += data[k][j]/nc # compute the ensemble average of the correlation functions at a given t

  for k in range(nb):
    bmean = 0.0
    for l in range(bs):
      bmean += data[l+k*bs][j] # sum of corr over a given bin
    resample[k][j] = (nc*corrmean[j]-bmean)/(nc-bs) # Jackknife resamples 

    corrvar[j] += (resample[k][j]-corrmean[j])*(resample[k][j]-corrmean[j])*(nb-1)/nb # compute the variance of Jackknife resamples

# compute the effective mass, m_eff

meffmean = np.zeros(NT)
meffvar = np.zeros(NT)

for k in range(nb):
  tmp = effmass_cosh(resample[k])
  for j in range(int(NT/2)-2):
    meffmean[j] += tmp[j]/nb

for k in range(nb):
  tmp = effmass_cosh(resample[k])
  for j in range(int(NT/2)-2):
    meffvar[j] += (tmp[j]-meffmean[j])*(tmp[j]-meffmean[j])

ntstart = 0
ntcut = int(NT/2)-2

x = np.arange(ntstart, ntcut, 1)
yc = corrmean[ntstart:ntcut]
ycerr = np.sqrt(corrvar[ntstart:ntcut]) 

ym = meffmean[ntstart:ntcut]
ymerr = np.sqrt(meffvar[ntstart:ntcut]) 


fig, plot = plt.subplots(1,2)
plot[0].errorbar(x, yc, ycerr)
plot[1].errorbar(x, ym, ymerr)
#plt.ylim([0.0,1.0])
plt.show()


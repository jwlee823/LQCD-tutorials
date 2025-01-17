#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# defining functions

# calculation of the action
def action(config, mass, omeg2):
  en = 0
  NT = len(config)

  for j in range(NT):
    uke = (config[(j+1)%NT]-config[j%NT])*(config[(j+1)%NT]-config[j%NT])*0.5*mass
    dke = (config[j%NT]-config[(j-1+NT)%NT])*(config[j%NT]-config[(j-1+NT)%NT])*0.5*mass
    pe = (config[j%NT]*config[j%NT])*0.5*mass*omeg2
    en += uke + dke + pe

  return en

# compute the average value of x
def calx(config):
  NT = len(config)
  xavg = np.sum(config)
  return xavg/NT

# resample the data with Jackknife method
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


# compute the effective mass - symmetrized version
def effmass1(corr):
  NT = len(corr)
  meff = np.zeros(int(NT/2))
  for j in range(int(NT/2)):
    corr[(j-1+NT)%NT] += corr[(-j+1+NT)%NT]
  for j in range(int(NT/2)):
    meff[j] = 0.5*(np.log((corr[(j-1+NT)%NT])/(corr[(j+1)%NT])))
  return meff

# compute the effective mass - cosh version
def effmass2(corr):
  NT = len(corr)
  meff = np.zeros(int(NT))
  for j in range(int(NT)):
    meff[j] = np.arccosh((corr[(j-1+NT)%NT]+corr[(j+1)%NT])/(2*corr[j]))
  return meff


# main code

# read data from csv file

NT = 24
m = 0.5
ntcut = int(NT/2)
bs = 1 # bin size
tmc = 100

outcorr = ['out_corr_NT',str(NT),'m',str(m),'.csv']
outmom = ['out_moments_NT',str(NT),'m',str(m),'.csv']
data = np.loadtxt("".join(outcorr), delimiter=',')
data_moments = np.loadtxt("".join(outmom), delimiter=',')
#data = np.loadtxt('out_corr_NT100m0.05.csv', delimiter=',')
#data_moments = np.loadtxt('out_moments_NT100m0.05.csv', delimiter=',')

nc = int(len(data)/NT) # number of configurations
nb = int(nc/bs) # number of resamples


data = np.reshape(data, (nc,NT)) 
data_moments = np.reshape(data_moments, (nc,4))

acf = autocorr(data_moments[:,0], tmc)

print(acf)

iao = np.zeros(tmc)
ao0 = 1.0/acf[0]
for j in range(tmc):
  iao[j] = 0.5 + ao0*np.sum(acf[0:j])

print(iao)

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


corrmean = np.zeros(NT)
corrvar = np.zeros(NT)
resample = np.zeros((nb,NT))

for j in range(NT):
  for k in range(nc):
    corrmean[j] += data[k][j]/nc # compute the ensemble average of the correlation functions at a given t

#  resample = np.zeros(nb)
  for k in range(nb):
    bmean = 0.0
    for l in range(bs):
      bmean += data[l+k*bs][j] # sum of corr over a given bin
    resample[k][j] = (nc*corrmean[j]-bmean)/(nc-bs) # Jackknife resamples 

    corrvar[j] += (resample[k][j]-corrmean[j])*(resample[k][j]-corrmean[j])*(nb-1)/nb # compute the variance of Jackknife resamples

#print(corrmean)
#print(np.sqrt(corrvar))
#print(corrvar)

meffmean = np.zeros(NT)
meffvar = np.zeros(NT)

for k in range(nb):
  tmp = effmass2(resample[k])
  for j in range(int(NT/2)):
    meffmean[j] += tmp[j]/nb

for k in range(nb):
  tmp = effmass2(resample[k])
  for j in range(int(NT/2)):
    meffvar[j] += (tmp[j]-meffmean[j])*(tmp[j]-meffmean[j])


#print(meffmean)
#print(np.sqrt(meffvar))
#print(meffvar)


x = np.arange(3,ntcut,1)
yc = corrmean[3:ntcut]
ycerr = np.sqrt(corrvar[3:ntcut]) 

ym = meffmean[3:ntcut]
ymerr = np.sqrt(meffvar[3:ntcut]) 

print(ym)
#print(yerr)

fig, plot = plt.subplots(1,2)
plot[0].errorbar(x, yc, ycerr)
plot[1].errorbar(x, ym, ymerr)
#plt.ylim([0.0,1.0])
plt.show()


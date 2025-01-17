#!/usr/bin/env python3

# Monte Carlo simulation of the quantum harmonic oscillator as an example of numerical calculations of path integrals on a lattice
# Sample code for LQCD tutorials presented in the 1st nuclear physics turtles lecture series held in Busan, Jan. 20, 2025

import numpy as np
import matplotlib.pyplot as plt

######################################################################################
# defining functions
######################################################################################

# prepare an initial configuration
def init(N):
  state = 2*np.random.rand(N)-1 # random, hot, disordered
#  state = np.ones(N) # unit, cold, ordered
  return state

# calculation of the action
def action(config, mass, omeg2):
  en = 0
  NT = len(config)

  for j in range(NT):
    uke = (config[(j+1)%NT]-config[j%NT])*(config[(j+1)%NT]-config[j%NT])*0.5*mass
    dke = (config[j%NT]-config[(j-1+NT)%NT])*(config[j%NT]-config[(j-1+NT)%NT])*0.5*mass
    pe = (config[j%NT]*config[j%NT])*0.5*mass*omeg2
    en += uke + dke + pe
#    en += uke + pe # kinetic term with a derivative only for the forward direction

  return en

# update configuration using Metropolis-Hasting
# compute delta_s by explicitly?
# update h as well?

def update(config, mass, omeg2, h):
  NT = len(config)
  accrate = 0.0 
  idrate = 0.8 # a desired acceptance rate

  for j in range(NT):
    a = np.random.randint(0, NT)
    newfield = config[a] + h*(np.random.rand()-0.5) # generate new field with random fluctuation

    uke = (config[(a+1)%NT]-config[a%NT])*(config[(a+1)%NT]-config[a%NT])*0.5*mass 
    dke = (config[(a)%NT]-config[(a+NT-1)%NT])*(config[(a)%NT]-config[(a+NT-1)%NT])*0.5*mass 
    pe = (config[a%NT]*config[a%NT])*0.5*mass*omeg2
    old = uke+dke+pe
#    old = uke+pe # kinetic term with a derivative only for the forward direction

    uke = (config[(a+1)%NT]-newfield)*(config[(a+1)%NT]-newfield)*0.5*mass 
    dke = (newfield-config[(a+NT-1)%NT])*(newfield-config[(a+NT-1)%NT])*0.5*mass 
    pe = (newfield*newfield)*0.5*mass*omeg2
    new = uke+dke+pe
#    new = uke+pe # kinetic term with a derivative only for the forward direction

    delta_s = new - old # the action difference
    prob = np.exp(-delta_s) # transition probability

    if np.random.rand() < prob: # Metropolis Hasting test
      config[a] = newfield
      accrate += 1.0/NT

#  print("acceptance rate", accrate, sep="=")
#  print("addjusted value of h", h, sep="=")

#  uh = h*accrate/idrate # adjustment of the magnitude of the random fluctuation
#  uconfig = config
#  return uconfig, uh # return both updated configurations and h value
  return config

# compute the average value of x
def calx(config):
  NT = len(config)
  xavg = np.sum(config)
  return xavg/NT

# compute the average value of x^2
def calx2(config):
  NT = len(config)
  x2avg = 0.0
  for j in range(NT):
    x2avg += config[j]*config[j]
  return x2avg/NT

# compute the average value of x^3
def calx3(config):
  NT = len(config)
  x3avg = 0.0
  for j in range(NT):
    x3avg += config[j]*config[j]*config[j]
  return x3avg/NT

# compute the average value of x^4
def calx4(config):
  NT = len(config)
  x4avg = 0.0
  for j in range(NT):
    x4avg += config[j]*config[j]*config[j]*config[j]
  return x4avg/NT


######################################################################################
# main code
######################################################################################

NT = 8 # lattice extent
mass = 0.5 # default mass = 0.5
omeg = mass
#mass = 0.5*2 # in the case of the forward derivative
#omeg = 0.5/np.sqrt(2) # in the case of the forward derivative
omeg2 = omeg*omeg 

h = 2.0 # magnitude of the random fluctuation
ninit = 10000 # number of trajectories for thermalization
nsweep = 200 # number of trajectories for each sweep
nconf = 10000 # number of sweeps for each configuration

nsave = int((ninit/nsweep+nconf)) # calculate <x> at every 10 traj. 

norm = 1.0/(nconf)

# initialize the configuration

config = init(NT)
print("initial config", config, sep="=")

xlist = np.zeros(nsave)
x2list = np.zeros(nsave)

# thermalize the configurations
for j in range(int(ninit/nsweep)):

  xlist[j%nsweep] = calx(config)
  x2list[j%nsweep] = calx2(config)

  for k in range(nsweep):
    update(config,  mass, omeg2, h)
#    uconfig, uh = update(config, mass, omeg2, h)
#    h, config = uh, uconfig

outcorr = ['out_corr_NT',str(NT),'m',str(mass),'.csv']
outmom = ['out_moments_NT',str(NT),'m',str(mass),'.csv']
mycorrfile = open("".join(outcorr), "w") # save correlators
mymomfile = open("".join(outmom), "w") # save x, x^2, x^3, x^4

# generate the configurations for measurements

for k in range(nconf): 

  ktmp = int(ninit/nsweep+k)
  xlist[ktmp] = calx(config)
  x2list[ktmp] = calx2(config)

  x1 = calx(config)
  x2 = calx2(config)
  x3 = calx3(config)
  x4 = calx4(config)

  np.savetxt(mymomfile, (x1,x2,x3,x4))

  corrtmp = np.zeros(NT)
  for dt in range(NT):
    for j in range(NT):
      corrtmp[dt] += config[j%NT]*config[(j+dt)%NT]/NT

  np.savetxt(mycorrfile, corrtmp, delimiter=",")

  for j in range(nsweep):
    update(config, mass, omeg2, h)
#    uconfig, uh = update(config, mass, omeg2, h)
#    h, config = uh, uconfig

print("updated config", config, sep="=")
enavg = action(config, mass, omeg2)


fig, axs = plt.subplots(1,2)

axs[0].plot(xlist)
axs[1].hist(xlist, bins=20)
plt.show()

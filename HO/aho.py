#!/usr/bin/env python3

# Monte Carlo simulation of the quantum anharmonic oscillator as an example of numerical calculations of path integrals on a lattice
# Sample code for LQCD tutorials presented in the 1st nuclear physics turtles lecture series held in Busan, Jan. 20, 2025

import numpy as np
import matplotlib.pyplot as plt

######################################################################################
# defining functions
######################################################################################

# initial configuration
def init(N):
  state = 2*np.random.rand(N)-1
  return state

# calculation of the action
def action(config, eta2, mass, la):
  en = 0
  NT = len(config)

  for j in range(NT):
    uke = (config[(j+1)%NT]-config[j%NT])*(config[(j+1)%NT]-config[j%NT])*0.5*mass
    dke = (config[(j)%NT]-config[(j+NT-1)%NT])*(config[(j)%NT]-config[(j+NT-1)%NT])*0.5*mass 
    pe = (config[j%NT]*config[j%NT]-eta2)*(config[j%NT]*config[j%NT]-eta2)*la
    en += uke +  pe
#    en += uke + dke + pe

  return en

# update configuration using Metropolis-Hasting
# compute delta_s by explicitly?
# update h as well?

def update(config, eta2, mass, la, h):
  NT = len(config)
  accrate = 0.0

  for j in range(NT):
    a = np.random.randint(0, NT)
    newfield = config[a] + h*(np.random.rand()-0.5)

    uke = (config[(a+1)%NT]-config[a%NT])*(config[(a+1)%NT]-config[a%NT])*0.5*mass 
    dke = (config[(a)%NT]-config[(a+NT-1)%NT])*(config[(a)%NT]-config[(a+NT-1)%NT])*0.5*mass 
    pe = (config[a%NT]*config[a%NT]-eta2)*(config[a%NT]*config[a%NT]-eta2)*la
    old = uke+pe
#    old = uke+dke+pe

    uke = (config[(a+1)%NT]-newfield)*(config[(a+1)%NT]-newfield)*0.5*mass 
    dke = (newfield-config[(a+NT-1)%NT])*(newfield-config[(a+NT-1)%NT])*0.5*mass 
    pe = (newfield*newfield-eta2)*(newfield*newfield-eta2)*la
    new = uke+pe
#    new = uke+dke+pe

    delta_s = new - old
    prob = np.exp(-delta_s)

    if np.random.rand() < prob:
      config[a] = newfield
      accrate += 1.0/NT

#  print("acceptance rate", accrate, sep="=")
  return config

# compute the average value of x
def calx(config):
  NT = len(config)
  xavg = np.sum(config)
  return xavg/NT

# compute the average value of x
def calx2(config):
  NT = len(config)
  x2avg = 0
  for j in range(NT):
    x2avg += config[j]*config[j]
  return x2avg/NT

######################################################################################
# main code
######################################################################################

NT = 24
factor = 0.05
mass = 0.5/factor 
la = 5.0*factor 
eta = 1.4 # default value = 1.4
eta2 = eta*eta

h = 2.0 # default value 1.2
ninit = 1000 # number of trajectories for thermalization
nsweep = 10 # number of trajectories for each sweep
nconf = 5000 # number of sweeps for each configuration

nsave = int((ninit/nsweep+nconf)) # calculate <x> at every 10 traj. 

norm = 1.0/(nconf)

# initialize the configuration
config = init(NT)
print("initial config", config, sep="=")

enavg = action(config, eta2, mass, la)
print("initial action", enavg, sep="=")

xlist = np.zeros(nsave)
x2list = np.zeros(nsave)

# thermalize the configurations
for j in range(ninit):

  xlist[j%nsweep] = calx(config)
  x2list[j%nsweep] = calx2(config)

  update(config, eta2, mass, la, h)

xavg = 0 # calculate the average value of x
x2avg = 0 # calculate the average value of x

for k in range(nconf): 

  ktmp = int(ninit/nsweep+k)
  xlist[ktmp] = calx(config)
  x2list[ktmp] = calx2(config)

  xavg += calx(config)
  x2avg += calx2(config)


  for j in range(nsweep):
    update(config, eta2, mass, la, h)

#print("acceptance rate", 100*acc/(nsweep*nconf), sep="=")

print("updated config", config, sep="=")

enavg = action(config, eta2, mass, la)

print("updated action", enavg, sep="=")

print("<x>", xavg*norm, sep="=")
print("<x^2>", x2avg*norm, sep="=")

plt.plot(xlist)
#plt.hist(xlist, bins=20)
plt.show()


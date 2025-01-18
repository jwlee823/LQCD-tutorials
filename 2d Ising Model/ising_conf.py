#!/usr/bin/env python3

# Monte Carlo simulation of the 2d ising model
# Sample code for LQCD tutorials presented in the 1st nuclear physics turtles lecture series held in Busan, Jan. 20, 2025

import numpy as np
import matplotlib.pyplot as plt

# test for random number generator
#lri=np.random.randint(2, size=10)
#print(lri)

#lrr=np.random.rand(10)
#print(lrr)

# to do list
# 1. introduce the stength of the interaction, J

######################################################################################
#defining functions
######################################################################################

# initial configuration - random
# | 1 -1 ... |
# | ....     |
# | .... 1 1 |
# to do: option for switching to other initial configuration e.g. unit
def init(N):
  state = 2*np.random.randint(2, size=(N,N))-1
#  state = np.ones((N,N))
  return state

# calculation of the action density: - sum_{i,j} s_i * s*j
# implement a periodic boundary condition: s_(N+i) = s_i
# to do: reduce the # of calculation by half?
def action(config):
  en = 0
  Ns = len(config)

  for i in range(Ns):
    for j in range(Ns):
      field = config[i,j]
      nb = config[(i+1)%Ns, j] + config[i, (j+1)%Ns] + config[(i-1)%Ns, j] + config[i, (j-1)%Ns]
      en += -nb*field

  return en/4

# update configurations using Metropolis-Hasting Monte Carlo algorithm
# 
def update(config, beta):
  Ns = len(config)

  for i in range(Ns):
    for j in range(Ns):
      a = np.random.randint(0, Ns)
      b = np.random.randint(0, Ns)
      field = config[a,b]
      nb = config[(a+1)%Ns, b] + config[a, (b+1)%Ns] + config[(a-1)%Ns, b] + config[a, (b-1)%Ns]
      delta_s = 2*field*nb
      prob = np.exp(-delta_s*beta)

      if delta_s < 0: # is this necessary?
        field *= -1
      elif np.random.rand() < prob:
        field *= -1
      config[a, b] = field
  return config

######################################################################################
# main code
######################################################################################

Ns = 16 # Ns x Ns lattice
beta = 0.4 # inverse temperature
nther = 1001 # number of updates for the thermalisation

config = init(Ns)
print("initial config",config, sep="=")

f = plt.figure(figsize=(15, 10), dpi=80)

dconf = np.arange(6*Ns*Ns).reshape((6,Ns,Ns))

plist = [0,1,30,100,300,1000]
j = 0

for i in range(nther):
  if i == plist[j]:
    dconf[j] = config
    j += 1
  update(config, beta)

X, Y = np.meshgrid(range(Ns), range(Ns))
f.add_subplot(2,3,1)
plt.pcolormesh(X, Y, dconf[0], cmap=plt.cm.RdBu)
f.add_subplot(2,3,2)
plt.pcolormesh(X, Y, dconf[1], cmap=plt.cm.RdBu)
f.add_subplot(2,3,3)
plt.pcolormesh(X, Y, dconf[2], cmap=plt.cm.RdBu)
f.add_subplot(2,3,4)
plt.pcolormesh(X, Y, dconf[3], cmap=plt.cm.RdBu)
f.add_subplot(2,3,5)
plt.pcolormesh(X, Y, dconf[4], cmap=plt.cm.RdBu)
f.add_subplot(2,3,6)
plt.pcolormesh(X, Y, dconf[5], cmap=plt.cm.RdBu)

plt.show()

print("last config",config, sep="=")

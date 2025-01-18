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

# compute the magnetization
def calcmag(config):
  mag = np.sum(config)
  return mag

######################################################################################
# main code
######################################################################################

Nt = 40 # number of data points along the temperature
dt = 0.02 # distance between adjacent data points
Ns = 16 # Ns x Ns lattice
t0 = 2.0 # initial temperature
#beta = 0.67
ninit = 2000 # number of updates for thermalisation
nconf = 1000 # number of configurations
nsweep = 10 # number of updates between adjacent configurations

pref = 1.0/(Ns*Ns*nconf)
pref2 = 1.0/(Ns*Ns*nconf*nconf)

config = init(Ns)
print("initial config",config, sep="=")

avgen = action(config)
print("initial action density", avgen/(Ns*Ns), sep="=")

temp = np.zeros(Nt)
enavg = np.zeros(Nt)
magavg = np.zeros(Nt)
shavg = np.zeros(Nt)
scavg = np.zeros(Nt)

output = ['out_N',str(Ns),'dt',str(dt),'temp',str(t0),'Nt',str(Nt),'.csv']
myfile = open("".join(output),"w")
#myfile = open("out_N128dt0.02temp2.0Nt40.csv", "w")

for k in range(Nt):

  temp[k] = t0 + dt*k
  beta = 1.0/temp[k] # inverse temperature

  for i in range(ninit):
    update(config, beta)

  en1 = 0.0
  m1 = 0.0
  en2 = 0.0
  m2 = 0.0

  for i in range(nconf):
    for j in range(nsweep):
      update(config, beta)

    en = action(config)
    mag = calcmag(config)

    np.savetxt(myfile, (en, mag))

    en1 += en
    m1 += mag
    en2 += en*en
    m2 += mag*mag
  
  enavg[k] = en1*pref
  magavg[k] = m1*pref
  shavg[k] = (en2*pref-en1*en1*pref2)*beta*beta
  scavg[k] = (m2*pref-m1*m1*pref2)*beta

print("last config",config, sep="=")


print("temperature", temp, sep="=")
print("average value of the action", enavg, sep="=")
print("average value of the magnetization", magavg, sep="=")
print("average value of the specific heat", shavg, sep="=")
print("average value of the susceptibility", scavg, sep="=")

f = plt.figure(figsize=(12, 8)); # plot the calculated values    

sp =  f.add_subplot(2, 2, 1 );
plt.scatter(temp, enavg, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

sp =  f.add_subplot(2, 2, 2 );
plt.scatter(temp, abs(magavg), s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');

sp =  f.add_subplot(2, 2, 3 );
plt.scatter(temp, shavg, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);  
plt.ylabel("Specific Heat ", fontsize=20);   plt.axis('tight');   

sp =  f.add_subplot(2, 2, 4 );
plt.scatter(temp, scavg, s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Susceptibility", fontsize=20);   plt.axis('tight');

plt.show()

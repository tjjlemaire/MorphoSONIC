# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:32:29 2019

@author: Maria
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from convergence import convergence

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format
from ExSONIC.core import ExtracellularCurrent

#Set logging level
import logging
logger.setLevel(logging.INFO)

# Fiber model parameters
pneuron = getPointNeuron('FH')       # mammalian fiber membrane equations
fiberD = 10e-6                       # fiber diameter (m)
rho_a = 54.7                         # axoplasm resistivity (Ohm.cm)
d_ratio = 0.6                        # axon / fiber diameter ratio
fiberL = 1e-2                       # fiber length (m) 

# Intracellular stimulation parameters
tstim = 100e-6   # s
toffset = 3e-2  # s
PRF = 100.      # Hz
DC = 1.         # -

x0 = 0              # point-electrode located above central node (m)
z0 = 100*fiberD     # point-electrode to fiber distance (m, 1 internode length)
mode = 'cathode'    # cathodic pulse   
rho_e = 300.0       # resistivity of external medium (Ohm.cm, from McNeal 1976)

psource = ExtracellularCurrent(x0, z0, rho=rho_e, mode=mode)

#Choose the number of nodes range
nnodes = np.logspace(0, 4, 100)
nnodes = np.asarray(np.ceil(nnodes) // 2 * 2 + 1, dtype=int)

#run the convergence
Ithrs = convergence(pneuron, fiberD, rho_a, d_ratio, fiberL, tstim, toffset, PRF, DC, psource, nnodes)

#Recalculate number of nodes to node length and 
nodeL = fiberL / nnodes
Ithr_ref = Ithrs[-1]
max_rel_error = 0.01
rel_errors = np.abs((Ithrs - Ithr_ref) / Ithr_ref)
min_nodes = np.interp(max_rel_error, rel_errors[::-1], nnodes[::-1], left=np.nan, right=np.nan)
print(min_nodes)
max_nodeL = np.interp(max_rel_error, rel_errors[::-1], nodeL[::-1], left=np.nan, right=np.nan)
print(max_nodeL)

#Plotting
#plt.plot(nnodes, rel_errors, label = 'errors')
fig, ax = plt.subplots()
ax.plot(nodeL, rel_errors * 100, label = 'errors')
ax.axhline(max_rel_error * 100, linestyle ='dashed', label = 'threshold error', color = 'k')
#plt.axvline(min_nodes, label = 'min nnodes', color = 'k')
ax.axvline(max_nodeL, label = 'max length', color = 'k')
#plt.xlabel('Number of nodes')
ax.set_xlabel('Node length, m')
ax.set_ylabel('relative error (%)')
ax.set_xscale('log')
ax.invert_xaxis()
ax.legend()
#plt.title(f'Minimum number of nodes = {int(min_nodes)}')
ax.set_title(f'Maximum node length = {max_nodeL * 1e6:.2f} um')

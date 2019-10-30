# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-10-28 18:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-10-30 15:56:57

import numpy as np
import matplotlib.pyplot as plt

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format
from ExSONIC.core import VextSennFiber, ExtracellularCurrent, PointSourceArray
from ExSONIC.plt import SectionCompTimeSeries


# Fiber model parameters
pneuron = getPointNeuron('FH')  # FrankenHaeuser-Huxley membrane equations
fiberD = 20e-6                  # fiber diameter (m)
nnodes = 21                     # number of nodes
rho_a = 110.0                   # axoplasm resistivity (Ohm.cm, from McNeal 1976)
d_ratio = 0.7                   # axon / fiber diameter ratio (from McNeal 1976)
nodeL = 2.5e-6                  # node length (m, from McNeal 1976)
fiber = VextSennFiber(pneuron, fiberD, nnodes, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)
xnodes = fiber.getNodeCoords()
inodes = np.arange(fiber.nnodes) + 1

# Electrode and stimulation parameters (from Reilly 1985)
rho_e = 300.0      # resistivity of external medium (Ohm.cm, from McNeal 1976)
z0 = fiber.interL  # point-electrode to fiber distance (m, 1 internode length)
x0 = 0.            # point-electrode located above central node (m)
mode = 'cathode'   # cathodic pulse
tstim = 1.5e-3     # s
toffset = 3e-3     # s
PRF = 100.         # Hz
DC = 1.            # -

# Stimulation current
I = -1e-3  # A

# Point-source array
nsources = 3
x = np.linspace(0, 2 * fiber.interL, nsources)
x -= (np.ptp(x) / 2 + fiber.interL)
z = np.ones(nsources) * z0
positions = list(zip(x, z))
psources = [ExtracellularCurrent((item, z0), rho=rho_e, mode=mode) for item in x]
rel_amps = np.linspace(-1, 1, nsources)
parray = PointSourceArray(psources, rel_amps)

# Figure 1: extracellular potentials
Ve = np.array([
    p.computeNodesAmps(fiber, I * rel_amp) for p, rel_amp in zip(psources, rel_amps)])
Ve_net = parray.computeNodesAmps(fiber, I)
fig1, ax = plt.subplots()
ax.set_title('Extracellular potential')
ax.set_xlabel('# node')
ax.set_ylabel('$V_e\ (mV)$')
ax.set_xticks(inodes)
for i, p in enumerate(psources):
    ax.plot(inodes, Ve[i], '--', c=f'C{i}', label=f'source {i + 1}')
ax.plot(inodes, Ve_net, c='k', label='combined effect')
ax.legend(frameon=False)

# Figure 2: activating functions
d2Ve = np.diff(Ve, 2) / fiber.interL**2
d2Ve_net = np.diff(Ve_net, 2) / fiber.interL**2
fig2, ax = plt.subplots()
ax.set_title('Activating function')
ax.set_xlabel('# node')
ax.set_ylabel('$d^2V_e / dx^2\ (mV/mm^2)$')
ax.set_xticks(inodes)
for i, p in enumerate(psources):
    ax.plot(inodes[1:-1], d2Ve[i], '--', c=f'C{i}', label=f'source {i + 1}')
ax.plot(inodes[1:-1], d2Ve_net, c='k', label='combined effect')
ax.legend(frameon=False)

# Fiber simulations
# Ithr = fiber.titrate(parray, tstim, toffset, PRF, DC)
# I = Ithr * 1.2
data, meta = fiber.simulate(parray, I, tstim, toffset, PRF, DC)
fig3 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

plt.show()

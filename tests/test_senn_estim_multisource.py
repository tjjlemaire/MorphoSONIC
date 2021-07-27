# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-10-28 18:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:47:06

import numpy as np
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format
from MorphoSONIC.models import SennFiber, ExtracellularCurrent, PointSourceArray
from MorphoSONIC.plt import SectionCompTimeSeries


# Fiber model parameters
fiberD = 20e-6  # fiber diameter (m)
nnodes = 21     # number of nodes
fiber = SennFiber(fiberD, nnodes)
xnodes = fiber.getNodeCoords()
inodes = np.arange(fiber.nnodes) + 1

# Electrode and stimulation parameters (from Reilly 1985)
rho_e = 300.0      # resistivity of external medium (Ohm.cm, from McNeal 1976)
z0 = fiber.interL  # point-electrode to fiber distance (m, 1 internode length)
x0 = 0.            # point-electrode located above central node (m)
mode = 'cathode'   # cathodic pulse
tstim = 1.5e-3     # s
toffset = 3e-3     # s
pp = PulsedProtocol(tstim, toffset)

# Stimulation current
I = -1e-3  # A

# Point-source array
nsources = 2
x = np.linspace(0, 2 * fiber.interL, nsources)
x -= (np.ptp(x) / 2 + fiber.interL)
z = np.ones(nsources) * z0
positions = list(zip(x, z))
psources = [ExtracellularCurrent((item, z0), rho=rho_e, mode=mode) for item in x]
rel_amps = np.linspace(-1, 1, nsources)
parray = PointSourceArray(psources, rel_amps)

# Figure 1: extracellular potentials
Ve = np.array([
    p.computeDistributedAmps(fiber, I * rel_amp) for p, rel_amp in zip(psources, rel_amps)])
Ve_net = parray.computeDistributedAmps(fiber, I)
fig1, ax = plt.subplots()
ax.set_title('Extracellular potential')
ax.set_xlabel('# node')
ax.set_ylabel('$V_e\ (mV)$')
ax.set_xticks(inodes)
for i, p in enumerate(psources):
    ax.plot(inodes, Ve[i], '--', c=f'C{i}', label=f'source {i + 1}')
ax.plot(inodes, Ve_net, c='k', label='combined effect')
ax.legend(frameon=False)

# Figure 2: activating function
Iinj = np.array([fiber.preProcessAmps(v) for v in Ve])
Iinj_net = fiber.preProcessAmps(Ve_net)
fig2, ax = plt.subplots()
ax.set_title('Activating function')
ax.set_xlabel('# node')
ax.set_ylabel('eq. injected current (nA)')
ax.set_xticks(inodes)
for i, p in enumerate(psources):
    ax.plot(inodes, Iinj[i], '--', c=f'C{i}', label=f'source {i + 1}')
ax.plot(inodes, Iinj_net, c='k', label='combined effect')
ax.legend(frameon=False)

# Fiber simulations
data, meta = fiber.simulate(parray, I, pp)
fig3 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

# TODO:
# - try to see if unidirectional activation an also be reached also with unique polarity across sources
# - vary sources - fiber distance
# - vary inter-source distance
# - vary current amplitudes
# - vary number of sources

plt.show()

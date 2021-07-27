# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:46:40

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger
from MorphoSONIC.models import SennFiber
from MorphoSONIC.sources import GaussianAcousticSource

# Set logging level
logger.setLevel(logging.INFO)

# Define sonophore parameters
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)

# Define fiber parameters
fiberD = 20e-6  # m
nnodes = 21

# Create fiber model
fiber = SennFiber(fiberD, nnodes, a=a, fs=fs)

# Define acoustic source
source = GaussianAcousticSource(
    0,                   # gaussian center (m)
    fiber.length / 10.,  # gaussian width (m)
    500e3,               # US frequency (Hz)
    A=100e3)             # peak acoustic amplitude (Pa)

# Define pulsing protocol
tpulse = 100e-6  # s
toffset = 3e-3   # s
pp = PulsedProtocol(tpulse, toffset)

# Run simulation
data, meta = fiber.simulate(source, pp)

# Extract simulation data from node 0
inode = 0
df = data[f'node{inode}']
t = df['t']  # s

# Extract currents time course from node 0 data
currents_data = fiber.getCurrentsDict(df)  # mA/m2

# Compute membrane current by retrieving axial current from net total current
currents_data['Membrane'] = currents_data['Net'] - currents_data['Ax']

# Extract point-neuron object from fiber
pneuron = fiber.pneuron

# Get dictionary of membrane currents descriptions
currents_descs = {c: pneuron.getPltVars()[c]['desc'] for c in pneuron.getCurrentsNames()}

# Add description for axial, membrane and total current temrs
currents_descs['iAx'] = 'axial current'
currents_descs['iMembrane'] = 'membrane current'
currents_descs['iNet'] = 'total current'

# Plot currents time course
fig, ax = plt.subplots()
ax.set_xlabel('time (ms)')
ax.set_ylabel('currents (A/m2)')
for k, v in currents_data.items():
    lbl = f'i{k}'
    ax.plot(t * 1e3, v * 1e-3, label=f'{lbl} ({currents_descs[lbl]})')
ax.legend()

plt.show()

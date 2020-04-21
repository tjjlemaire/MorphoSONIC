# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-19 14:36:59

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger

from ExSONIC.core import SennFiber
from ExSONIC.core.sources import *
from ExSONIC.plt import SectionCompTimeSeries

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

# Define various sources
sources = [
    IntracellularCurrent(
        sec_id=fiber.central_ID,  # target section
        I=3.0e-9),                # current amplitude (A)
    ExtracellularCurrent(
        x=(0., fiber.interL),  # point-source electrode position (m)
        I=-0.68e-3),           # current amplitude (A)
    GaussianVoltageSource(
        0,                   # gaussian center (m)
        fiber.length / 10.,  # gaussian width (m)
        Ve=-80.),            # peak extracellular voltage (mV)
    SectionAcousticSource(
        fiber.central_ID,  # target section
        500e3,             # US frequency (Hz)
        A=100e3),          # peak acoustic amplitude (Pa)
    GaussianAcousticSource(
        0,                   # gaussian center (m)
        fiber.length / 10.,  # gaussian width (m)
        500e3,               # US frequency (Hz)
        A=100e3),            # peak acoustic amplitude (Pa)
    PlanarDiskTransducerSource(
        (0., 0., 'focus'),  # transducer position (m)
        500e3,              # US frequency (Hz)
        r=2e-3,             # transducer radius (m)
        u=0.04)             # m/s
]

# Set pulsing protocol
pulse_width = 100e-6  # s
toffset = 3e-3        # s
pp = PulsedProtocol(pulse_width, toffset)

# Simulate model with each source and plot results
for source in sources:
    data, meta = fiber.simulate(source, pp)
    SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

plt.show()

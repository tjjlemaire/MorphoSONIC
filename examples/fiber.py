# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:45:43

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol, BalancedPulsedProtocol
from PySONIC.utils import logger

from MorphoSONIC.models import SennFiber
from MorphoSONIC.sources import *
from MorphoSONIC.plt import SectionCompTimeSeries

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
iintra_source = IntracellularCurrent(
    sec_id=fiber.central_ID,  # target section
    I=3.0e-9)                 # current amplitude (A)
iextra_source = ExtracellularCurrent(
    x=(0., fiber.interL),  # point-source electrode position (m)
    I=-0.70e-3)            # current amplitude (A)
voltage_source = GaussianVoltageSource(
    0,                   # gaussian center (m)
    fiber.length / 10.,  # gaussian width (m)
    Ve=-80.)             # peak extracellular voltage (mV)
section_US_source = SectionAcousticSource(
    fiber.central_ID,  # target section
    500e3,             # US frequency (Hz)
    A=100e3)           # peak acoustic amplitude (Pa)
gaussian_US_source = GaussianAcousticSource(
    0,                   # gaussian center (m)
    fiber.length / 10.,  # gaussian width (m)
    500e3,               # US frequency (Hz)
    A=100e3)             # peak acoustic amplitude (Pa)
transducer_source = PlanarDiskTransducerSource(
    (0., 0., 'focus'),  # transducer position (m)
    500e3,              # US frequency (Hz)
    r=2e-3,             # transducer radius (m)
    u=0.04)             # m/s

# Define pulsing protocols
tpulse = 100e-6  # s
xratio = 0.2     # (-)
toffset = 3e-3   # s
standard_pp = PulsedProtocol(tpulse, toffset)                  # (for US sources)
balanced_pp = BalancedPulsedProtocol(tpulse, xratio, toffset)  # (for electrical sources)

# Define source-protocol pairs
pairs = [
    (iintra_source, balanced_pp),
    (iextra_source, balanced_pp),
    (voltage_source, balanced_pp),
    (section_US_source, standard_pp),
    (gaussian_US_source, standard_pp),
    (transducer_source, standard_pp)
]

# Simulate model with each source-protocol pair, and plot results
for source, pp in pairs:
    data, meta = fiber.simulate(source, pp)
    SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

plt.show()

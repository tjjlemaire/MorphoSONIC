# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-04 16:53:24

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger

from ExSONIC.core import SennFiber
from ExSONIC.core.sources import ExtracellularCurrent, SectionAcousticSource
from ExSONIC.plt import SectionCompTimeSeries

# Set logging level
logger.setLevel(logging.INFO)

# Define point-neuron model
pneuron = getPointNeuron('RS')

# Define sonophore parameters
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)

# Define fiber parameters
fiberD = 20e-6  # m
nnodes = 21

# Create fiber model
fiber = SennFiber(fiberD, nnodes, a=a, fs=fs)

# Define electric and ultrasonic sources
EL_source = ExtracellularCurrent(
    x=(0., fiber.interL),  # m
    I=-0.68e-3)            # A
US_source = SectionAcousticSource(
    fiber.central_ID,  # target section
    500e3,             # Hz
    A=100e3)           # Pa

# Set pulsing protocol
pulse_width = 100e-6  # s
toffset = 3e-3        # s
pp = PulsedProtocol(pulse_width, toffset)

# Simulate model with each source modality and plot results
for source in [EL_source, US_source]:
    data, meta = fiber.simulate(source, pp)
    SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

plt.show()

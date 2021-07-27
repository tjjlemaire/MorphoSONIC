# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:45:54

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol, ElectricDrive, AcousticDrive
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger
from PySONIC.plt import GroupedTimeSeries
from MorphoSONIC.models import Node

# Set logging level
logger.setLevel(logging.INFO)

# Define point-neuron model
pneuron = getPointNeuron('RS')

# Define sonophore parameters
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)

# Create node model
node = Node(pneuron, a=a, fs=fs)

# Define electric and ultrasonic drives
EL_drive = ElectricDrive(20.)  # mA/m2
US_drive = AcousticDrive(
    500e3,  # Hz
    100e3)  # Pa

# Set pulsing protocol
tstim = 250e-3   # s
toffset = 50e-3  # s
PRF = 100.       # Hz
DC = 0.5         # -
pp = PulsedProtocol(tstim, toffset, PRF, DC)

# Simulate model with each drive modality and plot results
for drive in [EL_drive, US_drive]:
    data, meta = node.simulate(drive, pp)
    GroupedTimeSeries([(data, meta)]).render()

# Show figures
plt.show()

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-06-23 12:39:06
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:46:46

import logging
import matplotlib.pyplot as plt
from PySONIC.utils import logger
from PySONIC.core import PulsedProtocol, AcousticDrive
from PySONIC.neurons import getPointNeuron
from MorphoSONIC.plt import plotTimeseries0Dvs1D

logger.setLevel(logging.INFO)

# Parameters
pneuron = getPointNeuron('RS')
a = 32e-9       # m
Fdrive = 500e3  # Hz
Adrive = 100e3   # kPa
deff = 100e-9   # m
rs = 1e2        # Ohm.cm
cov = 0.5
drive = AcousticDrive(
    500e3,  # Hz
    100e3)  # kPa
pp = PulsedProtocol(100e-3, 50e-3)

# Comparative timeseries
fig = plotTimeseries0Dvs1D(pneuron, a, cov, rs, deff, drive, pp)

plt.show()

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-03 18:02:39
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-03 15:55:42

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger
from ExSONIC.core import MRGFiber
from ExSONIC.core.sources import ExtracellularCurrent
from ExSONIC.plt import SectionCompTimeSeries

# import pylustrator
# pylustrator.start()

logger.setLevel(logging.INFO)

# Fiber parameters
fiberD = 10e-6
nnodes = 21
fiber = MRGFiber(fiberD, nnodes, correction_level='axoplasm')

# Stimulation parameters
source = ExtracellularCurrent((0., 100e-6), rho=(300., 1200.))
pp = PulsedProtocol(100e-6, 3e-3)  # s

# Simulation
Ithr = fiber.titrate(source, pp)
data, meta = fiber.simulate(source.updatedX(Ithr), pp)

# Plot
fig = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

plt.show()

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-03 18:02:39
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-03-28 13:49:27

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger
from ExSONIC.core import mrgFiber, IextraMRGFiber
from ExSONIC.core.sources import ExtracellularCurrent
from ExSONIC.plt import SectionCompTimeSeries

# import pylustrator
# pylustrator.start()


logger.setLevel(logging.INFO)

# Fiber parameters
rs = 70.0      # Ohm.cm
nnodes = 21
pneuron = getPointNeuron('MRGnode')
fiberD = 10e-6
fiber = mrgFiber(IextraMRGFiber, pneuron, fiberD, nnodes, rs, correction_level='axoplasm')

# Stimulation parameters
source = ExtracellularCurrent((0., 100e-6), rho=(300., 1200.))
pp = PulsedProtocol(100e-6, 3e-3)  # s

# Simulation
Ithr = fiber.titrate(source, pp)
data, meta = fiber.simulate(source.updatedX(Ithr), pp)

# Plot
fig = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

plt.show()

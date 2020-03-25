# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-03 18:02:39
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-03-25 20:52:08

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger
from ExSONIC.core import mrgFiber, IintraMRGFiber, IextraMRGFiber
from ExSONIC.core.sources import IntracellularCurrent, ExtracellularCurrent
from ExSONIC.plt import SectionCompTimeSeries

logger.setLevel(logging.INFO)

# Fiber parameters
rs = 70.0      # Ohm.cm
nnodes = 21
pneuron = getPointNeuron('MRGnode')
fiberD = 10e-6

# Stimulation parameters
intra_source = lambda fiber: IntracellularCurrent(fiber.nnodes // 2)
extra_source = lambda _: ExtracellularCurrent((0., 100e-6), rho=(300., 1200.))
pp = PulsedProtocol(100e-6, 3e-3)  # s

# Simulations
for fclass, source_func in zip([IintraMRGFiber, IextraMRGFiber], [intra_source, extra_source]):
    fiber = mrgFiber(fclass, pneuron, fiberD, nnodes, rs)
    source = source_func(fiber)
    print(f'fiber model: {fiber}')
    print(fiber.logSectionsDetails())
    Ithr = fiber.titrate(source, pp)
    data, meta = fiber.simulate(source.updatedX(Ithr), pp)
    fig = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

plt.show()

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-03 18:02:39
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:47:00

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger
from MorphoSONIC.models import MRGFiber
from MorphoSONIC.sources import IntracellularCurrent, ExtracellularCurrent
from MorphoSONIC.plt import SectionCompTimeSeries

logger.setLevel(logging.INFO)

# Fiber parameters
fiberD = 10e-6
nnodes = 21
fiber_variants = {
    cl: MRGFiber.__original__(fiberD, nnodes, correction_level=cl)
    for cl in MRGFiber.correction_choices
}

# Stimulation parameters
source_funcs = [
    lambda fiber: ExtracellularCurrent((0., 100e-6), rho=(300., 1200.)),
    lambda fiber: IntracellularCurrent(fiber.central_ID)]
pp = PulsedProtocol(100e-6, 3e-3)  # s

# Simulations
for source_func in source_funcs:
    for cl, fiber in fiber_variants.items():
        source = source_func(fiber)
        Ithr = fiber.titrate(source, pp)
        data, meta = fiber.simulate(source.updatedX(Ithr), pp)
        fig = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

plt.show()

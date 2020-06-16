# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-06-16 11:15:10

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger

from ExSONIC.core import MRGFiber
from ExSONIC.core.sources import *
from ExSONIC.plt import SectionCompTimeSeries

# Set logging level
logger.setLevel(logging.INFO)

# Fiber models
fiberD = 10e-6  # m
nnodes = 21
fiber_classes = {
    'normal': MRGFiber.__original__,
    'sonic': MRGFiber}
fibers = {k: v(fiberD, nnodes) for k, v in fiber_classes.items()}

# Stimulation parameters
source = IntracellularCurrent(fibers['normal'].central_ID, I=1.1e-9)
pp = PulsedProtocol(100e-6, 3e-3, tstart=0.1e-3)

# For each fiber model
for lbl, fiber in fibers.items():
    # Simulate model
    data, meta = fiber.simulate(source, pp)
    # Plot resulting nodal transmembrane voltage traces
    SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodes.keys()).render()


plt.show()

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-05-12 17:24:42

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger

from ExSONIC.core import SennFiber
from ExSONIC.core.sources import GaussianVoltageSource
from ExSONIC.plt import SectionCompTimeSeries

# Set logging level
logger.setLevel(logging.INFO)

# Create fiber models
fiberD = 20e-6  # m
nnodes = 21
fibers = {k: v(fiberD, nnodes) for k, v in {
    'normal': SennFiber.__original__,
    'sonic': SennFiber
}.items()}

# Stimulation parameters
pp = PulsedProtocol(3e-3, 3e-3, tstart=1e-3)

# For each fiber model
for lbl, fiber in fibers.items():
    # Disable use of equivalent currents to ensure that extracellular mechanism is used
    fiber.use_equivalent_currents = False

    # Insert extracellular network in all sections
    for sec in fiber.seclist:
        sec.insertVext(xr=1e10, xg=1e0, xc=0)

    # Define voltage source
    source = GaussianVoltageSource(
        0,                   # gaussian center (m)
        fiber.length / 10.,  # gaussian width (m)
        Ve=-80.              # peak extracellular voltage (mV)
    )

    # Simulate model
    data, meta = fiber.simulate(source, pp)

    # Plot resulting voltage traces (transmembrane and extracellular)
    for k in ['Vm', 'Vext']:
        fig = SectionCompTimeSeries([(data, meta)], k, fiber.nodeIDs).render()
        fig.axes[0].set_title(f'{lbl} fiber - {k}')


plt.show()

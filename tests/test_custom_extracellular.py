# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-06-15 16:24:20

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger

from ExSONIC.core import MRGFiber
from ExSONIC.core.sources import *
from ExSONIC.plt import SectionCompTimeSeries

# Set logging level
logger.setLevel(logging.INFO)

# Create fiber models
fiberD = 10e-6  # m
nnodes = 21
fiber_classes = {
    'normal': MRGFiber.__original__,
    'sonic': MRGFiber}
fibers = {k: v(fiberD, nnodes) for k, v in fiber_classes.items()}
ref_fiber = fibers[list(fibers.keys())[0]]

# Stimulation parameters
source = IntracellularCurrent(ref_fiber.central_ID, I=1.1e-9)
pp = PulsedProtocol(100e-6, 3e-3, tstart=0.1e-3)

data, meta = {}, {}
# For each fiber model
for lbl, fiber in fibers.items():
    # Simulate model
    data[lbl], meta[lbl] = fiber.simulate(source, pp)

for lbl in data.keys():
    # Plot resulting voltage traces (transmembrane and extracellular)
    var_keys = ['Vm', 'Vext'] if fiber.has_ext_mech else ['Vm']
    for k in var_keys:
        for stype, sdict in fiber.sections.items():
            if stype == 'node':
                fig = SectionCompTimeSeries([(data[lbl], meta[lbl])], k, sdict.keys()).render()
                fig.axes[0].set_title(f'{lbl} fiber - {stype}s {k} profiles')


plt.show()

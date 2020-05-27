# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-05-27 17:34:19

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger

from ExSONIC.core import SennFiber, MRGFiber
from ExSONIC.core.sources import *
from ExSONIC.plt import SectionCompTimeSeries

# Set logging level
logger.setLevel(logging.INFO)

# Create fiber models
fiberD = 20e-6  # m
nnodes = 3
# fiber_class = SennFiber
fiber_class = MRGFiber

fibers = {k: v(fiberD, nnodes) for k, v in {
    'normal': fiber_class.__original__,
    'sonic': fiber_class
}.items()}

# Stimulation parameters
# source = ExtracellularCurrent(
#     (0., fibers['normal'].interL),  # electrode position (mm)
#     rho=300.0,                      # extracellular resistivity (Ohm.cm)
#     mode='cathode',                 # electrode polarity
#     I=-0.8e-6
# )
source = IntracellularCurrent(fibers['normal'].central_ID, I=4e-9)
pp = PulsedProtocol(100e-6, 3e-3, tstart=0.1e-3)

# source = GaussianVoltageSource(
#     0,                              # gaussian center (m)
#     fibers['normal'].length / 10.,  # gaussian width (m)
#     Ve=-80.                         # peak extracellular voltage (mV)
# )
# pp = PulsedProtocol(3e-3, 3e-3, tstart=1e-3)


# For each fiber model
for lbl, fiber in fibers.items():
    # Disable use of equivalent currents to ensure that extracellular mechanism is used
    # fiber.use_equivalent_currents = False

    # Insert extracellular network in all sections
    # for sec in fiber.seclist:
    #     sec.insertVext(xr=1e20, xg=1e2, xc=0)
    for sec in fiber.seclist:
        print(sec.__class__.__name__, sec)

    # Simulate model
    data, meta = fiber.simulate(source, pp)

    # Plot resulting voltage traces (transmembrane and extracellular)
    var_keys = ['Vm', 'Vext'] if fiber.has_ext_mech else ['Vm']
    for k in var_keys:
        for stype, sdict in fiber.sections.items():
            if stype == 'node':
                fig = SectionCompTimeSeries([(data, meta)], k, sdict.keys()).render()
                fig.axes[0].set_title(f'{lbl} fiber - {stype}s {k} profiles')


plt.show()

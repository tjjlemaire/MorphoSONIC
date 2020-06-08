# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-06-08 20:04:53

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
fiberD = 10e-6  # m
nnodes = 11
fiber_class = SennFiber
nnodes = 5
fiber_class = MRGFiber

fiber_classes = {
    'Q-based normal': fiber_class.__original__,
    'Q-based sonic': fiber_class
}
if hasattr(fiber_class, '__originalVbased__'):
    fiber_classes['V-based normal'] = fiber_class.__originalVbased__
fibers = {k: v(fiberD, nnodes) for k, v in fiber_classes.items()}
ref_fiber = list(fibers.values())[0]

# Stimulation parameters
# source = ExtracellularCurrent(
#     (0., ref_fiber.length.interL),  # electrode position (mm)
#     rho=300.0,                      # extracellular resistivity (Ohm.cm)
#     mode='cathode',                 # electrode polarity
#     I=-0.8e-6
# )
# source = IntracellularCurrent(ref_fiber.central_ID, I=1e-9)
# pp = PulsedProtocol(100e-6, 3e-3, tstart=0.1e-3)

source = GaussianVoltageSource(
    0,                       # gaussian center (m)
    ref_fiber.length / 10.,  # gaussian width (m)
    Ve=-80.                  # peak extracellular voltage (mV)
)
pp = PulsedProtocol(3e-3, 3e-3, tstart=1e-3)

data, meta = {}, {}
# For each fiber model
for lbl, fiber in fibers.items():

    # Disable use of equivalent currents to ensure that extracellular mechanism is used
    # fiber.use_equivalent_currents = False

    # If required: insert extracellular network in all sections
    if fiber_class != MRGFiber:
        for sec in fiber.seclist:
            sec.insertVext(xr=1e5, xg=1e3, xc=1e5)

    # Simulate model
    data[lbl], meta[lbl] = fiber.simulate(source, pp)

compkey = 'comp'
data[compkey] = data['Q-based sonic'] - data['Q-based normal']
meta[compkey] = meta['Q-based normal']

for lbl in data.keys():
    # Plot resulting voltage traces (transmembrane and extracellular)
    var_keys = ['Vm', 'Vext'] if fiber.has_ext_mech else ['Vm']
    for k in var_keys:
        for stype, sdict in fiber.sections.items():
            if stype == 'node':
                fig = SectionCompTimeSeries([(data[lbl], meta[lbl])], k, sdict.keys()).render()
                fig.axes[0].set_title(f'{lbl} fiber - {stype}s {k} profiles')


plt.show()

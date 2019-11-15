# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-11-15 11:18:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-15 18:03:24

import os
import logging
import csv
import numpy as np

from PySONIC.core import PulsedProtocol
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format, fileCache, selectDirDialog
from ExSONIC.core import IextraFiber, unmyelinatedFiber, ExtracellularCurrent

logger.setLevel(logging.INFO)

# Fiber model parameters
pneuron = getPointNeuron('sundt')          # C-fiber membrane equations
fiberD = 0.8e-6                            # peripheral axon diameter, from Sundt 2015 (m)
rho_a = 100.0                              # axoplasm resistivity, from Sundt 2015 (Ohm.cm)
fiberL = 1e-2                              # axon length (m)
maxNodeL_range = np.logspace(-5, -3, 100)  # maximum node length range

# Stimulation parameters
pp = PulsedProtocol(100e-6, 3e-2)          # short simulus
psource = ExtracellularCurrent((0, 1e-2))  # point-source located 1 cm above central node

# Select output directory
try:
    outdir = selectDirDialog(title='Select output directory')
except ValueError as err:
    logger.error(err)
    quit()

# Create log file
fname = 'convergence_results.csv'
fpath = os.path.join(outdir, fname)
delimiter = '\t'
logger.info(f'creating log file: "{fpath}"')
with open(fpath, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=delimiter)
    writer.writerow(['nodeL (m)', 'Ithr (A)'])

# Loop through parameter sweep
logger.info('running maxNodeL parameter sweep ({}m - {}m)'.format(
    *si_format([maxNodeL_range.min(), maxNodeL_range.max()], 2)))
for x in maxNodeL_range:

    # Initialize fiber with specific max node length
    fiber = unmyelinatedFiber(IextraFiber, pneuron, fiberD, rs=rho_a, fiberL=fiberL, maxNodeL=x)
    nodeL = fiber.nodeL  # m

    # Perform titration to find threshold current
    logger.info('running titration with maxNodeL = {}m (nodeL = {}m)'.format(*si_format([x, nodeL], 2)))
    Ithr = fiber.titrate(psource, pp)  # A
    logger.info(f'Ithr = {si_format(Ithr, 2)}A')

    # Log input-output pair into file
    logger.info('saving result to log file')
    with open(fpath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerow([nodeL, Ithr])

    # Clear fiber sections
    fiber.clear()

logger.info('parameter sweep successfully completed')

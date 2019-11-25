# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-11-15 11:18:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-25 12:18:22

import os
import logging
import csv
import numpy as np

from PySONIC.core import PulsedProtocol
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format, fileCache, selectDirDialog
from ExSONIC.core import IintraFiber, unmyelinatedFiber, IntracellularCurrent

logger.setLevel(logging.INFO)

# Unmyelinated fiber model parameters
pneuron = getPointNeuron('sundt')          # C-fiber membrane equations
fiberD = 0.8e-6                            # peripheral axon diameter, from Sundt 2015 (m)
rho_a = 1e2                                # axoplasm resistivity, from Sundt 2015 (Ohm.cm)
fiberL = 5e-3                              # axon length (m)
maxNodeL_range = np.logspace(-5, -3, 100)  # maximum node length range: from 10 um to 1 mm

# Stimulation parameters
pp = PulsedProtocol(1e-3, 10e-3)

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
    writer.writerow(['maxNodeL (m)', 'nodeL (m)', 'Ithr (A)'])

# Loop through parameter sweep
logger.info('running maxNodeL parameter sweep ({}m - {}m)'.format(
    *si_format([maxNodeL_range.min(), maxNodeL_range.max()], 2)))
for x in maxNodeL_range[::-1]:

    # Initialize fiber with specific max node length
    logger.info(f'creating model with maxNodeL = {x * 1e6:.2f} um ...')
    fiber = unmyelinatedFiber(
        IintraFiber, pneuron, fiberD, rs=rho_a, fiberL=fiberL, maxNodeL=x)
    logger.info(f'resulting node length: {fiber.nodeL * 1e6:.2f} um')

    # Perform titration to find threshold current
    psource = IntracellularCurrent(fiber.nnodes // 2)
    logger.info(f'running titration with intracellular current injected at node {psource.inode}')
    Ithr = fiber.titrate(psource, pp)  # A
    if not np.isnan(Ithr):
        logger.info(f'Ithr = {si_format(Ithr, 2)}A')

    # Log input-output pair into file
    logger.info('saving result to log file')
    with open(fpath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerow([x, fiber.nodeL, Ithr])

    # Clear fiber sections
    fiber.clear()

logger.info('parameter sweep successfully completed')

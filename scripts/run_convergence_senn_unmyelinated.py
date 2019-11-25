# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-11-15 11:18:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-25 13:18:59

import os
import logging
import matplotlib.pyplot as plt
import csv
import numpy as np

from PySONIC.core import PulsedProtocol
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format, fileCache, selectDirDialog
from ExSONIC.core import IintraFiber, unmyelinatedFiber, IntracellularCurrent
from ExSONIC.plt import SectionCompTimeSeries

logger.setLevel(logging.INFO)

# Unmyelinated fiber model parameters
pneuron = getPointNeuron('sundt')          # C-fiber membrane equations
fiberD = 0.8e-6                            # peripheral axon diameter, from Sundt 2015 (m)
rho_a = 1e2                                # axoplasm resistivity, from Sundt 2015 (Ohm.cm)
fiberL = 5e-3                              # axon length (m)
maxNodeL_range = np.logspace(-5, -3, 100)  # maximum node length range: from 10 um to 1 mm

# Stimulation parameters
pp = PulsedProtocol(1e-3, 15e-3)

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
    writer.writerow(['maxNodeL (m)', 'nodeL (m)', 'Ithr (A)', 'CV (m/s)', 'dV (mV)'])

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

    # If fiber is excited
    if not np.isnan(Ithr):
        logger.info(f'Ithr = {si_format(Ithr, 2)}A')

        # Simulate fiber at 1.1 times threshold current
        data, meta = fiber.simulate(psource, 1.1 * Ithr, pp)

        # Compute CV and spike amplitude
        try:
            cv = fiber.getConductionVelocity(data, out='median')  # m/s
            dV = fiber.getSpikeAmp(data, out='median')            # mV
            logger.info(f'CV = {cv:.2f} m/s')
            logger.info(f'dV = {dV:.2f} mV')
        except AssertionError as err:
            # Plot membrane potential traces for specific duration at threshold current
            fig = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()
            plt.show()
    else:
        # Otherwise, assign NaN values to them
        cv, dV = np.nan, np.nan

    # Log input-output pair into file
    logger.info('saving result to log file')
    with open(fpath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerow([x, fiber.nodeL, Ithr, cv, dV])

    # Clear fiber sections
    fiber.clear()

logger.info('parameter sweep successfully completed')

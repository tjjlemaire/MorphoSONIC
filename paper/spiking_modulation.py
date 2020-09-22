# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-09-22 17:35:12

import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger, si_format

from ExSONIC.core import SennFiber
from ExSONIC.core.sources import GaussianAcousticSource
from ExSONIC.plt import SectionCompTimeSeries, spatioTemporalMap

# Set logging level
logger.setLevel(logging.INFO)


def getPulseTrain(tpulse, npulses, PRF):
    ''' Get a pulse train protocol for a given pulse duration, number of pulses and PRF.

        :param tpulse: pulse duration (s)
        :param npulses: number of pulses
        :param PRF: pulse repetition frequency (Hz)
        :return: PulsedProtocol object
    '''
    DC = tpulse * PRF
    tstim = npulses / PRF
    tstart = 1 / PRF - tpulse
    return PulsedProtocol(tstim + tstart, 0., PRF=PRF, DC=DC, tstart=tstart)


# Plot parameters
fontsize = 10

# Fiber model
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)
fiberD = 20e-6  # m
nnodes = 21
fiber = SennFiber(fiberD, nnodes, a=a, fs=fs)

# Acoustic source
Fdrive = 500e3  # Hz
Adrive = 300e3  # Pa
w = 5e-3  # m
sigma = GaussianAcousticSource.from_FWHM(w)  # m
source = GaussianAcousticSource(0, sigma, Fdrive, Adrive)

# Pulsing parameters
npulses = 10
tpulse = 100e-6  # s
PRF_max = 0.99 / tpulse  # Hz
PRF_min = PRF_max / 10  # Hz
nPRF = 6
PRFs = np.logspace(np.log10(PRF_min), np.log10(PRF_max), nPRF)

# For each pulse duration - PRF combination:
for PRF in PRFs:
    # Get pulsing protocol
    pp = getPulseTrain(tpulse, npulses, PRF)

    # Simulate model
    data, meta = fiber.simulate(source, pp)

    # Plots resulting Qm timeseries and spatiotemporal maps
    # SectionCompTimeSeries([(data, meta)], 'Qm', fiber.nodeIDs).render()
    fig = spatioTemporalMap(fiber, data, 'Qm', fontsize=fontsize)
    # fig.suptitle(f'PD = {si_format(tpulse, 1)}s, PRF = {si_format(PRF, 1)}Hz', fontsize=fontsize)

plt.show()

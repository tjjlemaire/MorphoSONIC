# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-09-22 20:16:33

import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger, si_format

from ExSONIC.core import SennFiber
from ExSONIC.core.sources import GaussianAcousticSource
from ExSONIC.plt import spatioTemporalMap

# Set logging level
logger.setLevel(logging.INFO)


def getPulseTrain(PD, npulses, PRF):
    ''' Get a pulse train protocol for a given pulse duration, number of pulses and PRF.

        :param PD: pulse duration (s)
        :param npulses: number of pulses
        :param PRF: pulse repetition frequency (Hz)
        :return: PulsedProtocol object
    '''
    DC = PD * PRF
    tstim = npulses / PRF
    tstart = 1 / PRF - PD
    return PulsedProtocol(tstim + tstart, 0., PRF=PRF, DC=DC, tstart=tstart)


# Plot parameters
fontsize = 10
plot_spikes = True

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
PDs = [100e-6, 200e-6, 500e-6, 1e-3]  # s
nPRF = 6

PRFs = {}
FRs = {}

# For each pulse duration
for PD in PDs:
    PD_key = f'PD = {si_format(PD, 1)}s'

    # Compute relevant PRF range
    PRF_max = 0.99 / PD  # Hz
    PRF_min = PRF_max / 100  # Hz
    PRFs[PD_key] = np.logspace(np.log10(PRF_min), np.log10(PRF_max), nPRF)

    FRs[PD_key] = []

    # For each PRF
    for PRF in PRFs[PD_key]:
        PRF_key = f'PRF = {si_format(PRF, 1)}Hz'

        # Get pulsing protocol
        pp = getPulseTrain(PD, npulses, PRF)

        # Simulate model
        data, meta = fiber.simulate(source, pp)

        # Plots resulting Qm timeseries and spatiotemporal maps
        fig = spatioTemporalMap(fiber, data, 'Qm', fontsize=fontsize, plot_spikes=plot_spikes)
        # fig.suptitle(', '.join(PD_key, PRF_key), fontsize=fontsize)

        # Detect spikes on end node
        tspikes = fiber.getEndSpikeTrain(data)
        if tspikes is not None:
            FR = tspikes.size / pp.tstim
        else:
            FR = np.nan
        FRs[PD_key].append(FR)

    FRs[PD_key] = np.array(FRs[PD_key])

FR_fig, FR_ax = plt.subplots()
FR_ax.set_xscale('log')
FR_ax.set_yscale('log')
FR_ax.set_xlabel('PRF (Hz)')
FR_ax.set_ylabel('FR (Hz)')
for k in FRs.keys():
    FR_ax.plot(PRFs[k], FRs[k], label=k)
xlims = FR_ax.get_xlim()
FR_ax.plot(xlims, xlims, '--', c='k')
FR_ax.legend(frameon=False)
plt.show()

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-09-25 19:08:48

import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger, si_format

from ExSONIC.core import SennFiber, UnmyelinatedFiber
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


# Fiber models
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)
fibers = {
    # 'unmyelinated': UnmyelinatedFiber(0.8e-6, fiberL=5e-3, a=a, fs=fs),
    'myelinated': SennFiber(10e-6, 21, a=a, fs=fs),
}

# Charactersitic chronaxies
chronaxies = {
    'myelinated': 76e-6,  # s
    'unmyelinated': 8e-3  # s
}

# US parameters
Fdrive = 500e3  # Hz
Adrive = 300e3  # Pa

# Pulsing parameters
npulses = 10
nPD = 1
nPRF = 1
ncombs = len(fibers) * nPD * nPRF

# Plot parameters
fontsize = 10
plot_spikes = True
colors = plt.get_cmap('tab20c').colors
colors = dict(zip(fibers.keys(), [colors[:3], colors[4:7]]))

PRFs = {}
FRs = {}
for k, fiber in fibers.items():

    # Define acoustic source
    w = fiber.length / 5  # m
    sigma = GaussianAcousticSource.from_FWHM(w)  # m
    source = GaussianAcousticSource(0, sigma, Fdrive, Adrive)

    # Define pulse duration range
    PDs = np.logspace(-1, 1, nPD) * chronaxies[k]  # s

    # For each pulse duration
    PRFs[k] = {}
    FRs[k] = {}
    for PD in PDs:
        PD_key = f'PD = {si_format(PD, 1)}s'

        # Define PRF range
        PRF_max = 0.99 / PD  # Hz
        PRF_min = PRF_max / 100  # Hz
        PRFs[k][PD_key] = np.logspace(np.log10(PRF_min), np.log10(PRF_max), nPRF)
        FRs[k][PD_key] = []

        # For each PRF
        for PRF in PRFs[k][PD_key]:
            PRF_key = f'PRF = {si_format(PRF, 1)}Hz'

            # Get pulsing protocol
            pp = getPulseTrain(PD, npulses, PRF)

            # Simulate model
            data, meta = fiber.simulate(source, pp)

            # Plots resulting Qm timeseries and spatiotemporal maps
            if ncombs <= 15:
                fig = spatioTemporalMap(
                    fiber, source, data, 'Qm', fontsize=fontsize, plot_spikes=plot_spikes)
                # fig.suptitle(', '.join(PD_key, PRF_key), fontsize=fontsize)

            # Detect spikes on end node
            tspikes = fiber.getEndSpikeTrain(data)

            # Compute firing rate
            if tspikes is not None:
                FR = tspikes.size / pp.tstim
            else:
                FR = np.nan
            FRs[k][PD_key].append(FR)

        FRs[k][PD_key] = np.array(FRs[k][PD_key])

# Plot FR vs PRF across cell types for various PDs
FR_fig, FR_ax = plt.subplots()
FR_ax.set_xscale('log')
FR_ax.set_yscale('log')
FR_ax.set_xlabel('PRF (Hz)')
FR_ax.set_ylabel('FR (Hz)')
for k, FRdict in FRs.items():
    clist = colors[k]
    for c, (PD_key, FR) in zip(clist, FRdict.items()):
        FR_ax.plot(PRFs[k][PD_key], FR, label=f'{k} - {PD_key}', c=c)
xlims = FR_ax.get_xlim()
FR_ax.plot(xlims, xlims, '--', c='k')
FR_ax.legend(frameon=False)
plt.show()

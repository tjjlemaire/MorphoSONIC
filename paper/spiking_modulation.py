# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-09-30 22:26:27

import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.core import getPulseTrainProtocol, LogBatch
from PySONIC.utils import logger, si_format
from PySONIC.plt import setNormalizer, XYMap

from ExSONIC.core import SennFiber, UnmyelinatedFiber
from ExSONIC.core.sources import GaussianAcousticSource
from ExSONIC.plt import spatioTemporalMap

from root import datadir

# Set logging level
logger.setLevel(logging.INFO)


class FRvsPRFBatch(LogBatch):

    in_key = 'PRF'
    out_keys = ['FR']
    suffix = 'FR'
    unit = 'Hz'

    def __init__(self, fiber, source, PD, npulses, PRFs=None, nPRF=10, **kwargs):
        self.fiber = fiber
        self.source = source
        self.PD = PD
        self.npulses = npulses
        if PRFs is None:
            PRFs = self.getPRFrange(nPRF)
        super().__init__(PRFs, **kwargs)

    def compute(self, PRF):
        pp = getPulseTrainProtocol(self.PD, self.npulses, PRF)
        data, meta = fiber.simulate(source, pp)
        tspikes = fiber.getEndSpikeTrain(data)
        if tspikes is None:
            return np.nan
        return np.mean(1 / np.diff(tspikes))

    @property
    def sourcecode(self):
        codes = self.source.filecodes
        return f'{self.source.key}_{"_".join(codes.values())}'

    def corecode(self):
        return f'FRvsPRF_{self.fiber.modelcode}_{self.sourcecode}_PD{si_format(self.PD, 1)}s'

    def getPRFrange(self, n):
        ''' Get pulse-duration-dependent PRF range. '''
        PRF_max = 0.99 / self.PD  # Hz
        PRF_min = PRF_max / 100  # Hz
        return np.logspace(np.log10(PRF_min), np.log10(PRF_max), n)


class NormalizedFiringRateMap(XYMap):

    xkey = 'duty cycle'
    xfactor = 1e0
    xunit = '-'
    ykey = 'amplitude'
    yfactor = 1e0
    yunit = 'Pa'
    zkey = 'normalized firing rate'
    zunit = '-'
    zfactor = 1e0
    suffix = 'FRmap'

    def __init__(self, fiber, source, DCs, amps, npulses, PRF, root='.'):
        self.fiber = fiber
        self.source = source
        self.PRF = PRF
        self.npulses = npulses
        super().__init__(root, DCs, amps)

    @property
    def sourcecode(self):
        codes = self.source.filecodes
        if 'A' in codes:
            del codes['A']
        return f'{self.source.key}_{"_".join(codes.values())}'

    def corecode(self):
        return f'normFRmap_{self.fiber.modelcode}_{self.sourcecode}_PRF{si_format(self.PRF, 1)}Hz'

    @property
    def title(self):
        return f'Normalized firing rate map - {self.fiber}, {self.source}, {si_format(self.PRF)}Hz PRF'

    def compute(self, x):
        DC, A = x
        source.A = A
        pp = getPulseTrainProtocol(DC / self.PRF, self.npulses, self.PRF)
        data, meta = fiber.simulate(source, pp)
        tspikes = fiber.getEndSpikeTrain(data)
        if tspikes is None:
            return np.nan
        return np.mean(1 / np.diff(tspikes)) / self.PRF


def plotFRvsPRF(PRFs, FRs, cmaps):
    ''' Plot FR vs PRF across cell types for various PDs. '''
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('pulse repetition frequency (Hz)')
    ax.set_ylabel('firing rate (Hz)')
    PRF_min = 1e0
    PRF_max = max(max(v.max() for v in val.values()) for val in PRFs.values())
    PRF_range = [PRF_min, PRF_max]
    ax.plot(PRF_range, PRF_range, '--', c='k')
    ax.set_xlim(*PRF_range)
    ax.set_ylim(*PRF_range)
    for k, FRdict in FRs.items():
        nPDs = len(FRdict)
        _, sm = setNormalizer(plt.get_cmap(cmaps[k]), (0, 1))
        xstart = 0.2  # avoid white-ish colors
        clist = [sm.to_rgba((1 - xstart) / (nPDs - 1) * i + xstart) for i in range(nPDs)]
        for c, (PD_key, FR) in zip(clist, FRdict.items()):
            lbl = f'{k} - {PD_key}'
            if np.all(np.isnan(FR)):
                print(f'{lbl}: all NaNs')
            else:
                PRF = PRFs[k][PD_key]
                ax.plot(np.hstack(([PRF_min], PRF)), np.hstack(([PRF_min], FR)), label=lbl, c=c)
                ax.axvline(PRF.max(), c=c, linestyle='--')
    ax.legend(frameon=False)
    plt.show()


# Fiber models
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)
fibers = {
    'unmyelinated': UnmyelinatedFiber(0.8e-6, fiberL=5e-3, a=a, fs=fs),
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
nPD = 10
nPRF = 50
ncombs = len(fibers) * nPD * nPRF

# Plot parameters
fontsize = 10
plot_spikes = True
cmaps = {
    'myelinated': 'Blues',
    'unmyelinated': 'Oranges'
}
colors = plt.get_cmap('tab20c').colors
colors = dict(zip(fibers.keys(), [colors[:3], colors[4:7]]))

subsets = {
    'myelinated': [],
    'unmyelinated': []
}

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
        # Run FR batch
        PD_key = f'PD = {si_format(PD, 1)}s'
        frbatch = FRvsPRFBatch(fiber, source, PD, npulses, nPRF=nPRF, root=datadir)
        PRFs[k][PD_key] = frbatch.inputs
        FRs[k][PD_key] = frbatch.run()

    # Plot spatiotemporal maps for fiber-specific subsets
    for PD, PRF in subsets[k]:
        key = f'PD = {si_format(PD, 1)}s, PRF = {si_format(PRF, 1)}Hz'
        pp = getPulseTrainProtocol(PD, npulses, PRF)
        data, meta = fiber.simulate(source, pp)
        fig = spatioTemporalMap(
            fiber, source, data, 'Qm', fontsize=fontsize, plot_spikes=plot_spikes)
        fig.suptitle(key, fontsize=fontsize)

plotFRvsPRF(PRFs, FRs, cmaps)

# map_PRFs = {
#     'myelinated': [100],
#     'unmyelinated': []
# }

# nperax = 5
# DCs = np.linspace(0.01, 1, nperax)
# amps = np.logspace(np.log10(1e0), np.log10(600e3), nperax)
# for k, fiber in fibers.items():
#     # Define acoustic source
#     w = fiber.length / 5  # m
#     sigma = GaussianAcousticSource.from_FWHM(w)  # m
#     source = GaussianAcousticSource(0, sigma, Fdrive)
#     for PRF in map_PRFs[k]:
#         frmap = NormalizedFiringRateMap(fiber, source, DCs, amps, npulses, PRF, root=datadir)
#         frmap.run()
#         frmap.render(yscale='log', zbounds=(0, 1))


plt.show()

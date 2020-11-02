# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-11-02 12:38:40

import os
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.core import getPulseTrainProtocol, LogBatch
from PySONIC.utils import logger, si_format, bounds
from PySONIC.plt import setNormalizer, XYMap

from ExSONIC.core import SennFiber, UnmyelinatedFiber
from ExSONIC.core.sources import GaussianAcousticSource
from ExSONIC.plt import spatioTemporalMap

from root import datadir, figdir

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
        data, meta = fiber.simulate(self.source, pp)
        return fiber.getEndFiringRate(data)

    @property
    def sourcecode(self):
        codes = self.source.filecodes
        return f'{self.source.key}_{"_".join(codes.values())}'

    def corecode(self):
        return f'FRvsPRF_{self.fiber.modelcode}_{self.sourcecode}_PD{si_format(self.PD, 1)}s'

    def getPRFrange(self, n):
        ''' Get pulse-duration-dependent PRF range. '''
        PRF_max = 0.99 / self.PD  # Hz
        PRF_min = max(PRF_max / 100, 10)  # Hz
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
        self.source.A = A
        pp = getPulseTrainProtocol(DC / self.PRF, self.npulses, self.PRF)
        data, meta = fiber.simulate(self.source, pp)
        return fiber.getEndFiringRate(data) / self.PRF

    def onClick(self, event):
        DC, A = self.getOnClickXY(event)
        self.source.A = A
        pp = getPulseTrainProtocol(DC / self.PRF, self.npulses, self.PRF)
        data, meta = fiber.simulate(self.source, pp)
        fig = spatioTemporalMap(self.fiber, self.source, data, 'Qm', fontsize=fontsize)
        plt.show()


def plotFRvsPRF(PRFs, FRs, cmaps):
    ''' Plot FR vs PRF across cell types for various PDs. '''
    fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 3.5))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('pulse repetition frequency (Hz)')
    ax.set_ylabel('firing rate (Hz)')

    PRF_min = 1e0
    PRF_max = 0.
    for key, val in FRs.items():
        for k, v in val.items():
            imax = np.where(~np.isnan(v))[0][-1]
            PRF_max = max(PRFs[key][k][imax], PRF_max)
    PRF_range = np.array([PRF_min, PRF_max])
    ax.plot(PRF_range, PRF_range, '--', c='k')
    for x in [2]:
        ax.plot(PRF_range, PRF_range * x, '--', c='k')
        ax.plot(PRF_range, PRF_range / x, '--', c='k')
    ax.set_xlim(*PRF_range)
    ax.set_ylim(*PRF_range)

    for k, FRdict in FRs.items():
        nPDs = len(FRdict)
        _, sm = setNormalizer(plt.get_cmap(cmaps[k]), (0, 1))
        xstart = 0.4  # avoid white-ish colors
        clist = [sm.to_rgba((1 - xstart) / (nPDs - 1) * i + xstart) for i in range(nPDs)]
        for c, (PD_key, FR) in zip(clist, FRdict.items()):
            lbl = f'{k} - {PD_key}'
            if np.all(np.isnan(FR)):
                logger.info(f'{lbl}: all NaNs')
            else:
                PRF = PRFs[k][PD_key]
                if np.all(~np.isnan(FR[:3])):  # if starts with 3 real values
                    s0, s1, *_ = np.diff(FR) / np.diff(PRF)
                    if np.isclose(s1, s0, atol=1e-2):  # if first 2 slopes are similar
                        # Extend linear range to minimum PRF (assuming to vertical offset)
                        PRF, FR = np.hstack(([PRF_min], PRF)), np.hstack(([s0 * PRF_min], FR))
                ax.plot(PRF, FR, label=lbl, c=c)
                # ax.axvline(PRF.max(), c=c, linestyle='--')
    ax.legend(frameon=False)
    return fig


# Fiber models
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)
fibers = {
    'unmyelinated': UnmyelinatedFiber(0.8e-6, fiberL=5e-3, a=a, fs=fs),
    'myelinated': SennFiber(10e-6, 21, a=a, fs=fs),
}

# # Charactersitic chronaxies
# chronaxies = {
#     'myelinated': 76e-6,  # s
#     'unmyelinated': 8e-3  # s
# }

PDs = {
    'myelinated': np.array([20e-6, 100e-6, 1e-3]),
    'unmyelinated': np.array([1e-3, 5e-3, 10e-3])
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
Qbounds = {
    'myelinated': (-175, 75),
    'unmyelinated': (-80, 36)
}
cmaps = {
    'unmyelinated': 'Blues',
    'myelinated': 'Oranges'
}
subset_colors = {
    'unmyelinated': 'C0',
    'myelinated': 'C1'
}
maponly = False

# Define acoustic sources
sources = {}
for k, fiber in fibers.items():
    w = fiber.length / 5  # m
    sigma = GaussianAcousticSource.from_FWHM(w)  # m
    sources[k] = GaussianAcousticSource(0, sigma, Fdrive, Adrive)

# FR vs PRF batches
PRFs = {}
FRs = {}
for k, fiber in fibers.items():
    # Define pulse duration range
    # PDs = np.logspace(-1, 1, nPD) * chronaxies[k]  # s

    # For each pulse duration
    PRFs[k] = {}
    FRs[k] = {}
    for PD in PDs[k]:
        # Run FR batch
        PD_key = f'PD = {si_format(PD, 1)}s'
        frbatch = FRvsPRFBatch(fiber, sources[k], PD, npulses, nPRF=nPRF, root=datadir)
        PRFs[k][PD_key] = frbatch.inputs
        FRs[k][PD_key] = frbatch.run()

FRfig = plotFRvsPRF(PRFs, FRs, cmaps)

# Spatiotemporal maps for fiber-specific subsets
subsets = {
    'myelinated': [
        (100e-6, 500),
        (100e-6, 2600),
        (20e-6, 5000),
    ],
    'unmyelinated': [
        (10e-3, 15.),
        (5e-3, 15.),
        (1e-3, 400.)
    ]
}
FRax = FRfig.axes[0]
minFR = FRax.get_ylim()[0]
subset_FRs = {}
for k, fiber in fibers.items():
    subset_FRs[k] = []
    for PD, PRF in subsets[k]:
        key = f'PD = {si_format(PD, 1)}s, PRF = {si_format(PRF, 1)}Hz'
        pp = getPulseTrainProtocol(PD, npulses, PRF)
        data, meta = fiber.simulate(sources[k], pp)
        # fpath = fiber.simAndSave(sources[k], pp, overwrite=False, outputdir=datadir)
        # with open(fpath, 'rb') as fh:
        #     frame = pickle.load(fh)
        #     data, meta = frame['data'], frame['meta']
        fig = spatioTemporalMap(
            fiber, sources[k], data, 'Qm', fontsize=fontsize,
            cmap=cmaps[k], zbounds=Qbounds[k], maponly=maponly)
        if not maponly:
            fig.suptitle(key, fontsize=fontsize)
            ext = '.pdf'
            kwargs = {}
        else:
            ext = '.png'
            kwargs = {'dpi': 300}
        fname = f'fig6A_Qmap_{k}_{key}{ext}'.replace(' ', '').replace(',', '_').replace('=', '')
        fig.savefig(os.path.join(figdir, fname), transparent=True, **kwargs)
        subset_FRs[k].append(fiber.getEndFiringRate(data))

    subset_FRs[k] = np.array(subset_FRs[k])  # convert to array
    subset_FRs[k][np.isnan(subset_FRs[k])] = minFR  # convert nans to inferior ylim
    FRax.scatter([x[1] for x in subsets[k]], subset_FRs[k],
                 c=[subset_colors[k]], zorder=2.5)

FRfig.savefig(os.path.join(figdir, 'fig6A_raw.pdf'), transparent=True)

map_PRFs = {
    'myelinated': [500, 2600],
    'unmyelinated': [50., 200.]
}
FRbounds = {
    'myelinated': (0, 1),
    'unmyelinated': (0, 3.1)
}
nperax = 40
DCs = np.linspace(0.0, 1, nperax)
amps = np.logspace(np.log10(10e3), np.log10(600e3), nperax)
for k, fiber in fibers.items():
    for PRF in map_PRFs[k]:
        frmap = NormalizedFiringRateMap(fiber, sources[k], DCs, amps, npulses, PRF, root=datadir)
        frmap.run()
        fig = frmap.render(yscale='log', cmap=cmaps[k], zbounds=FRbounds[k], interactive=True)
        code = f'fig6B_normFRmap_{k}_PRF_{si_format(PRF, 1)}Hz.pdf'
        fig.savefig(os.path.join(figdir, code), transparent=True)

plt.show()

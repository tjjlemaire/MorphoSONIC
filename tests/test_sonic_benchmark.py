# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-08-07 20:09:58

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger

from ExSONIC.core import SennFiber, UnmyelinatedFiber, MRGFiber
from ExSONIC.core.sources import *
from ExSONIC.plt import SectionCompTimeSeries, mergeFigs, plotFieldDistribution

# Set logging level
logger.setLevel(logging.DEBUG)


def compareResponses(fiber, source, pp, varkeys, **kwargs):
    # Run simulations and cycle-average solutions
    detailed = {'sonic': False, 'full': True}
    data, meta = {}, {}
    for k, v in detailed.items():
        data[k], meta[k] = fiber.benchmark(detailed=v).simulate(source, pp)
        cavg_spdf = data[k].cycleAveraged(1 / source.f)
        cavg_spdf.prepend()
        for sectionk in cavg_spdf.keys():
            cavg_spdf[sectionk]['Vm'][0] = fiber.pneuron.Vm0
            cavg_spdf[sectionk]['Cm'][0] = fiber.pneuron.Cm0
        data[f'cycle-avg-{k}'] = cavg_spdf
        meta[f'cycle-avg-{k}'] = meta[k]

    # Generate comparison figures
    for i, vk in enumerate(varkeys):
        figs = {}
        for k, d in data.items():
            figs[k] = SectionCompTimeSeries([(d, meta[k])], vk, fiber.nodes.keys()).render(**kwargs)
        fig = mergeFigs(figs['full'], figs['sonic'], alpha=0.2, inplace=True)
        fig.axes[0].set_title(f'{vk} - comparison')

        fig = mergeFigs(figs['cycle-avg-full'], figs['cycle-avg-sonic'], alpha=1.0, inplace=True)
        fig.axes[0].set_title(f'{vk} - cycle-avg-comparison')


# Acoustic source
Fdrive = 500e3  # Hz
Adrive = 300e3  # Pa
sigma = GaussianAcousticSource.from_FWHM(1e-3)  # m
source = GaussianAcousticSource(0., sigma=sigma, f=Fdrive, A=Adrive)

# Pulsing protocol
pp = PulsedProtocol(1e-3, 0.1e-3)

# Plot params
varkeys = ['Qm', 'Vm']

# Fiber models
a = 32e-9       # m
fs = 1.
# fiber = SennFiber(10e-6, 5, a=a, fs=fs)
fiber = UnmyelinatedFiber(0.8e-6, nnodes=2, a=a, fs=fs)
# fs_range = [0.1, 0.2, 0.5, 1.0]
fiber = MRGFiber(10e-6, 2, a=a, fs=fs, inter_fs=0.1)
# compareResponses(fiber, source, pp, varkeys)
# inter_fs = 0.5
# for inter_fs in [0.01, 0.5, 1.0]:
#     fiber = MRGFiber(10e-6, 2, a=a, fs=fs, inter_fs=inter_fs)
plotFieldDistribution(fiber, source)
compareResponses(fiber, source, pp, varkeys, cmap='tab10')

plt.show()

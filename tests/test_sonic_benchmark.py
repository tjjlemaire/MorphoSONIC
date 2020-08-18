# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-08-18 15:52:21

import logging
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger
from ExSONIC.core import SennFiber, UnmyelinatedFiber, MRGFiber
from ExSONIC.core.sources import *
from ExSONIC.plt import SectionCompTimeSeries, mergeFigs, plotFieldDistribution

# Set logging level
logger.setLevel(logging.DEBUG)


def getDualCmap(cmap_key):
    ''' Restrict a qualitative colormap to its first 2 colors. '''
    return LinearSegmentedColormap.from_list(
        cmap_key, plt.get_cmap(cmap_key).colors[:2])


def getLinearGammaSource(fiber, gamma_bounds, **kwargs):
    ''' Generate a linear gamma source along a fiber. '''
    xmin, xmax = fiber.getXBounds()
    m = np.diff(gamma_bounds) / (xmax - xmin)
    p = gamma_bounds[0] - m * xmin
    gamma_dict = {k: m * v + p for k, v in fiber.getXCoords().items()}
    return GammaSource(gamma_dict, **kwargs)


def cycleAverageSolution(data, fiber, source):
    ''' Cycle-average a simulation output '''
    cavg_data = data.cycleAveraged(1 / source.f)
    cavg_data.prepend()
    for sectionk in cavg_data.keys():
        cavg_data[sectionk]['Vm'][0] = fiber.pneuron.Vm0
        cavg_data[sectionk]['Cm'][0] = fiber.pneuron.Cm0
    return cavg_data


def compareSolutions(fiber, source, pp, varkeys, **kwargs):
    ''' Run SONIC and full simulations and compare outputs. '''
    # Run simulations
    detailed = {'sonic': False, 'full': True}
    data, meta = {}, {}
    for k, v in detailed.items():
        data[k], meta[k] = fiber.benchmark(detailed=v).simulate(source, pp)

    # Cycle-average full solution
    k = 'full'
    data[f'cycle-avg-{k}'] = cycleAverageSolution(data[k], fiber, source)
    meta[f'cycle-avg-{k}'] = meta[k]

    # Generate comparison figures
    for i, vk in enumerate(varkeys):
        figs = {}
        for k, d in data.items():
            figs[k] = SectionCompTimeSeries([(d, meta[k])], vk, fiber.nodes.keys()).render(**kwargs)
        fig = mergeFigs(
            figs['full'], figs['cycle-avg-full'], figs['sonic'],
            alphas=[0.2, 1.0, 1.0], linestyles=['-', '--', '-'], inplace=True)
        fig.axes[0].set_title(f'{vk} - comparison')


# Gamma source
Fdrive = 500e3             # Hz
gamma_bounds = (0.8, 0.1)  # (-)

# Pulsing protocol
pp = PulsedProtocol(1e-3, 0.1e-3)

# Plot params
varkeys = ['Qm', 'Vm']

# Fiber models
a = 32e-9       # m
fs = 1.
inter_fs = 0.1
nnodes = 2
fiber = SennFiber(10e-6, nnodes, a=a, fs=fs)
# fiber = UnmyelinatedFiber(0.8e-6, nnodes=nnodes, a=a, fs=fs)
# fiber = MRGFiber(10e-6, 2, a=a, fs=fs, inter_fs=inter_fs)

source = getLinearGammaSource(fiber, gamma_bounds, f=Fdrive)
plotFieldDistribution(fiber, source)
compareSolutions(fiber, source, pp, varkeys, cmap=getDualCmap('tab10'))

plt.show()

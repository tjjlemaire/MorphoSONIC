# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-09-24 13:42:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-09-28 09:41:04

import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.utils import logger, expandRange, bounds
from PySONIC.core import BilayerSonophore
from PySONIC.neurons import passiveNeuron

from ExSONIC.core import SennFiber, UnmyelinatedFiber
from PySONIC.multicomp import GammaMap, SonicBenchmark, ModelDivergenceMap, GammaDivergenceMap
# from ExSONIC.core.benchmark import *

logger.setLevel(logging.INFO)

# Parameters
root = 'data'
nperax = 20
a = 32e-9  # m
Fdrive = 500e3  # Hz
freqs = np.logspace(np.log10(20e3), np.log10(4e6), nperax)  # Hz
amps = np.logspace(3, 6, 20)  # kPa
default_gammas = np.array([1.2, 0.8])
default_ga = 1e0  # dummy value (S/m2)
default_gm = 1e0  # dummy value (S/m2)
passive = True
div_criterion = 'rmse'
tau_bounds = (1e-7, 1e-2)  # s
div_levels = [0.1, 1.0, 10.0]  # mV
tau_range = np.logspace(*np.log10(tau_bounds), nperax)  # s

# Fibers
fibers = {
    'myelinated': SennFiber(10e-6, 11),
    'unmyelinated': UnmyelinatedFiber(0.8e-6)
}

# Cell-type-specific simulation times
tstops = {
    'myelinated': 1e-3,  # s
    'unmyelinated': 10e-3  # s
}

# Cell-type-specific gamma maps
# rheobase_gammas = {
#     'myelinated': 0.32,
#     'unmyelinated': 0.30
# }
# fig = plt.figure(constrained_layout=True, figsize=(7, 3.5))
# graph_cbar_width_ratio = 6
# gs = fig.add_gridspec(1, 2 * graph_cbar_width_ratio + 1)
# subplots = {
#     'a': gs[:, :graph_cbar_width_ratio],
#     'b': gs[:, graph_cbar_width_ratio:2 * graph_cbar_width_ratio],
#     'c': gs[:, 2 * graph_cbar_width_ratio]
# }
# axes = {k: fig.add_subplot(v) for k, v in subplots.items()}
# for ax, (k, fiber) in zip(axes.values(), fibers.items()):
#     bls = BilayerSonophore(a, fiber.pneuron.Cm0, fiber.pneuron.Qm0)
#     Qm = fiber.pneuron.Qm0
#     gamma_map = GammaMap(root, bls, Qm, freqs, amps)
#     gamma_map.run()
#     # gamma_map.toPickle(root)
#     levels = sorted(np.hstack(([rheobase_gammas[k]], default_gammas)))
#     gamma_map.render(zbounds=(0, 1.7), levels=levels, ax=ax, cbarax=axes['c'], fs=10, title=k)

# Cell-type-specific benchmarks
# for k, fiber in fibers.items():
#     ga = fiber.ga_node_to_node * 1e4  # S/m2
#     bm = SonicBenchmark(fiber.pneuron, ga, Fdrive, default_gammas, passive=False)
#     bm.tauax = 1 / bm.Fdrive
#     t, sol = bm.simAllMethods(tstops[k])
#     bm.plotQnorm(t, sol)
#     bm.phaseplotQnorm(t, sol)
#     bm.logDivergences(t, sol)

# Passive model
Cm0 = 1e-2  # F/m2
Vm0 = -70.0  # mV
pneuron = passiveNeuron(Cm0, default_gm, Vm0)
# bm = SonicBenchmark(pneuron, default_ga, Fdrive, default_gammas, passive=True)
# bm.taum = 1.62e-3     # s
# bm.tauax = 483.29e-6  # s
# t, sol = bm.simAllMethods(bm.passive_tstop)
# bm.plot(t, sol)
# bm.logDivergences(t, sol)

# Passive divergence maps
fgamma_pairs = {'default': (Fdrive, default_gammas)}
for k, s in zip(['low gamma', 'high gamma'], [-1, 1]):  # vary gamma amplitudes
    fgamma_pairs[k] = (Fdrive, default_gammas + s * 0.2)
for k, x in zip(['low gradient', 'high gradient'], [0.5, 2]):  # vary gamma gradient
    fgamma_pairs[k] = (Fdrive, np.array(expandRange(*default_gammas[::-1], exp_factor=x))[::-1])
for k, f in zip(['low f', 'high f'], bounds(freqs)):  # vary Fdrive
    fgamma_pairs[k] = (f, default_gammas)
insets = {}
for k, fiber in fibers.items():
    gax = fiber.ga_node_to_node * 1e4  # S/m2
    gm = fiber.pneuron.gLeak           # S/m2
    taum = fiber.pneuron.Cm0 / gm      # s
    tauax = fiber.pneuron.Cm0 / gax    # s
    insets[k] = (taum, tauax)
fig = plt.figure(constrained_layout=True, figsize=(9, 3.5))
graph_cbar_width_ratio = 3
gs = fig.add_gridspec(2, 5 * graph_cbar_width_ratio + 1)
subplots = {
    'a': gs[:, :2 * graph_cbar_width_ratio],
    'b': gs[0, 2 * graph_cbar_width_ratio:3 * graph_cbar_width_ratio],
    'c': gs[1, 2 * graph_cbar_width_ratio:3 * graph_cbar_width_ratio],
    'd': gs[0, 3 * graph_cbar_width_ratio:4 * graph_cbar_width_ratio],
    'e': gs[1, 3 * graph_cbar_width_ratio:4 * graph_cbar_width_ratio],
    'f': gs[0, 4 * graph_cbar_width_ratio:5 * graph_cbar_width_ratio],
    'g': gs[1, 4 * graph_cbar_width_ratio:5 * graph_cbar_width_ratio],
    'h': gs[:, 5 * graph_cbar_width_ratio],
}
axes = {k: fig.add_subplot(v) for k, v in subplots.items()}
cbarax = axes['h']
for (k, (f, gammas)), ax in zip(fgamma_pairs.items(), axes.values()):
    bm = SonicBenchmark(pneuron, default_ga, f, gammas, passive=passive)
    divmap = ModelDivergenceMap(root, bm, div_criterion, tau_range, tau_range)
    divmap.run()
    divmap.render(
        levels=div_levels, insets=insets, ax=ax, cbarax=cbarax, fs=10, title=k,
        minimal=k != 'default', interactive=True)


# Cell-type-specific gamma divmaps
# gamma_range = np.linspace(0, 1.6, nperax)
# for k, fiber in fibers.items():
#     ga = fiber.ga_node_to_node * 1e4  # S/m2
#     bm = SonicBenchmark(fiber.pneuron, ga, Fdrive, default_gammas, passive=False)
#     divmap = GammaDivergenceMap(root, bm, div_criterion, gamma_range, gamma_range, tstop=tstops[k])
#     divmap.run()
#     divmap.render(levels=div_levels, fs=10, title=k, interactive=True)

plt.show()

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-09-24 13:42:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-10-05 11:47:42

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.utils import logger, expandRange, bounds, si_format
from PySONIC.core import BilayerSonophore
from PySONIC.neurons import passiveNeuron

from ExSONIC.core import SennFiber, UnmyelinatedFiber
from PySONIC.multicomp import GammaMap, SonicBenchmark, ModelDivergenceMap, GammaDivergenceMap
# from ExSONIC.core.benchmark import *
from ExSONIC.constants import *
from root import datadir, figdir

logger.setLevel(logging.INFO)

# Parameters
root = datadir
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
div_levels = [1.0]  # mV
tau_range = np.logspace(*np.log10(tau_bounds), nperax)  # s
interactive = True
fontsize = 10

# Fibers
fibers = {
    'myelinated': SennFiber(10e-6, 11),
    'unmyelinated': UnmyelinatedFiber(0.8e-6)
}

# Figure
fig = plt.figure(constrained_layout=True, figsize=(10, 7))
gs = fig.add_gridspec(8, 10)
subplots = {
    'a': gs[:4, 5:],
    'g1': gs[:2, :2],
    'g2': gs[:2, 2:4],
    'gcbar': gs[:2, 4],
    'b': gs[2, :5],
    'c': gs[3, :5],
    'd': gs[4:6, :2],
    'e': gs[6:8, :2],
    '_1': gs[4:, 2],
    'f': gs[4:6, 3:5],
    'g': gs[6:8, 3:5],
    '_2': gs[4:, 5],
    'h': gs[4:6, 6:8],
    'i': gs[6:8, 6:8],
    '_3': gs[4:, 8],
    'j': gs[4:, 9],
}
axes = {k: fig.add_subplot(v) for k, v in subplots.items()}

# Cell-type-specific gamma maps
rheobase_gammas = {
    'myelinated': 0.32,
    'unmyelinated': 0.30
}
cbarax = axes['gcbar']
gammabounds = (0, 1.7)
gammamaps_axes = [axes['g1'], axes['g2']]
for ax, (k, fiber) in zip(gammamaps_axes, fibers.items()):
    bls = BilayerSonophore(a, fiber.pneuron.Cm0, fiber.pneuron.Qm0)
    Qm = fiber.pneuron.Qm0
    gamma_map = GammaMap(root, bls, Qm, freqs, amps)
    gamma_map.run()
    # gamma_map.toPickle(root)
    levels = sorted(np.hstack(([rheobase_gammas[k]], default_gammas)))
    gamma_map.render(zbounds=gammabounds, levels=levels, ax=ax, cbarax=cbarax, fs=fontsize, title=k)
    ax.get_xaxis().get_minor_formatter().labelOnlyBase = False
    ax.get_xaxis().get_major_formatter().labelOnlyBase = True
ax.set_ylabel('')
cbarax.set_aspect(15)
cbarax.set_ylabel('gamma', labelpad=-10)
gamma_map.cbar.set_ticks(gammabounds)

# Passive model
Cm0 = 1e-2  # F/m2
Vm0 = -70.0  # mV
pneuron = passiveNeuron(Cm0, default_gm, Vm0)

# Default passive divergence map
ax = axes['a']
cbarax = axes['j']
fiber_insets = {}
for k, fiber in fibers.items():
    gax = fiber.ga_node_to_node * 1e4  # S/m2
    gm = fiber.pneuron.gLeak           # S/m2
    taum = fiber.pneuron.Cm0 / gm      # s
    tauax = fiber.pneuron.Cm0 / gax    # s
    fiber_insets[k] = (taum, tauax)

div_insets = {
    '(i)': (1.13e-6, 42.81e-6),
    '(ii)': (42.81e-6, 615.85e-9)
}
bm = SonicBenchmark(pneuron, default_ga, Fdrive, default_gammas, passive=passive)
divmap = ModelDivergenceMap(root, bm, div_criterion, tau_range, tau_range)
divmap.run()
divmap.render(
    ax=ax, cbarax=cbarax, fs=fontsize, title=f'f = {bm.fstr}, g = {bm.gammastr}',
    levels=div_levels, insets={**fiber_insets, **div_insets}, interactive=interactive)

# Characteristic traces
traces_axes = [axes['b'], axes['c']]
ylims = (-85, -40)
for ax, (k, taus) in zip(traces_axes, div_insets.items()):
    bm.setTimeConstants(*taus)
    t, sol = bm.simAllMethods(bm.passive_tstop)
    bm.logDivergences(t, sol)
    bm.plotQnorm(t, sol, ax=ax, notitle=True)
    ax.set_ylim(*ylims)
    ax.set_yticks(ylims)
    ax.set_yticklabels([])
    # ax.set_title(f'taum = {si_format(taus[0], 1)}s, tauax = {si_format(taus[1], 1)}s',
    #              fontsize=fontsize)
    # ax.set_title(k, fontsize=fontsize)
    # ax.set_ylabel('mV', labelpad=-10, fontsize=fontsize)
    ax.set_ylabel('')
ax = traces_axes[0]
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_xlabel('')
ax = traces_axes[-1]
s = ax.spines['bottom']
trep = np.power(10, np.floor(np.log10(np.ptp(ax.get_xlim()) / S_TO_MS)))
ax.set_xticks([])
s = ax.spines['bottom']
s.set_bounds(0, trep * S_TO_MS)
s.set_position(('outward', 3))
s.set_linewidth(3.0)
ax.set_xlabel(f'{si_format(trep)}s', fontsize=fontsize)

# Other passive divergence maps
fgamma_pairs = {}
for k, s in zip(['low gamma', 'high gamma'], [-1, 1]):  # vary gamma amplitudes
    fgamma_pairs[k] = (Fdrive, default_gammas + s * 0.2)
for k, x in zip(['low gradient', 'high gradient'], [0.5, 2]):  # vary gamma gradient
    fgamma_pairs[k] = (Fdrive, np.array(expandRange(*default_gammas[::-1], exp_factor=x))[::-1])
for k, f in zip(['low f', 'high f'], bounds(freqs)):  # vary Fdrive
    fgamma_pairs[k] = (f, default_gammas)
submaps_axes = [axes[k] for k in ['d', 'e', 'f', 'g', 'h', 'i']]
for (k, (f, gammas)), ax in zip(fgamma_pairs.items(), submaps_axes):
    bm = SonicBenchmark(pneuron, default_ga, f, gammas, passive=passive)
    divmap = ModelDivergenceMap(root, bm, div_criterion, tau_range, tau_range)
    divmap.run()
    title = f'f = {bm.fstr}' if f != Fdrive else f'g = {bm.gammastr}'
    divmap.render(
        ax=ax, cbarax=cbarax, fs=fontsize, title=title, insets=fiber_insets,
        minimal=True, interactive=interactive, levels=div_levels)

# Colorbar aspect
cbarax.set_aspect(15)

fig.savefig(os.path.join(figdir, 'benchmark_raw.pdf'), transparent=True)

# Cell-type-specific benchmarks and gamma divergence maps
gamma_range = np.linspace(0, 1.6, nperax)
tstops = {
    'myelinated': 1e-3,  # s
    'unmyelinated': 10e-3  # s
}

fig = plt.figure(constrained_layout=True, figsize=(10, 7))
gs = fig.add_gridspec(6, 2)
subplots = {
    'a': gs[:3, 0],
    'b': gs[0, 1],
    'c': gs[1, 1],
    'd': gs[2, 1],
    'e': gs[3:6, 0],
    'f': gs[3, 1],
    'g': gs[4, 1],
    'h': gs[5, 1],
}
axes = {k: fig.add_subplot(v) for k, v in subplots.items()}
# cbarax = None
mainkeys = ['a', 'e']
subkeys = list(set(list(axes.keys())) - set(mainkeys))
mainaxes = [axes[k] for k in mainkeys]
subaxes = [axes[k] for k in subkeys]

full_insets = {
    'myelinated': {
        '(i)': (0.59, 0.67),
        '(ii)': (0.84, 0.93),
        '(iii)': (0.67, 1.52)
    },
    'unmyelinated': {
        '(i)': (1.01, 0.),
        '(ii)': (1.18, 0.51),
        '(iii)': (1.52, 0.08)
    }
}

isubax = 0
for ax, (k, fiber) in zip(mainaxes, fibers.items()):
    ga = fiber.ga_node_to_node * 1e4  # S/m2
    bm = SonicBenchmark(fiber.pneuron, ga, Fdrive, default_gammas, passive=False)
    title = f'{k} - f = {bm.fstr}'
    for ik, gammapair in full_insets[k].items():
        subax = subaxes[isubax]
        bm.gammas = gammapair
        t, sol = bm.simAndSave(tstops[k], outdir=datadir)
        bm.plotQnorm(t, sol, ax=subax)
        subax.set_title(ik)
        subax.set_ylabel('')
        # bm.logDivergences(t, sol)
        isubax += 1
    logger.info(title)
    divmap = GammaDivergenceMap(
        root, bm, div_criterion, gamma_range, gamma_range, tstop=tstops[k])
    divmap.run()
    divmap.render(fs=fontsize, title=title, ax=ax, cbarax=cbarax,
                  levels=div_levels, insets=full_insets[k], interactive=interactive)

# cbarax.set_aspect(15)
fig.savefig(os.path.join(figdir, 'benchmark_raw2.pdf'), transparent=True)


plt.show()

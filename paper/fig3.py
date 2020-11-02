# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-09-14 11:50:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-11-02 12:20:58

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.utils import logger, si_format, bounds, padleft
from PySONIC.plt import *
from ExSONIC.core import SennFiber, UnmyelinatedFiber
from ExSONIC.core.sources import *
from ExSONIC.constants import *
from ExSONIC.plt import setAxis
from PySONIC.core import PulsedProtocol

from root import figdir

fontsize = 10

logger.setLevel(logging.INFO)

# US source
Fdrive = 500e3  # Hz
Adrive = 120e3  # Pa
w = 5e-3  # FWHM (m)
sigma = GaussianSource.from_FWHM(w)  # m
source = GaussianAcousticSource(0., sigma, Fdrive, Adrive)

# Fiber objects
a = 32e-9       # m
fs = 0.8        # (-)
fibers = {
    'myelinated': SennFiber(10e-6, 21, a=a, fs=fs),
    'unmyelinated': UnmyelinatedFiber(0.8e-6, fiberL=20e-3, a=a, fs=fs)
}

# Protocols
pps = {
    'myelinated': PulsedProtocol(1e-3, 0.5e-3),
    'unmyelinated': PulsedProtocol(10e-3, 20e-3)
}

# Plot parameters
xfactor = S_TO_MS
cmap = plt.get_cmap('sym_viridis_r')
ntraces_max = 30
varkeys = ['Cm', 'Vm', 'Qm']
precision = {'Cm': 1, 'Vm': 0, 'Qm': 0}
signed = {'Cm': False, 'Vm': True, 'Qm': True}
currents_precision = {'myelinated': 0, 'unmyelinated': 1}
dQnorm_thr = 5.  # mV
ypad = -10

nrows = len(varkeys) + 2
ncols = len(fibers)

nrows = nrows * 2 + 1
indexes = [0, 2, 4, 7, 9]
fig = plt.figure(figsize=(6, 8), constrained_layout=True)
gs = fig.add_gridspec(nrows, ncols)
col_index = {'myelinated': 0, 'unmyelinated': 1}
subplots = {k: [gs[i:i + 2, j] for i in indexes] for k, j in col_index.items()}
axdict = {k: [fig.add_subplot(sp) for sp in v] for k, v in subplots.items()}
cbar_ax = fig.add_subplot(gs[6, :])
iax_xscale = len(varkeys) - 1

for k, axes in axdict.items():
    for ax in axes[:-1]:
        for sk in ['top', 'right', 'bottom']:
            ax.spines[sk].set_visible(False)
        ax.set_xlim(-0.05 * pps[k].tstop * xfactor, pps[k].tstop * xfactor)
        ax.set_xticks([])
        ax.axvspan(0, pps[k].tstim * xfactor, facecolor='silver', alpha=0.5)
    axes[iax_xscale].set_xlabel(f'{si_format(pps[k].tstim)}s', fontsize=fontsize)
    s = axes[iax_xscale].spines['bottom']
    s.set_visible(True)
    s.set_bounds(0, pps[k].tstim * xfactor)
    s.set_position(('outward', 3))
    s.set_linewidth(3.0)

for i, (k, fiber) in enumerate(fibers.items()):
    axes = axdict[k]

    logger.info(f'{fiber}: length = {fiber.length * 1e3:.1f} mm')
    data, meta = fiber.simulate(source, pps[k])
    df = data[fiber.central_ID]
    currents = fiber.getCurrentsDict(df)
    tthr = timeThreshold(df['t'].values, df['Qm'].values / fiber.pneuron.Cm0 * V_TO_MV, dQnorm_thr)
    buildup_charges_norm = fiber.getBuildupContributions(df, tthr)

    indexes = np.linspace(0, fiber.nnodes - 1, min(fiber.nnodes, ntraces_max)).astype(int)
    ids = [f'node{i}' for i in indexes]
    norm, sm = setNormalizer(cmap, bounds(indexes))
    colors = [sm.to_rgba(i) for i in indexes]
    axes[0].set_title(f'{k} axon', fontsize=fontsize)
    for ax, varkey in zip(axes, ['Cm', 'Vm', 'Qm']):
        varinfo = fiber.getPltVars()[varkey]
        ylabel = f'{varinfo["label"]} ({varinfo["unit"]})'
        ax.set_ylabel(ylabel.replace('_', '').replace('^', ''), fontsize=fontsize, labelpad=ypad)
        for sec_id, color in zip(ids, colors):
            df = data[sec_id]
            t, y = df['t'].values * xfactor, df[varkey].values * varinfo.get('factor', 1.0)
            ax.plot(np.insert(t, 0, -0.02 * pps[k].tstop * xfactor), padleft(y), c=color)
        setAxis(ax, precision[varkey], signed[varkey])

    ax = axes[-2]
    ax.set_ylabel('I (A/m2)', fontsize=fontsize, labelpad=ypad)
    yfactor = MA_TO_A
    t = np.insert(df['t'].values, 0, -0.02 * pps[k].tstop) * xfactor
    inet = currents.pop('Net')
    for current_key, y in currents.items():
        ax.plot(t, padleft(y) * yfactor, label=current_key, linewidth=2)
    ax.plot(t, padleft(inet) * yfactor, '--', label='Net', c='k', linewidth=2)
    # ax.legend(frameon=False, fontsize=fontsize)
    setAxis(ax, currents_precision[k], True)
    ax.axvline(tthr * xfactor, c='k', linestyle=':')
    ax.text(0.5, 0.0, f'Dt+5mV = {si_format(tthr, 1)}s',
            fontsize=fontsize, transform=ax.transAxes)

    ax = axes[-1]
    for sk in ['top', 'right', 'bottom']:
        ax.spines[sk].set_visible(False)
    ax.tick_params(length=0, axis='x')
    ax.set_ylabel('DQm/Cm0 (mV)', labelpad=ypad)
    if 'P' not in buildup_charges_norm:
        buildup_charges_norm['_'] = 0.
    buildup_charges_norm_abs = {k: np.abs(v) for k, v in buildup_charges_norm.items()}
    colors = plt.get_cmap('tab10').colors
    xlabels = list(buildup_charges_norm_abs.keys())
    y = np.array(list(buildup_charges_norm_abs.values()))

    # Sort by descending absolute value
    isorted = np.argsort(y)[::-1]
    xlabels = [xlabels[i] for i in isorted]
    colors = [colors[i] for i in isorted]
    y = [y[i] for i in isorted]

    x = np.arange(len(xlabels))
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_yscale('log')
    ax.set_ylim(1e-7, 1e1)
    ax.set_yticks([1e-7, 1e1])
    ax.bar(x, y, color=colors)
    for ax in axes:
        for item in ax.get_yticklabels():
            item.set_fontsize(fontsize)

ax = cbar_ax
cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
imin, imax = ax.get_ylim()
lims = [imin, (imin + imax) / 2, imax]
ax.tick_params(length=0)
cbar.set_ticks(lims)
cbar.set_ticklabels(['proximal', 'central', 'distal'])
for item in ax.get_yticklabels():
    item.set_fontsize(fontsize)

fig.savefig(os.path.join(figdir, 'fig3_raw.pdf'), transparent=True)

plt.show()

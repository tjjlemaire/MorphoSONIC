# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-08-24 19:34:35
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-11-02 12:24:15

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.utils import logger, si_format
from PySONIC.core import AcousticDrive
from ExSONIC.core import Node, SennFiber, UnmyelinatedFiber, StrengthDurationBatch
from ExSONIC.core.sources import *
from ExSONIC.utils import rheobase, chronaxie

from root import datadir, figdir

logger.setLevel(logging.INFO)

fontsize = 10


def getAcousticDrive(Fdrive):
    drive = AcousticDrive(Fdrive)
    drive.key = 'A'
    return drive


def emptyClone(d):
    return {k: {} for k in d.keys()}


def setSharedLims(axes, xlims, ylims):
    for ax in axes:
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
    for ax in axes[1:]:
        ax.get_shared_x_axes().join(ax, axes[0])
        ax.get_shared_y_axes().join(ax, axes[0])


def plotTypicalSDs(ax, thrs_dict, xfactor=1, yfactor=1, colors=None, plt_markers=True):
    if colors is None:
        colors = plt.get_cmap('tab10').colors
    ymax = ax.get_ylim()[1]
    y_corrected = {}
    for i, (k, thrs) in enumerate(thrs_dict.items()):
        c = colors[i]
        y = np.abs(thrs)
        y_corrected[k] = y.copy() * yfactor
        ax.plot(durations * xfactor, y * yfactor, label=k, c=c)
        if any(np.isnan(y)):
            ilastnan = np.where(np.isnan(y))[0][-1]
            ax.plot(
                durations[ilastnan:ilastnan + 2] * xfactor,
                [ymax, y[ilastnan + 1] * yfactor],
                '--', color=c)
            y_corrected[k][ilastnan] = ymax
        if plt_markers:
            tch, yrh = chronaxie(durations, thrs), rheobase(thrs)
            ych = 2 * yrh
            xmin, ymin = ax.get_xlim()[0], ax.get_ylim()[0]
            ax.plot([xmin, tch * xfactor], [ych * yfactor] * 2, ':', c=c)
            ax.plot([tch * xfactor] * 2, [ymin, ych * yfactor], ':', c=c)
            ax.scatter(tch * xfactor, ych * yfactor, c=[c, ], s=20, zorder=2.5)

            axis_to_data = ax.transAxes + ax.transData.inverted()
            data_to_axis = axis_to_data.inverted()
            ax_tch, ax_ych = data_to_axis.transform((tch * xfactor, ych * yfactor))
            _, ax_yrh = data_to_axis.transform((tch * xfactor, yrh * yfactor))

            ax.text(ax_tch + 0.02, 0.02, f'tch = {si_format(tch, 0)}s', c=c, fontsize=fontsize,
                    transform=ax.transAxes)
            ysymbol = 'Vrh' if yfactor == 1 else 'Arh'
            yunit = 'mV' if yfactor == 1 else 'Pa'
            ax.text(1.0, ax_yrh + 0.02, f'{ysymbol} = {si_format(yrh)}{yunit}', c=c,
                    fontsize=fontsize, transform=ax.transAxes, horizontalalignment='right')
            ax.text(ax_tch / 2, ax_ych + 0.02, f'2{ysymbol}', c=c, fontsize=fontsize,
                    transform=ax.transAxes, horizontalalignment='center')


def plotSDandMarkers(ax, durations, thrs_dict, xfactor=1, yfactor=1, colors=None):
    if colors is None:
        colors = plt.get_cmap('tab10').colors
    for (key, sub_thrs_dict), c in zip(thrs_dict.items(), colors):
        rhs, tchs = [], []
        for k, thrs in sub_thrs_dict.items():
            rh, tch = rheobase(thrs), chronaxie(durations, thrs)
            rhs.append(rh)
            tchs.append(tch)
            if not np.isnan(tch):
                ax.plot(durations * xfactor, np.abs(thrs) * yfactor, c='silver')
        rhs, tchs = np.array(rhs), np.array(tchs)
        if any(np.isnan(tchs)):
            inotnan = np.where(~np.isnan(tchs))[0]
            tchs, rhs = tchs[inotnan], rhs[inotnan]
        ax.plot(tchs * xfactor, 2 * rhs * yfactor, c=c, linewidth=2, zorder=2)
        scatter_kwargs = {'c': [c, ], 's': 20, 'zorder': 2.5}
        ax.scatter(tchs[0] * xfactor, 2 * rhs[0] * yfactor, **scatter_kwargs)

        # dx = tchs[-1] - tchs[-2]
        # dy = 2 * (rhs[-1] - rhs[-2])
        # dx, dy = ax.transData.transform_point((dx, dy))
        # print(dx, dy)
        # phi = np.arctan2(dx, dy)
        # print(phi)
        ax.scatter(tchs[-1] * xfactor, 2 * rhs[-1] * yfactor, marker=(3, 0, 0), **scatter_kwargs)


# Durations and offset
durations = np.logspace(-5, 0, 20)  # s
toffset = 10e-3  # s

# Default gaussian width
w = 5e-3  # FWHM (m)
sigma = GaussianSource.from_FWHM(w)  # m

# Default US parameters
Fdrive = 500e3  # Hz
a = 32e-9       # m
fs = 0.8        # (-)

# Variation ranges
myel_diams = np.array([5, 10, 20]) * 1e-6        # m
unmyel_diams = np.array([0.2, 0.8, 1.5]) * 1e-6  # m
widths = np.linspace(1, 10, 5) * 1e-3  # m
freqs = np.logspace(np.log10(20), np.log10(4000), 9) * 1e3  # Hz
radii = np.logspace(np.log10(a / 2), np.log10(2 * a), 9)  # m
coverages = np.linspace(0.0, 1.0, 21)[1:]  # (-)

# Plot parameters
colors = list(plt.get_cmap('tab20').colors)
xfactor = 1e3

# Default fibers
fibers = {
    'unmyelinated': UnmyelinatedFiber(0.8e-6, fiberL=5e-3, a=a, fs=fs),
    'myelinated': SennFiber(10e-6, 21, a=a, fs=fs)
}

# Corresponding nodes
nodes = {fiber.pneuron.name: Node(fiber.pneuron, a=a, fs=fs) for fiber in fibers.values()}

# EL: default SD curves
source = GaussianVoltageSource(0., sigma, mode='cathode')
EL_Athrs = {}
for k, fiber in fibers.items():
    logger.info(f'fiber length = {fiber.length * 1e3:.2f} mm, source FWHM = {w * 1e3:.2f} mm')
    sd_batch = StrengthDurationBatch('Vext (mV)', source, fiber, durations, toffset, root=datadir)
    EL_Athrs[k] = sd_batch.run()

EL_thrs_variations = {}

# EL: impact of fiber diameter
source = GaussianVoltageSource(0., sigma, mode='cathode')
EL_thrs_variations['fiber diameter'] = emptyClone(fibers)
for fiberD in myel_diams:
    k = f'fiberD = {si_format(fiberD, 1)}m'
    fiber = SennFiber(fiberD, 21, a=a, fs=fs)
    logger.info(f'fiber length = {fiber.length * 1e3:.2f} mm, source FWHM = {w * 1e3:.2f} mm')
    sd_batch = StrengthDurationBatch('Vext (mV)', source, fiber, durations, toffset, root=datadir)
    EL_thrs_variations['fiber diameter']['myelinated'][k] = sd_batch.run()
for fiberD in unmyel_diams:
    k = f'fiberD = {si_format(fiberD, 1)}m'
    fiber = UnmyelinatedFiber(fiberD, fiberL=5e-3, a=a, fs=fs)
    logger.info(f'fiber length = {fiber.length * 1e3:.2f} mm, source FWHM = {w * 1e3:.2f} mm')
    sd_batch = StrengthDurationBatch('Vext (mV)', source, fiber, durations, toffset * 2, root=datadir)
    EL_thrs_variations['fiber diameter']['unmyelinated'][k] = sd_batch.run()

# EL: impact of beam width
EL_thrs_variations['beam width'] = emptyClone(fibers)
for key, fiber in fibers.items():
    for w in widths:
        k = f'w = {si_format(w, 1)}m'
        source = GaussianVoltageSource(0., GaussianSource.from_FWHM(w), mode='cathode')
        logger.info(f'fiber length = {fiber.length * 1e3:.2f} mm, source FWHM = {w * 1e3:.2f} mm')
        sd_batch = StrengthDurationBatch('Vext (mV)', source, fiber, durations, toffset, root=datadir)
        EL_thrs_variations['beam width'][key][k] = sd_batch.run()

# US: default SD curves
source = GaussianAcousticSource(0., sigma, Fdrive)
US_Athrs = {}
for k, fiber in fibers.items():
    logger.info(f'fiber length = {fiber.length * 1e3:.2f} mm, source FWHM = {w * 1e3:.2f} mm')
    sd_batch = StrengthDurationBatch('A (Pa)', source, fiber, durations, toffset, root=datadir)
    US_Athrs[k] = sd_batch.run()

# US: point-neuron SD curves
drive = getAcousticDrive(Fdrive)
US_node_Athrs = {}
for k, node in nodes.items():
    sd_batch = StrengthDurationBatch('A (Pa)', drive, node, durations, toffset, root=datadir)
    US_node_Athrs[node.pneuron.name] = sd_batch.run()

US_thrs_variations = {}

# US: impact of fiber diameter
source = GaussianAcousticSource(0., sigma, Fdrive)
US_thrs_variations['fiber diameter'] = emptyClone(fibers)
for fiberD in myel_diams:
    k = f'fiberD = {si_format(fiberD)}m'
    fiber = SennFiber(fiberD, 21, a=a, fs=fs)
    logger.info(f'fiber length = {fiber.length * 1e3:.2f} mm, source FWHM = {w * 1e3:.2f} mm')
    sd_batch = StrengthDurationBatch('A (Pa)', source, fiber, durations, toffset, root=datadir)
    US_thrs_variations['fiber diameter']['myelinated'][k] = sd_batch.run()
for fiberD in unmyel_diams:
    k = f'fiberD = {si_format(fiberD)}m'
    fiber = UnmyelinatedFiber(fiberD, fiberL=5e-3, a=a, fs=fs)
    logger.info(f'fiber length = {fiber.length * 1e3:.2f} mm, source FWHM = {w * 1e3:.2f} mm')
    sd_batch = StrengthDurationBatch('A (Pa)', source, fiber, durations, toffset * 2, root=datadir)
    US_thrs_variations['fiber diameter']['unmyelinated'][k] = sd_batch.run()

# US: impact of beam width
widths = np.linspace(1, 10, 5) * 1e-3  # m
US_thrs_variations['beam width'] = emptyClone(fibers)
for key, fiber in fibers.items():
    for w in widths:
        k = f'w = {si_format(w)}m'
        source = GaussianAcousticSource(0., GaussianSource.from_FWHM(w), Fdrive)
        logger.info(f'fiber length = {fiber.length * 1e3:.2f} mm, source FWHM = {w * 1e3:.2f} mm')
        sd_batch = StrengthDurationBatch('A (Pa)', source, fiber, durations, toffset, root=datadir)
        US_thrs_variations['beam width'][key][k] = sd_batch.run()

# US: impact of frequency
US_thrs_variations['US frequency'] = emptyClone(nodes)
for key, refnode in nodes.items():
    node = Node(refnode.pneuron, a=a, fs=1)
    for x in freqs:
        k = f'f = {si_format(x, 1)}Hz'
        sd_batch = StrengthDurationBatch(
            'A (Pa)', getAcousticDrive(x), node, durations, toffset, root=datadir)
        US_thrs_variations['US frequency'][key][k] = sd_batch.run()

# US: impact of sonophore radius
drive = getAcousticDrive(Fdrive)
US_thrs_variations['sonophore radius'] = emptyClone(nodes)
for key, refnode in nodes.items():
    for x in radii:
        k = f'a = {si_format(x, 1)}m'
        node = Node(refnode.pneuron, a=x, fs=1)
        sd_batch = StrengthDurationBatch('A (Pa)', drive, node, durations, toffset, root=datadir)
        US_thrs_variations['sonophore radius'][key][k] = sd_batch.run()

# US: impact of sonophore coverage
drive = getAcousticDrive(Fdrive)
US_thrs_variations['sonophore coverage'] = emptyClone(nodes)
for key, refnode in nodes.items():
    for x in coverages:
        k = f'fs = {x * 1e2:.1f}%'
        node = Node(refnode.pneuron, a=a, fs=x)
        sd_batch = StrengthDurationBatch('A (Pa)', drive, node, durations, toffset, root=datadir)
        US_thrs_variations['sonophore coverage'][key][k] = sd_batch.run()


# Figure
fig = plt.figure(constrained_layout=True, figsize=(5, 8))
fig.canvas.set_window_title('SD_curves_new')
gs = fig.add_gridspec(5, 3)
subplots = {
    'a': gs[:2, :2],
    'b': gs[0, 2],
    'c': gs[1, 2],
    'd': gs[2:4, :2],
    'e': gs[2, 2],
    'f': gs[3, 2],
    'g': gs[4, 0],
    'h': gs[4, 1],
    'i': gs[4, 2],
}
axes = {k: fig.add_subplot(v) for k, v in subplots.items()}
for ax in axes.values():
    for k in ['top', 'right']:
        ax.spines[k].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
EL_keys = ['a', 'b', 'c']
US_keys = ['d', 'e', 'f', 'g', 'h', 'i']
main_keys = [k[0] for k in [EL_keys, US_keys]]
other_keys = list(set(axes.keys()) - set(main_keys))

# EL panel
kwargs = {'xfactor': xfactor, 'yfactor': 1, 'colors': colors[::2]}
setSharedLims([axes[k] for k in EL_keys], xlims=(1e-2, 1e3), ylims=(1e1, 1e5))
axes['a'].set_title('EL SD curves', fontsize=fontsize)
axes['a'].set_ylabel(f'threshold peak voltage (mV)', fontsize=fontsize)
plotTypicalSDs(axes['a'], EL_Athrs, **kwargs)
axes['a'].legend(frameon=False, fontsize=fontsize)
for key, (label, thrs_dict) in zip(EL_keys[1:], EL_thrs_variations.items()):
    axes[key].set_title(label, fontsize=fontsize)
    plotSDandMarkers(axes[key], durations, thrs_dict, **kwargs)

# US panel
kwargs = {'xfactor': xfactor, 'yfactor': 1e-3, 'colors': colors[::2]}
kwargs2 = kwargs.copy()
kwargs2['colors'] = colors[1::2]
setSharedLims([axes[k] for k in US_keys], xlims=(1e-2, 1e3), ylims=(1e1, 1e3))
axes['d'].set_title('US SD curves', fontsize=fontsize)
axes['d'].set_ylabel(f'threshold peak pressure (kPa)', fontsize=fontsize)
plotTypicalSDs(axes['d'], US_Athrs, **kwargs)
plotTypicalSDs(axes['d'], US_node_Athrs, plt_markers=False, **kwargs2)
axes['d'].legend(frameon=False, fontsize=fontsize)
for key, (label, thrs_dict) in zip(US_keys[1:], US_thrs_variations.items()):
    axes[key].set_title(label, fontsize=fontsize)
    plotSDandMarkers(axes[key], durations, thrs_dict, **kwargs)

# Post-processing layout
for ax in axes.values():
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fontsize)
for k in main_keys:
    axes[k].set_xlabel('pulse duration (ms)', fontsize=fontsize)
for k in other_keys:
    ax = axes[k]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()

# Save as pdf
fig.savefig(os.path.join(figdir, 'fig4_raw.pdf'), transparent=True)

plt.show()

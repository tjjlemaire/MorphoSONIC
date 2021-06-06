# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-08-22 14:14:17
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-04 15:39:56

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest

from PySONIC.utils import logger, si_format
from PySONIC.plt import setNormalizer
from ExSONIC.core import PlanarDiskTransducerSource, ExtracellularCurrent
from ExSONIC.plt import setAxis

from root import datadir, figdir


logger.setLevel(logging.INFO)

xc = (0., 0., 0.)  # m
radii = [1e-3, 1e-2]  # m
freqs = [500e3, 2e6]  # Hz
nperslice = 400
radii_dense = np.linspace(1e-3, 1e-2, 5)
xfocal = np.linspace(-20e-3, 20e-3, nperslice)
fontsize = 10


def computeTransverseFocalDistribution(r, Fdrive, x):
    code = f'{r * 1e3:.0f}mm_{Fdrive * 1e-3:.0f}kHz'
    fname = f'focalslice_{code}.csv'
    fpath = os.path.join(datadir, fname)
    if os.path.isfile(fpath):
        logger.info(f'Loading Pac focal distribution from file {fname}')
        amps = np.loadtxt(fpath)
    else:
        source = PlanarDiskTransducerSource(xc, Fdrive, u=1, r=r)
        logger.info(f'Computing Pac focal distribution for {source}')
        amps = np.squeeze(source.DPSM2d(x, [source.getFocalDistance()]))  # Pa
        logger.info(f'Saving Pac field in file {fname}')
        np.savetxt(fpath, amps)
    isGaussian(amps, 0.05)
    return amps


def isGaussian(x, alpha):
    _, p = normaltest(x)
    is_normal = p < alpha
    s = 'looks' if is_normal else 'does not look'
    logger.info(f'Distribution {s} gaussian (p = {p:.2e})')
    return is_normal


def getFWHM(x, y):
    xneg = x <= 0.
    return 2 * np.abs(np.interp(0.5 * y.max(), y[xneg], x[xneg]))


def addFiberAxisArrow(ax):
    arrowwidth = 0.7
    xarrow = [0.5 - arrowwidth / 2, 0.5 + arrowwidth / 2]
    yarrow = 0.6
    ax.text(np.mean(xarrow), yarrow + 0.1, 'fiber axis', fontsize=fontsize,
            transform=ax.transAxes, ha='center', color='w')
    ax.annotate('', xy=(xarrow[0], yarrow), xytext=(xarrow[1], yarrow),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(edgecolor='w', facecolor='w', arrowstyle='<|-|>'))


# Figure backbone
fig = plt.figure(constrained_layout=True, figsize=(6, 8))
fig.canvas.set_window_title('fields')
gs = fig.add_gridspec(6, 4)
subplots = {
    'a': gs[:2, 0],
    'b': gs[:2, 1],
    'c': gs[2:4, 0],
    'd': gs[2:4, 1],
    'e': gs[:2, 2:],
    'f': gs[2:4, 2:],
    'g': gs[4:, 0],
    'h': gs[4, 1],
    'i': gs[5, 1],
    'j': gs[4:, 2:],
}
axes = {k: fig.add_subplot(v) for k, v in subplots.items()}


# US 2D fields
axkeys = ['a', 'b', 'c', 'd']
iax = 0
for Fdrive in freqs:
    for r in radii:
        title = f'r = {r * 1e3:.0f} mm, f = {si_format(Fdrive)}Hz'

        # Create transducer object
        source = PlanarDiskTransducerSource(xc, Fdrive, u=1, r=r)

        # Determine evaluation plane
        dfocal = source.getFocalDistance()

        x = np.linspace(-5 * r, 5 * r, nperslice)
        z = np.linspace(0.1e-3, 20 * r, nperslice)

        # Compute and plot amplitude distribution over 2D field
        sourcecode = '_'.join(source.filecodes.values())
        fname = f'2Dfield_{sourcecode}.csv'
        fpath = os.path.join(datadir, fname)
        if os.path.isfile(fpath):
            logger.info(f'Loading Pac field from file {fname}')
            amps = np.loadtxt(fpath)
        else:
            logger.info(f'Computing Pac field for {source}')
            amps = source.DPSM2d(x, z)  # Pa
            logger.info(f'Saving Pac field in file {fname}')
            np.savetxt(fpath, amps)
        ax = axes[axkeys[iax]]
        ax.set_aspect(1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        sm = ax.pcolormesh(x * 1e3, z * 1e3, amps.T, cmap='viridis')
        ax.axhline(dfocal * 1e3, c='white', ls='--')
        iax += 1

for ax, Fdrive in zip([axes['a'], axes['c']], freqs):
    ax.set_ylabel(f'{si_format(Fdrive)}Hz', fontsize=fontsize)
for ax, r in zip([axes['a'], axes['b']], radii):
    ax.set_title(f'{r * 1e3:.0f} mm', fontsize=fontsize)
    ax.spines['bottom'].set_visible(False)

for ax, r in zip([axes['c'], axes['d']], radii):
    scale = 5 * r * 1e3  # mm
    s = ax.spines['bottom']
    s.set_bounds(-scale, 0)
    s.set_position(('outward', 3))
    s.set_linewidth(3.0)
    ax.set_xlabel(f'{scale:.0f} mm', fontsize=fontsize)

# Transverse pressure distributions at focal distance
ax = axes['e']
ax.set_title(f'(b) transverse pressure distributions\nat focal distance are bell-shaped',
             fontsize=fontsize)
ax.set_xlabel('Transverse distance (mm)', fontsize=fontsize)
ax.set_ylabel('relative amplitude\nat focal distance', fontsize=fontsize)
colors = plt.get_cmap('tab20').colors
icolor = 0
for r in radii:
    for Fdrive in freqs:
        color = colors[icolor]
        code = f'{r * 1e3:.0f}mm_{Fdrive * 1e-3:.0f}kHz'
        amps = computeTransverseFocalDistribution(r, Fdrive, xfocal)  # Pa
        rel_amps = amps / amps.max()  # (-)
        ax.plot(xfocal * 1e3, rel_amps, c=color, label=code)
        FWHM = getFWHM(xfocal, rel_amps)
        logger.info(f'FWHM = {FWHM * 1e3:.2f} mm')
        for i in [-0.5, 0.5]:
            ax.axvline(i * FWHM * 1e3, c=color, linestyle='--')
        icolor += 1
xlims = [xfocal.min() * 1e3, xfocal.max() * 1e3]
ax.set_xticks(xlims)
ax.set_xlim(*xlims)
ylims = [0, 1]
ax.set_yticks(ylims)
ax.set_ylim(*ylims)
ax.legend(frameon=False, fontsize=fontsize)

# FWHM at focal distance vs transducer radius
ax = axes['f']
US_FWHM_vs_radius = {}
for Fdrive in freqs:
    US_FWHM_vs_radius[f'{si_format(Fdrive)}Hz'] = np.array([
        getFWHM(xfocal, computeTransverseFocalDistribution(r, Fdrive, xfocal))
        for r in radii_dense
    ])
ax.set_title(f'(c) transducer radius regulates\nbeam width at focal distance', fontsize=fontsize)
ax.set_xlabel('transducer radius (mm)', fontsize=fontsize)
ax.set_ylabel('FWHM (mm)', fontsize=fontsize)
radii_dense = np.linspace(1e-3, 1e-2, 5)
for c, (k, FWHMs) in zip(['k', 'silver'], US_FWHM_vs_radius.items()):
    ax.plot(radii_dense * 1e3, np.array(FWHMs) * 1e3, label=k, c=c)
xlims = [radii_dense.min() * 1e3, radii_dense.max() * 1e3]
ax.set_xticks(xlims)
ax.set_xlim(*xlims)
setAxis(ax, 0, False)
ax.legend(frameon=False, fontsize=fontsize)

# Create electrode object
rho = (175, 1211)  # resistitivty tensor (Ohm.cm)
xc = (0., 0.)      # xz position (m)
source = ExtracellularCurrent(xc, I=1, mode='anode', rho=rho)

# EL: 2D voltage field
ax = axes['g']
ax.set_title('(d) medium anisotropy dictates\nextracellular voltage field', fontsize=fontsize)
dx = 10e-3
x = np.linspace(-dx / 2, dx / 2, nperslice)
zoffset = 3e-3
z = np.linspace(zoffset, zoffset + dx, nperslice)
amps = np.array([
    [source.Vext(source.I, source.vectorialDistance((xx, zz))) for zz in z]
    for xx in x])
sm = ax.pcolormesh(x * 1e3, z * 1e3, amps.T, cmap='viridis')
ax.set_aspect(1.0)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['left'].set_visible(False)
scale = np.ptp(x) / 2 * 1e3  # mm
s = ax.spines['bottom']
s.set_bounds(-scale, 0)
s.set_position(('outward', 3))
s.set_linewidth(3.0)
ax.set_xlabel(f'{scale:.0f} mm', fontsize=fontsize)

# EL: FWHM of transverse distribution vs source distance
ax = axes['j']
ax.set_title(f'(e) electrode distance regulates width\nof transverse voltage distribution',
             fontsize=fontsize)
ax.set_xlabel('source distance (mm)', fontsize=fontsize)
ax.set_ylabel('FWHM (mm)', fontsize=fontsize)
zdense = np.linspace(0.1e-3, 2e-3, 10)
FWHMs = []
for z in zdense:
    logger.info(f'Computing transverse distribution at z = {z}')
    amps = np.array([source.Vext(source.I, source.vectorialDistance((xx, z))) for xx in xfocal])
    rel_amps = amps / amps.max()  # (-)
    FWHM = getFWHM(xfocal, rel_amps)
    FWHMs.append(FWHM)
ax.plot(zdense * 1e3, np.array(FWHMs) * 1e3, c='k')
xlims = [zdense.min() * 1e3, zdense.max() * 1e3]
ax.set_xticks(xlims)
ax.set_xlim(*xlims)
setAxis(ax, 0, False)

# Colorbar
ax = axes['h']
norm, sm = setNormalizer('viridis', (0, 1), scale='lin')
cbar = fig.colorbar(sm, cax=ax)
ax.set_ylabel('relative\namplitude')
cbar.set_ticks([0, 1])

# Reference frame
ax = axes['i']
ax.set_xlabel('transverse\ndirection')
ax.set_ylabel('axial\ndirection')
ax.set_xticks([])
ax.set_yticks([])
for ax in [axes['a'], axes['g']]:
    addFiberAxisArrow(ax)

for ax in axes.values():
    for sk in ['top', 'right']:
        ax.spines[sk].set_visible(False)

fig.savefig(os.path.join(figdir, f'figS1_raw.pdf'), transparent=True)

plt.show()

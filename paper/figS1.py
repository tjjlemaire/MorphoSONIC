# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-08-22 14:14:17
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-07 13:20:48

import os
import logging
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import normaltest

from PySONIC.utils import logger, si_format, rangecode, isIterable
from PySONIC.plt import setNormalizer, XYMap
from ExSONIC.core import PlanarDiskTransducerSource, ExtracellularCurrent
from ExSONIC.plt import setAxis

from root import datadir, figdir


logger.setLevel(logging.DEBUG)

xc = (0., 0., 0.)  # m
radii = [1e-3, 1e-2]  # m
freqs = [500e3, 2e6]  # Hz
nperslice = 400
radii_dense = np.linspace(1e-3, 1e-2, 5)
xfocal = np.linspace(-20e-3, 20e-3, nperslice)
phase_insets_bounds = (-2e-3, 2e-3)  # m
is_xinset = np.all([xfocal > phase_insets_bounds[0], xfocal < phase_insets_bounds[1]], axis=0)
xinset = xfocal[is_xinset]
fontsize = 10


def getPacField(source, x, y, z):
    scode = f'r_{source.r * 1e3:.0f}mm_f_{source.f * 1e-3:.0f}kHz'
    xyz = []
    for k, v in {'x': x, 'y': y, 'z': z}.items():
        xyz.append(rangecode(v, k, 'm') if isIterable(v) else f'{k}{si_format(v, 1, space="")}m')
    xyzcode = '_'.join(xyz)
    fname = f'Pacfield_{scode}_{xyzcode}.csv'
    fpath = os.path.join(datadir, fname)
    if os.path.isfile(fpath):
        logger.info(f'Loading Pac field from file {fname}')
        Pac_field = np.loadtxt(fpath, dtype=np.complex128)
    else:
        logger.info(f'Computing Pac focal distribution for {source}')
        Pac_field = source.DPSM(x, y, z)
        logger.info(f'Saving Pac field in file {fname}')
        np.savetxt(fpath, Pac_field)
    return Pac_field


def isGaussian(x, alpha):
    _, p = normaltest(x)
    is_normal = p < alpha
    s = 'looks' if is_normal else 'does not look'
    logger.info(f'Distribution {s} gaussian (p = {p:.2e})')
    return is_normal


def getFWHM(x, y):
    xneg = x <= 0.
    return 2 * np.abs(np.interp(0.5 * y.max(), y[xneg], x[xneg]))


# Encpasulation to enable multiprocessing
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-s', '--save', default=False, action='store_true', help='Save figure')
    args = parser.parse_args()

    # Figure backbone
    fig = plt.figure(constrained_layout=True, figsize=(11, 6))
    fig.canvas.manager.set_window_title('fields')
    gs = fig.add_gridspec(8, 7)
    subplots = {
        'a1': gs[:4, 0],
        'a2': gs[:4, 1],
        'a3': gs[4:, 0],
        'a4': gs[4:, 1],
        'sources': gs[4:, 2],
        'cbar1': gs[:4, 2],
        'b1': gs[:2, 3],
        'b2': gs[:2, 4],
        'b3': gs[2:4, 3],
        'b4': gs[2:4, 4],
        'c': gs[4:, 3:5],
        'd': gs[:4, 5],
        'e': gs[4:, 5:],
        'cbar2': gs[:4, 6],
    }
    axes = {k: fig.add_subplot(v) for k, v in subplots.items()}

    # US 2D fields
    axkeys = ['a1', 'a2', 'a3', 'a4']
    iax = 0
    for r in radii:
        for Fdrive in freqs:
            # Create transducer object
            source = PlanarDiskTransducerSource(xc, Fdrive, u=1, r=r)

            # Determine evaluation plane
            x = np.linspace(-5 * r, 5 * r, nperslice)
            z = np.linspace(0.1e-3, 20 * r, nperslice)

            # Compute and plot amplitude distribution over 2D field
            Pac_field = getPacField(source, x, 0, z)
            ax = axes[axkeys[iax]]
            ax.set_aspect(1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for sk in ['left', 'bottom']:
                ax.spines[sk].set_visible(False)
            xedges, zedges = [XYMap.computeMeshEdges(xx, 'lin') for xx in [x, z]]
            sm = ax.pcolormesh(
                xedges * 1e3, zedges * 1e3, np.abs(Pac_field).T, cmap='viridis', rasterized=True)
            ax.axhline(source.getFocalDistance() * 1e3, c='white', ls='--')

            iax += 1

    for ax, Fdrive in zip([axes['a1'], axes['a2']], freqs):
        ax.set_title(f'f = {si_format(Fdrive)}Hz', fontsize=fontsize)
    for ax, Fdrive in zip([axes['a3'], axes['a4']], freqs):
        ax.set_title(f'f = {si_format(Fdrive)}Hz', fontsize=fontsize)
    for ax, r in zip([axes['a1'], axes['a3']], radii):
        ax.set_xlabel(f'r = {r * 1e3:.0f} mm', fontsize=fontsize)
        scale = 5 * r * 1e3  # mm
        s = ax.spines['left']
        s.set_visible(True)
        s.set_bounds(0, scale)
        s.set_position(('outward', 3))
        s.set_linewidth(3.0)
        ax.set_ylabel(f'{scale:.0f} mm', fontsize=fontsize)

    # Transverse surface source distribution
    ax = axes['sources']
    transducer = PlanarDiskTransducerSource(xc, Fdrive, u=1, r=1e-3)
    xs, ys = transducer.getXYSources()
    ax.plot(xs * 1e3, ys * 1e3, 'o', c='k', markersize=01.0)
    ax.add_patch(Circle((0, 0), radius=transducer.r * 1e3, fc='none', ec='k'))
    ax.set_xticks([])
    ax.set_yticks([])
    for sk in ['bottom', 'left']:
        ax.spines[sk].set_visible(False)
    ax.set_aspect(1.0)

    # Transverse pressure distributions at focal distance
    # Amplitude
    colors = plt.get_cmap('tab20').colors
    axes['b1'].set_ylabel('rel. A', fontsize=fontsize)
    axes['b3'].set_ylabel('phase (rad)', fontsize=fontsize)
    icolor = 0
    max_phase_shift = 0.
    for r in radii:
        for Fdrive in freqs:
            color = colors[icolor]
            code = f'{r * 1e3:.0f}mm {si_format(Fdrive, 0, space="")}Hz'
            source = PlanarDiskTransducerSource(xc, Fdrive, u=1, r=r)
            dfocal = source.getFocalDistance()
            # z = np.linspace(0.1e-3, 20 * r, nperslice)
            z = np.linspace(0.1e-3, 2 * dfocal, nperslice)
            izfocal = int(np.interp(dfocal, z, np.arange(z.size)))
            ix0 = int(np.interp(0., xfocal, np.arange(xfocal.size)))

            # Transverse distributions
            Pac_x = getPacField(source, xfocal, 0, dfocal)
            ax = axes['b1']
            amps_x = np.abs(Pac_x)  # Pa
            amps_x /= amps_x.max()  # (-)
            ax.plot(xfocal * 1e3, amps_x, c=color, label=code, clip_on=False)
            FWHM = getFWHM(xfocal, amps_x)
            logger.info(f'FWHM = {FWHM * 1e3:.2f} mm')
            for i in [-0.5, 0.5]:
                ax.axvline(i * FWHM * 1e3, c=color, linestyle='--', clip_on=False)
            phases_x = np.angle(Pac_x)  # rad
            y = np.unwrap(phases_x) / np.pi
            y = y[is_xinset] - y[ix0]
            max_phase_shift = max(np.nanmax(np.abs(y)), max_phase_shift)
            axes['b3'].plot(xinset * 1e3, y, c=color, clip_on=False)

            # Longitudinal distributions
            Pac_z = getPacField(source, 0, 0, z)
            ax = axes['b2']
            amps_z = np.abs(Pac_z)  # Pa
            amps_z /= amps_z.max()  # (-)
            ax.plot(z / dfocal, amps_z, c=color, clip_on=False)
            phases_z = np.angle(Pac_z)  # rad

            is_zinset = np.all(
                [z > dfocal + phase_insets_bounds[0], z < dfocal + phase_insets_bounds[1]], axis=0)
            y = np.unwrap(phases_z) / np.pi
            y = y[is_zinset] - y[izfocal]
            max_phase_shift = max(np.nanmax(np.abs(y)), max_phase_shift)
            zinset = z[is_zinset]
            axes['b4'].plot((zinset - dfocal) * 1e3, y, c=color, clip_on=False)

            icolor += 1

    for axkey in ['b1', 'b2', 'b3', 'b4']:
        ax = axes[axkey]
        ax.margins(0)
        ax.set_xticks(ax.get_xlim())
        ax.set_yticks(ax.get_ylim())
    for axkey in ['b1', 'b2']:
        axes[axkey].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    for axkey in ['b3', 'b4']:
        ax = axes[axkey]
        ax.axhline(0, ls='--', c='k', clip_on=False)
        ylims = [-np.ceil(max_phase_shift), np.ceil(max_phase_shift)]
        ax.set_ylim(*ylims)
        ax.set_yticks(ylims)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f} PI'))
    for axkey in ['b1', 'b3', 'b4']:
        axes[axkey].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axes['b2'].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f} d'))
    axes['b1'].legend(frameon=False, fontsize=fontsize)

    # FWHM at focal distance vs transducer radius
    ax = axes['c']
    US_FWHM_vs_radius = {}
    for Fdrive in freqs:
        FWHMs = []
        for r in radii_dense:
            source = PlanarDiskTransducerSource(xc, Fdrive, u=1, r=r)
            Pac_x = getPacField(source, xfocal, 0, source.getFocalDistance())
            FWHMs.append(getFWHM(xfocal, np.abs(Pac_x)))
        US_FWHM_vs_radius[f'{si_format(Fdrive)}Hz'] = np.array(FWHMs)

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
    ax = axes['d']
    dx = 10e-3
    xzratio = 2.0
    x = np.linspace(-dx / 2, dx / 2, nperslice)
    zoffset = 3e-3
    z = np.linspace(zoffset, zoffset + dx * xzratio, nperslice)
    amps = np.array([
        [source.Vext(source.I, source.vectorialDistance((xx, zz))) for zz in z]
        for xx in x])
    xedges, zedges = [XYMap.computeMeshEdges(xx, 'lin') for xx in [x, z]]
    sm = ax.pcolormesh(xedges * 1e3, zedges * 1e3, amps.T, cmap='viridis', rasterized=True)
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_visible(False)
    scale = np.ptp(x) / 2 * 1e3  # mm
    s = ax.spines['left']
    s.set_bounds(zoffset * 1e3, zoffset * 1e3 + scale)
    s.set_position(('outward', 3))
    s.set_linewidth(3.0)
    ax.set_ylabel(f'{scale:.0f} mm', fontsize=fontsize)

    # EL: FWHM of transverse distribution vs source distance
    ax = axes['e']
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
    norm, sm = setNormalizer('viridis', (0, 1), scale='lin')
    for k in ['cbar1', 'cbar2']:
        ax = axes[k]
        cbar = fig.colorbar(sm, cax=ax)
        ax.set_ylabel('normalized \namplitude', fontsize=fontsize)
        cbar.set_ticks([])
        ax.set_title('1', fontsize=fontsize)
        ax.set_xlabel('0', fontsize=fontsize)
        ax.set_aspect(1.0)

    for ax in axes.values():
        for sk in ['top', 'right']:
            ax.spines[sk].set_visible(False)
    for axkey in ['b1', 'b2', 'b3', 'b4', 'c', 'e']:
        ax = axes[axkey]
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fontsize)

    if args.save:
        fig.savefig(os.path.join(figdir, f'figS1_raw.pdf'), transparent=True)

plt.show()

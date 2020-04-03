# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-26 17:11:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-03 12:24:20

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal

from PySONIC.plt import GroupedTimeSeries, CompTimeSeries
from PySONIC.utils import logger, si_format, getPow10

from .core import *
from .utils import loadData, chronaxie, extractIndexesFromLabels


class SectionGroupedTimeSeries(GroupedTimeSeries):
    ''' Plot the time evolution of grouped variables in a specific section. '''

    def __init__(self, section_id, filepaths, pltscheme=None):
        ''' Constructor. '''
        self.section_id = section_id
        super().__init__(filepaths, pltscheme=pltscheme)

    @staticmethod
    def getModel(meta):
        return getModel(meta)

    def figtitle(self, model, meta):
        return super().figtitle(model, meta) + f' - {self.section_id} section'

    def getData(self, entry, frequency=1, trange=None):
        if entry is None:
            raise ValueError('non-existing data')
        if isinstance(entry, str):
            data, meta = loadData(entry, frequency)
        else:
            data, meta = entry
        data = data[self.section_id]
        data = data.iloc[::frequency]
        if trange is not None:
            tmin, tmax = trange
            data = data.loc[(data['t'] >= tmin) & (data['t'] <= tmax)]
        return data, meta

    def render(self, *args, **kwargs):
        figs = super().render(*args, **kwargs)
        for fig in figs:
            title = fig.canvas.get_window_title()
            fig.canvas.set_window_title(title + f'_{self.section_id}')


class SectionCompTimeSeries(CompTimeSeries):
    ''' Plot the time evolution of a specific variable across sections, for a specific condition '''

    def __init__(self, filepath, varname, sections):
        self.entry = filepath[0]
        self.model = None
        self.ref_meta = None
        super().__init__(sections, varname)

    def getModel(self, meta):
        if self.model is None:
            self.ref_meta = meta.copy()
            del self.ref_meta['section']
            self.model = getModel(meta)
        else:
            comp_meta = meta.copy()
            del comp_meta['section']
            if comp_meta != self.ref_meta:
                return getModel(meta)
        return self.model

    def getData(self, section, frequency=1, trange=None):
        if self.entry is None:
            raise ValueError('non-existing data')
        if isinstance(self.entry, str):
            data, meta = loadData(self.entry, frequency)
        else:
            data, meta = self.entry
        meta = meta.copy()
        meta['section'] = section
        data = data[section]
        data = data.iloc[::frequency]
        if trange is not None:
            tmin, tmax = trange
            data = data.loc[(data['t'] >= tmin) & (data['t'] <= tmax)]
        return data, meta

    def getCompLabels(self, comp_values):
        return np.arange(len(comp_values)), comp_values

    def checkColors(self, colors):
        if colors is None:
            nlevels = len(self.filepaths)
            if nlevels < 4:
                colors = [f'C{i}' for i in range(nlevels)]
            else:
                norm = mpl.colors.Normalize(0, nlevels - 1)
                sm = plt.cm.ScalarMappable(norm=norm, cmap='plasma')
                sm._A = []
                colors = [sm.to_rgba(i) for i in range(nlevels)]
        return colors

    def addLegend(self, fig, ax, handles, labels, fs, color=None, ls=None):
        nlabels = len(labels)
        use_cbar_legend = False
        if nlabels > 3:
            out = extractIndexesFromLabels(labels)
            if out is not None:
                prefix, label_indexes = out
                sorted_indexes = (np.array(range(nlabels)) + min(label_indexes)).tolist()
                if label_indexes == sorted_indexes:
                    use_cbar_legend = True
        if use_cbar_legend:
            colors = [h.get_color() for h in handles]
            cmap = mpl.colors.ListedColormap(colors)
            bounds = np.arange(nlabels + 1) + 1 + min(label_indexes)
            ticks = bounds[:-1] + 0.5
            if nlabels > 10:
                ticks = [ticks[0], ticks[-1]]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            fig.subplots_adjust(left=0.1, right=0.8, bottom=0.15, top=0.95, hspace=0.5)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.8])
            cb = mpl.colorbar.ColorbarBase(
                cbar_ax,
                cmap=cmap,
                norm=norm,
                ticks=ticks,
                boundaries=bounds,
                format='%1i'
            )
            cbar_ax.tick_params(axis='both', which='both', length=0)
            cbar_ax.set_title(f'{prefix} index', size=fs)
        else:
            super().addLegend(fig, ax, handles, labels, fs, color=None, ls=None)


def thresholdCurve(fiber, x, thrs, thrs2=None,
                   xname='duration', xfactor=1e6, xunit='s',
                   yname='current', yfactor=1, yunit='A',
                   y2name='charge', y2factor=1, y2unit='C',
                   scale='log', plot_chr=True, fs=12, colors=None, limits=None, xlimits=None):

    if colors is None:
        colors = plt.get_cmap('tab10').colors

    fig, ax = plt.subplots()
    prefix = si_format(1 / yfactor, space='')[1:]
    ax.set_title(f'{fiber}', fontsize=fs)
    ax.set_xlabel(f'{xname} ({si_format(1 / xfactor, space="")[1:]}{xunit})', fontsize=fs)
    ax.set_ylabel(f'threshold {yname} ({prefix}{yunit})', fontsize=fs)
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
    testvalues = thrs[list(thrs.keys())[0]]
    testvalues = testvalues[np.logical_not(np.isnan(testvalues))]
    if np.all(testvalues < 0.):
        thrs = {k: -v for k, v in thrs.items()}
        if thrs2 is not None:
            thrs2 = {k: -v for k, v in thrs2.items()}
    for i, k in enumerate(thrs.keys()):
        ax.plot(x * xfactor, thrs[k] * yfactor, label=k, color=colors[i])
        if plot_chr:
            ax.axvline(chronaxie(x, thrs[k]) * xfactor, linestyle='-.', color=colors[i])
    if scale != 'log':
        if xlimits is None:
            ax.set_xlim(0., x.max() * xfactor)
            ax.set_ylim(0., ax.get_ylim()[1])
        else:
            ax.set_xlim(xlimits[0] * xfactor, xlimits[1] * xfactor)
    else:
        ax.set_xlim(x.min() * xfactor, x.max() * xfactor)
        if limits is None:
            ymin = np.nanmin([np.nanmin(v) for v in thrs.values()])
            ymax = np.nanmax([np.nanmax(v) for v in thrs.values()])
            ymin = getPow10(ymin * yfactor, 'down')
            ymax = getPow10(ymax * yfactor, 'up')
        else:
            ymin = limits[0] * yfactor
            ymax = limits[1] * yfactor
        ax.set_ylim(ymin, ymax)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.legend(fontsize=fs / 1.8, frameon=False, loc='upper left', ncol=2)

    if thrs2 is not None:
        ax2 = ax.twinx()
        prefix = si_format(1 / y2factor, space='')[1:]
        ax2.set_ylabel(f'threshold {y2name} ({prefix}{y2unit})', fontsize=fs)
        if scale == 'log':
            ax2.set_yscale('log')
        for i, k in enumerate(thrs2.keys()):
            ax2.plot(x * xfactor, thrs2[k] * y2factor, linestyle='--', color=colors[i])
        if scale != 'log':
            ax2.set_ylim(0., ax2.get_ylim()[1])
        else:
            ymin2 = min([np.nanmin(v) for v in thrs2.values()])
            ymax2 = max([np.nanmax(v) for v in thrs2.values()])
            ymin2 = getPow10(ymin2 * y2factor, 'down')
            ymax2 = getPow10(ymax2 * y2factor, 'up')
            ax2.set_ylim(ymin2, ymax2)
    return fig


def strengthDurationCurve(fiber, durations, thrs, **kwargs):
    return thresholdCurve(fiber, durations, thrs, xname='duration',
                          xfactor=1e6, xunit='s', **kwargs)


def strengthDistanceCurve(fiber, distances, thrs, **kwargs):
    return thresholdCurve(fiber, distances, thrs, xname='distance',
                          xfactor=1e3, xunit='m', plot_chr=False, **kwargs)


def plotConvergenceResults(df, inkey, outkeys, rel_eps_thr_Ithr=0.05, rel_eps_thr=0.01,
                           axesdirection='d'):
    ''' Plot output metrics of convergence study.

        :param df: dataframe with input values (parameter of interest) and output metrics
        :param inkey: key of the input parameter
        :param outkeys: keys of the output parameters
        :param direction: direction of the x axes used also to find the threshold
            ('a' ascending, 'd' descending)
        :param rel_eps_thr: relative error threshold for the output metrics
        :return: figure handle
    '''
    # Initialize dictionaries
    eps = {}      # relative errors of each output metrics
    xin_thr = {}  # threshold input values according to each output metrics

    # Extract input range and figure out if it must be reversed
    xin = df[inkey].values
    # reverse = xin[-1] < xin[0]

    # Create figure backbone
    fig, axes = plt.subplots(len(outkeys) + 1, 1, figsize=(6, 9))
    ax = axes[-1]
    ax.set_xlabel(inkey)
    ax.set_ylabel('relative errors (%)')
    ax.axhline(rel_eps_thr * 100, linestyle=':', color='k',
               label=f'{rel_eps_thr * 1e2:.1f} % threshold')
    ax.axhline(rel_eps_thr_Ithr * 100, linestyle='-.', color='k',
               label=f'{rel_eps_thr_Ithr * 1e2:.1f} % threshold')

    # For each output
    for i, k in enumerate(outkeys):
        xout = df[k].values

        # Plot output evolution
        axes[i].set_ylabel(k)
        axes[i].plot(xin, xout, c='k')
        ymin, ymax, yconv = np.nanmin(xout), np.nanmax(xout), xout[-1]
        yptp, ydelta = ymax - ymin, 0.8 * yconv
        if ymax - yconv > yconv - ymin:
            ytopaxis = min(yconv + ydelta, ymax + 0.05 * yptp)
            axes[i].set_ylim(
                ymin - 0.08 * (ytopaxis - ymin), ytopaxis)
        else:
            ybottomaxis = max(yconv - ydelta, ymin - 0.05 * yptp)
            axes[i].set_ylim(
                ybottomaxis, ymax + 0.08 * (ymax - ybottomaxis))

        # Compute and plot relative error w.r.t. reference (last) value
        xref = xout[-1]
        eps[k] = np.abs((xout - xref) / xref)
        axes[-1].plot(xin, eps[k] * 100, label=k, c=f'C{i}')

        # Compute and plot input value yielding threshold relative error
        j = eps[k].size - 1
        if i == 0:
            rel_thr = rel_eps_thr_Ithr
        else:
            rel_thr = rel_eps_thr
        while eps[k][j] <= rel_thr and j > 0:
            j -= 1
        xin_thr[k] = xin[j + 1]
        axes[-1].axvline(xin_thr[k], linestyle='dashed', color=f'C{i}')

    # Compute minimal required input value to satisfy all relative error threshold on all inputs
    # logger.info(f'Relative error threshold Ithr = {rel_eps_thr_Ithr * 1e2:.1f} %')
    # logger.info(f'Relative error threshold for CV and dV = {rel_eps_thr * 1e2:.1f} %')
    if axesdirection == 'd':
        logger.info(f'max {inkey} = {min(xin_thr.values()):.2e}')
    else:
        logger.info(f'To reach convergence {inkey} = {max(xin_thr.values()):.2e}')
    logger.info(f'Convergence excitation current threshold = {(df.values[-1,2]*1e9):.2f} nA')
    logger.info(f'Convergence conduction velocity = {df.values[-1,3]:.2f} m/s')
    logger.info(f'Convergence spike amplitude = {df.values[-1,4]:.2f} mV')

    # Post-process figure
    axes[-1].set_ylim(-5, 30)
    axes[-1].legend(frameon=False)
    for ax in axes:
        ax.set_xscale('log')
        if axesdirection == 'd':
            ax.invert_xaxis()
    fig.tight_layout()

    return fig


def plotFiberXCoords(fiber, fs=12):
    ''' Plot the x coordinates of a fiber model, per section type. '''
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_title(f'{fiber} - x-coordinates per section type', fontsize=fs)
    ax.set_xlabel('section mid-point x-coordinate (mm)', fontsize=fs)
    ax.set_ylabel('section type', fontsize=fs)
    ax.set_yticks(range(len(fiber.sectypes)))
    ax.set_yticklabels(fiber.sectypes)
    for i, (k, xcoords) in enumerate(fiber.getXCoords().items()):
        ax.plot(xcoords * 1e3, np.ones(xcoords.size) * i, '|', markersize=15, label=k)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    fig.tight_layout()
    return fig


def plotFieldDistribution(fiber, source, fs=12):
    ''' Plot a source's field distribution over a fiber, per section type. '''
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_title(f'{fiber} - field distribution from {source}', fontsize=fs)
    ax.set_xlabel('section mid-point x-coordinate (mm)', fontsize=fs)
    ax.set_ylabel('Extracellular voltage (mV)', fontsize=fs)
    field_dict = source.computeDistributedAmps(fiber)
    for k, xcoords in fiber.getXCoords().items():
        ax.plot(xcoords * 1e3, field_dict[k], '.', label=k)
    ylims = ax.get_ylim()
    if source.I < 0.:
        ax.set_ylim(ylims[0], -0.05 * ylims[0])
    else:
        ax.set_ylim(-0.05 * ylims[1], ylims[1])
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.legend(fontsize=fs, frameon=False)
    fig.tight_layout()
    return fig


def plotMRGLookups(fiberD_range=None, interp_methods=None, fs=12):
    ''' Plot MRG morphological parameters interpolated over a fiber diameter range. '''

    # Define diameters ranges
    ref_diams = mrg_lkp.refs['fiberD']
    if fiberD_range is None:
        fiberD_range = bounds(ref_diams)
    diams = np.linspace(*fiberD_range, 100)

    # Define interpolation methods
    if interp_methods is None:
        interp_methods = mrg_lkp.interp_choices

    # Define factor function
    factor = lambda k: 1e0 if k == 'nlayers' else 1e6

    # Create figure backbone
    nouts = len(mrg_lkp.outputs)
    fig, axes = plt.subplots(1, nouts, figsize=(nouts * 3, 2.5))
    for ax, k in zip(axes, mrg_lkp.keys()):
        yunit = '' if k == 'nlayers' else '(um)'
        ax.set_xlabel('fiber diameter (um)', fontsize=fs)
        ax.set_ylabel(f'{k} {yunit}', fontsize=fs)
        ax.plot(ref_diams * 1e6, mrg_lkp[k] * factor(k), '.', c='k')
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)

    # Interpolate over fiber range with each method and plot resulting profiles
    default_method = mrg_lkp.interp_method
    for interp_method in interp_methods:
        mrg_lkp.interp_method = interp_method
        interp_mrg_lkp = mrg_lkp.project('fiberD', diams)
        label = f'{interp_method} method'
        for ax, (k, v) in zip(axes, interp_mrg_lkp.items()):
            ax.plot(diams * 1e6, v * factor(k), label=label)

    # Set lookup interpolation method back to default
    mrg_lkp.interp_method = default_method

    axes[0].legend(frameon=False)
    title = fig.suptitle(f'MRG morphological parameters', fontsize=fs)
    fig.tight_layout()
    title.set_y(title._y + 0.03)
    return fig


def plotFiberDiameterDistributions(n=50, fs=12):
    ''' Plot the diameter distribution of different types of peripheral fibers. '''
    fibers_dict = {
        'Aα': {
            'bounds': (13, 20),
            'myelinated': True,
            'implemented': True,
            'label': 'myelinated'
        },
        'Aβ': {
            'bounds': (6, 12),
            'myelinated': True,
            'implemented': True
        },
        'Aδ': {
            'bounds': (1, 5),
            'myelinated': True,
            'implemented': False
        },
        'C': {
            'bounds': (0.2, 1.5),
            'myelinated': False,
            'implemented': True,
            'label': 'unmyelinated'
        }
    }

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.set_yticks([])
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('diameter (um)', fontsize=fs)
    for item in ax.get_xticklabels():
        item.set_fontsize(fs)
    for key in ['top', 'left', 'right']:
        ax.spines[key].set_visible(False)
    g = signal.gaussian(n, std=8)
    for k, d in fibers_dict.items():
        drange = np.linspace(*d['bounds'], n)
        color = 'royalblue' if d['myelinated'] else 'orangered'
        label = d.get('label', None)
        ax.plot(drange, g, color, linewidth=2.5, label=label)
        ax.text(np.mean(d['bounds']), 1.07, k, color=color, size=fs + 2, weight='bold',
                horizontalalignment='center')
        if d['implemented']:
            ax.fill_between(drange, 0, g, color=color, alpha=0.5)
    ax.legend(fontsize=fs, frameon=False, bbox_to_anchor=(.9, 1), loc='upper left')
    fig.tight_layout()

    return fig

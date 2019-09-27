# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-26 17:11:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-27 13:44:07

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from PySONIC.plt import GroupedTimeSeries, CompTimeSeries
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format, getPow10

from .core import ExtendedSonicNode, VextSennFiber, IinjSennFiber, SonicSennFiber
from .utils import loadData, chronaxie, extractIndexesFromLabels


def getModel(meta):
    ''' Return appropriate model object based on a dictionary of meta-information. '''
    simkey = meta['simkey']
    if simkey == 'nano_ext_SONIC':
        model = ExtendedSonicNode(
            getPointNeuron(meta['neuron']),
            meta['rs'],
            a=meta['a'],
            Fdrive=meta['Fdrive'],
            fs=meta['fs'],
            deff=meta['deff']
        )
    elif simkey == 'senn_Vext':
        model = VextSennFiber(
            getPointNeuron(meta['neuron']),
            meta['fiberD'], meta['nnodes'], rs=meta['rs'])
    elif simkey == 'senn_Iinj':
        model = IinjSennFiber(
            getPointNeuron(meta['neuron']),
            meta['fiberD'], meta['nnodes'], rs=meta['rs'])
    elif simkey == 'senn_SONIC':
        model = SonicSennFiber(
            getPointNeuron(meta['neuron']), meta['fiberD'], meta['nnodes'],
            a=meta['a'], Fdrive=meta['Fdrive'], fs=meta['fs'], rs=meta['rs'])
    else:
        raise ValueError(f'Unknown model type:{simkey}')
    return model


def figtitle(meta):
    ''' Return appropriate title based on simulation metadata. '''
    if meta['DC'] < 1:
        wavetype = 'PW'
        suffix = ', {:.2f}Hz PRF, {:.0f}% DC'.format(meta['PRF'], meta['DC'] * 1e2)
    else:
        wavetype = 'CW'
        suffix = ''
    simkey = meta['simkey']
    if simkey == 'nano_ext_SONIC':
        return 'extended SONIC node ({} neuron, {:.1f}nm, {:.0f}% coverage, deff = {:.0f} nm, rs = {:.0f} Ohm.cm): {} A-STIM {:.0f}kHz {:.2f}kPa, {:.0f}ms{}'.format(
                    meta['neuron'], meta['a'] * 1e9, meta['fs'] * 1e2, meta['deff'] * 1e9, meta['rs'], wavetype, meta['Fdrive'] * 1e-3,
                    meta['Adrive'] * 1e-3, meta['tstim'] * 1e3, suffix, meta['method'])
    elif simkey == 'senn_Vext':
        return 'SENN fiber ({} neuron, d = {:.1f}um, {} nodes), ({:.1f}, {:.1f})mm point-source {} E-STIM {:.2f}mA, {:.2f}ms{}'.format(
            meta['neuron'],
            meta['fiberD'] * 1e6,
            meta['nnodes'],
            meta['psource'].x * 1e3, meta['psource'].z * 1e3,
            wavetype,
            meta['A'] * 1e3,
            meta['tstim'] * 1e3,
            suffix
        )
    elif simkey == 'senn_Iinj':
        return 'SENN fiber ({} neuron, d = {:.1f}um, {} nodes), node {} point-source {} E-STIM {:.2f}nA, {:.2f}ms{}'.format(
            meta['neuron'],
            meta['fiberD'] * 1e6,
            meta['nnodes'],
            meta['psource'].inode,
            wavetype,
            meta['A'] * 1e9,
            meta['tstim'] * 1e3,
            suffix
        )
    elif simkey == 'senn_SONIC':
        psource = meta['psource']
        try:
            s = f'node {psource.inode} point-source'
        except AttributeError:
            s = f'{si_format(psource.r)}m radius planar transducer'

        return 'SONIC SENN fiber ({} neuron, a = {:.1f}nm, d = {:.1f}um, {} nodes), {} {} A-STIM {:.0f}kHz, {:.2f}kPa, {:.2f}ms{}'.format(
            meta['neuron'],
            meta['a'] * 1e9,
            meta['fiberD'] * 1e6,
            meta['nnodes'],
            s,
            wavetype,
            meta['Fdrive'] * 1e-3,
            meta['A'] * 1e-3,
            meta['tstim'] * 1e3,
            suffix
        )

    return 'dummy title'


class SectionGroupedTimeSeries(GroupedTimeSeries):
    ''' Plot the time evolution of grouped variables in a specific section. '''

    def __init__(self, section_id, filepaths, pltscheme=None):
        ''' Constructor. '''
        self.section_id = section_id
        super().__init__(filepaths, pltscheme=pltscheme)

    @staticmethod
    def getModel(meta):
        return getModel(meta)

    def figtitle(self, *args, **kwargs):
        return figtitle(*args, **kwargs) + f' - {self.section_id} section'

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
    ''' Plot the time evolution of a specific variable across sections, for one specific condition '''

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

    def figtitle(self, *args, **kwargs):
        return figtitle(*args, **kwargs)

    def checkColors(self, colors):
        if colors is None:
            nlevels = len(self.filepaths)
            if nlevels < 4:
                colors = [f'C{i}' for i in range(nlevels)]
            else:
                norm = mpl.colors.Normalize(0, nlevels - 1)
                sm = plt.cm.ScalarMappable(norm=norm, cmap='autumn')
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
                sorted_indexes = list(range(nlabels))
                if label_indexes == sorted_indexes:
                    logger.debug(f'creating colorbar legend for {prefix} index')
                    use_cbar_legend = True
        if use_cbar_legend:
            colors = [h.get_color() for h in handles]
            cmap = mpl.colors.ListedColormap(colors)
            bounds = np.arange(nlabels + 1) + 1
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


def thresholdCurve(fiber, x, thrs,
                   xname='duration', xfactor=1e6, xunit='s',
                   yname='current', yfactor=1, yunit='A',
                   scale='log', plot_chr=True, fs=12):

    fig, ax = plt.subplots()
    prefix = si_format(1 / yfactor, space='')[1:]
    ax.set_title(f'{fiber} - strength-{xname} curve', fontsize=fs)
    ax.set_xlabel(f'{xname} ({si_format(1 / xfactor, space="")[1:]}{xunit})', fontsize=fs)
    ax.set_ylabel(f'threshold {yname} ({prefix}{yunit})', fontsize=fs)
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
    if np.all(thrs[list(thrs.keys())[0]] < 0.):
        thrs = {k: -v for k, v in thrs.items()}
    for i, k in enumerate(thrs.keys()):
        ax.plot(x * xfactor, thrs[k] * yfactor, color=f'C{i}', label=k)
        if plot_chr:
            ax.axvline(chronaxie(x, thrs[k]) * xfactor, linestyle='--', color=f'C{i}')
    if scale != 'log':
        ax.set_xlim(0., x.max() * xfactor)
        ax.set_ylim(0., ax.get_ylim()[1])
    else:
        ax.set_xlim(x.min() * xfactor, x.max() * xfactor)
        ymin = min([np.nanmin(v) for v in thrs.values()])
        ymax = max([np.nanmax(v) for v in thrs.values()])
        ymin = getPow10(ymin * yfactor ,'down')
        ymax = getPow10(ymax * yfactor ,'up')
        ax.set_ylim(ymin, ymax)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.legend(fontsize=fs, frameon=False)
    return fig


def strengthDurationCurve(fiber, durations, thrs, **kwargs):
    return thresholdCurve(fiber, durations, thrs, xname='duration', xfactor=1e6, xunit='s', **kwargs)


def strengthDistanceCurve(fiber, distances, thrs, **kwargs):
    return thresholdCurve(fiber, distances, thrs, xname='distance', xfactor=1e3, xunit='m',
                          plot_chr=False, **kwargs)
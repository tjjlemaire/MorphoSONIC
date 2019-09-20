# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-26 17:11:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-20 11:30:51

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from PySONIC.plt import GroupedTimeSeries, CompTimeSeries
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import si_format, getPow10

from .core import ExtendedSonicNode, VextSennFiber, IinjSennFiber, SonicSennFiber
from .utils import loadData, chronaxie


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
        return 'SONIC SENN fiber ({} neuron, a = {:.1f}nm, d = {:.1f}um, {} nodes), node {} point-source {} A-STIM {:.0f}kHz, {:.2f}kPa, {:.2f}ms{}'.format(
            meta['neuron'],
            meta['a'] * 1e9,
            meta['fiberD'] * 1e6,
            meta['nnodes'],
            meta['psource'].inode,
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
                norm = matplotlib.colors.Normalize(0, nlevels - 1)
                sm = plt.cm.ScalarMappable(norm=norm, cmap='autumn')
                sm._A = []
                colors = [sm.to_rgba(i) for i in range(nlevels)]
        return colors


def strengthDurationCurve(fiber, durations, thrs, yfactor=1, scale='log', unit='A', name='current',
                          plot_chr=True, fs=12):
    fig, ax = plt.subplots()
    prefix = si_format(1 / yfactor, space='')[1:]
    ax.set_title(f'{fiber} - strength-duration curve', fontsize=fs)
    ax.set_xlabel('duration (us)', fontsize=fs)
    ax.set_ylabel(f'threshold {name} ({prefix}{unit})', fontsize=fs)
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
    if np.all(thrs[list(thrs.keys())[0]] < 0.):
        thrs = {k: -v for k, v in thrs.items()}
    for i, k in enumerate(thrs.keys()):
        ax.plot(durations * 1e6, thrs[k] * yfactor, color=f'C{i}', label=k)
        if plot_chr:
            ax.axvline(chronaxie(durations, thrs[k]) * 1e6, linestyle='--', color=f'C{i}')
    if scale != 'log':
        ax.set_xlim(0., durations.max() * 1e6)
        ax.set_ylim(0., ax.get_ylim()[1])
    else:
        ax.set_xlim(durations.min() * 1e6, durations.max() * 1e6)
        ymin = min([np.nanmin(v) for v in thrs.values()])
        ymax = max([np.nanmax(v) for v in thrs.values()])
        ymin = getPow10(ymin * yfactor ,'down')
        ymax = getPow10(ymax * yfactor ,'up')
        ax.set_ylim(ymin, ymax)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.legend(fontsize=fs, frameon=False)
    return fig

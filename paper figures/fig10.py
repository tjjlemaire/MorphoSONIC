# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-15 20:31:54

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format, selectDirDialog
from PySONIC.plt import TimeSeriesPlot
from PySONIC.parsers import FigureParser
from ExSONIC.core import SonicNode, ExtendedSonicNode

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basEName
figbase = os.path.splitext(__file__)[0]


def prependTimeSeries(df, tonset):
    df0 = pd.DataFrame([df.iloc[0]])
    df = pd.concat([df0, df], ignore_index=True)
    df['t'][0] = -tonset
    return df


def plotComparativeResponses(pneuron, rs, deff, a, fs, Fdrive, Adrive, tstim, toffset, PRF, DC,
                             fontsize=12):

    # Simulate punctual SONIC model with specific membrane coverage
    punctual_model = SonicNode(pneuron, a=a, Fdrive=Fdrive, fs=fs)
    punctual_data, _ = punctual_model.simulate(Adrive, tstim, toffset, PRF, DC)

    # Simulate extended SONIC model with specific membrane coverage
    ext_model = ExtendedSonicNode(
        pneuron, rs, a=a, Fdrive=Fdrive, fs=fs, deff=deff)
    ext_data, _ = ext_model.simulate(Adrive, tstim, toffset, PRF, DC)

    # Add onset to solutions
    tonset = 5e-3
    punctual_data = prependTimeSeries(punctual_data, tonset)
    for k, df in ext_data.items():
        ext_data[k] = prependTimeSeries(df, tonset)

    # Get stimulus patches
    t = punctual_data['t'].values  # s
    stimon = punctual_data['stimstate']
    tpatch_on, tpatch_off = TimeSeriesPlot.getStimPulses(t, stimon)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    for ax in axes:
        for key in ['right', 'top']:
            ax.spines[key].set_visible(False)
        ax.set_xlim(-tonset * 1e3, (tstim + toffset) * 1e3)
        ax.set_ylim(-100, 50)
        ax.set_yticks(ax.get_ylim())
        for i in range(tpatch_on.size):
            ax.axvspan(tpatch_on[i] * 1e3, tpatch_off[i] * 1e3, edgecolor='none',
                       facecolor='#8A8A8A', alpha=0.2)
    ax = axes[0]
    ax.set_ylabel('$\\rm V_m^*\ (mV)$', fontsize=fontsize)
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax = axes[1]
    ax.set_xlabel('time (ms)', fontsize=fontsize)
    ax.set_ylabel('$\\rm Q_m\ (nC/cm^2)$', fontsize=fontsize)
    ax.set_xticks([0, (tstim + toffset) * 1e3])
    ax.set_xticklabels(['{:.0f}'.format(x) for x in ax.get_xticks()])

    # Plot membrane potential and charge density profiles
    colors = plt.get_cmap('Paired').colors[:2]
    linestyles = ['-', '--']
    for i, (key, df) in enumerate(ext_data.items()):
        axes[0].plot(df['t'] * 1e3, df['Vm'], linestyles[i], c=colors[i], label=f'ext. model: {key}')
        axes[1].plot(df['t'] * 1e3, df['Qm'] * 1e5, linestyles[i], c=colors[i])
    axes[0].plot(t * 1e3, punctual_data['Vm'], c='dimgrey', label='punctual model')
    axes[1].plot(t * 1e3, punctual_data['Qm'] * 1e5, c='dimgrey')

    # Add legend
    axes[0].legend(
        frameon=False, fontsize=fontsize, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=3, mode='expand', borderaxespad=0.)

    # Post-process figure
    for ax in axes:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fontsize)
    fig.canvas.set_window_title(figbase + 'a')

    return fig


def computeAthr0D(pneuron, a, Fdrive, tstim, toffset, PRF, DC, fs_range):
    Athr = np.empty(fs_range.size)
    for i, fs in enumerate(fs_range):
        logger.info('computing threshold amplitude for fs = {:.0f}%'.format(fs * 1e2))
        model = SonicNode(pneuron, a=a, Fdrive=Fdrive, fs=fs)
        Athr[i] = model.titrate(tstim, toffset, PRF=PRF, DC=DC)
        model.clear()
    return Athr


def computeAthr1D(pneuron, a, Fdrive, tstim, toffset, PRF, DC, rs, deff, fs_range):
    Athr = np.empty(fs_range.size)
    for i, fs in enumerate(fs_range):
        logger.info('computing threshold amplitude for deff = {}m, fs = {:.0f}%'.format(
            si_format(deff), fs * 1e2))
        model = ExtendedSonicNode(pneuron, rs, a=a, Fdrive=Fdrive, fs=fs, deff=deff)
        Athr[i] = model.titrate(tstim, toffset, PRF=PRF, DC=DC, xfunc=model.isExcited)
        model.clear()
    return Athr


def getData(fpath, func, args, fs_range):
    fs_key = 'fs (%)'
    Athr_key = 'Athr (kPa)'
    if os.path.isfile(fpath):
        logger.info('loading data from file: "{}"'.format(fpath))
        df = pd.read_csv(fpath, sep=',')
        assert np.allclose(df[fs_key].values, fs_range * 1e2), 'fs range not matching'
        Athr = df[Athr_key].values
    else:
        logger.info('computing data')
        Athr = func(*args, fs_range)
        df = pd.DataFrame({fs_key: fs_range * 1e2, Athr_key: Athr})
        df.to_csv(fpath, sep=',', index=False)
    return Athr


def plotComparativeThresholds(inputdir, pneuron, rs, deff, a, fs, Fdrive, tstim, toffset, PRF, DC,
                              fontsize=12):

    # Determine naming convention for intermediate files
    fcode = 'Athr_vs_fs_{}s'.format(si_format(tstim, 0, space=''))
    fname_0D = '{}_0D.csv'.format(fcode)
    fname_1D = '{}_1D.csv'.format(fcode)

    # Compute threshold amplitudes with point-neuron model
    args0D = [pneuron, a, Fdrive, tstim, toffset, PRF, DC]
    Athr0D = getData(os.path.join(inputdir, fname_0D), computeAthr0D, args0D, fs)

    # Compute threshold amplitudes with compartmental neuron model
    args1D = args0D + [rs, deff]
    Athr1D = getData(os.path.join(inputdir, fname_1D), computeAthr1D, args1D, fs)

    # Plot threshold curves as a function of coverage fraction, for various sub-membrane depths
    fig, ax = plt.subplots()
    ax.set_xlabel('sonophore membrane coverage (%)', fontsize=fontsize)
    ax.set_ylabel('amplitude (kPa)', fontsize=fontsize)
    ax.plot(fs * 100, Athr0D * 1e-3, c='dimgrey', label='punctual model')
    ax.plot(fs * 100, Athr1D * 1e-3, c='C0', label='extended model')
    ax.set_yscale('log')
    ax.set_xlim(0, 100)
    ax.set_ylim(1e1, 6e2)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fontsize)
    ax.legend(frameon=False, fontsize=fontsize, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=2, mode='expand', borderaxespad=0.)

    fig.canvas.set_window_title(figbase + 'b')

    return fig


def main():
    parser = FigureParser(['a', 'b'])
    parser.addInputDir()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    figset = args['subset']
    inputdir = args['inputdir']
    logger.info('Generating panels {} of {}'.format(figset, figbase))

    # Model parameters
    pneuron = getPointNeuron('RS')
    a = 32e-9  # m
    deff = 100e-9  # m
    rs = 1e2  # Ohm.cm

    # Stimulation parameters
    Fdrive = 500e3  # Hz
    PRF = 100.  # Hz
    DC = 1.

    # Generate figures
    figs = []
    if 'a' in figset:
        fs = 0.5
        Adrive = 50e3  # kPa
        tstim = 100e-3  # s
        toffset = 50e-3  # s
        figs.append(plotComparativeResponses(
            pneuron, rs, deff, a, fs, Fdrive, Adrive, tstim, toffset, PRF, DC))
    if 'b' in figset:
        fs_range = np.linspace(0.01, 0.99, 99)
        tstim = 1000e-3  # s
        toffset = 0e-3  # s
        figs.append(plotComparativeThresholds(
            inputdir, pneuron, rs, deff, a, fs_range, Fdrive, tstim, toffset, PRF, DC))

    if args['save']:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(args['outpudir'], figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()

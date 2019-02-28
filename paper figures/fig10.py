# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 10:51:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-02-28 10:44:47

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.neurons import getNeuronsDict
from PySONIC.utils import logger, si_format, selectDirDialog, getStimPulses
from ExSONIC.utils import radialGeometry
from ExSONIC._0D import Sonic0D
from ExSONIC._1D import SeriesConnector, Sonic1D


# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basename
figbase = os.path.splitext(__file__)[0]


def plotResponse(neuron, rs, deff, a, fs, Fdrive, Adrive, tstim, toffset, PRF, DC, verbose,
                 fontsize=12):

    Vm0 = neuron.Vm0
    Qm0 = Vm0 * neuron.Cm0 * 1e2

    # Initialize 0D model with specific membrane coverage
    model = Sonic0D(neuron, a=a, Fdrive=Fdrive, fs=fs, verbose=verbose)
    model.setUSdrive(Adrive)

    # Simulate 0D model and retrieve membrane potential and charge density profiles
    t_0D, y, stimon = model.simulate(tstim, toffset, PRF, DC, None, None)
    t_0D *= 1e3
    Qm_0D = y[0] * 1e5
    Vmeff_0D = y[1]

    # Initialize 0D model with specific membrane coverage
    nodeD, nodeL = radialGeometry(deff, a * 1e-3, fc=fs)
    interD = 1.  # um
    interL = 0.  # um
    connector = SeriesConnector(vref='Vmeff_{}'.format(neuron.name))
    model = Sonic1D(neuron, rs, nodeD, nodeL, interD=interD, interL=interL,
                    a=a, Fdrive=Fdrive, connector=connector, verbose=verbose)
    model.setUSdrive(Adrive, 'first')

    # Simulate 1D model and retrieve distributed membrane potential and charge density profiles
    t_1D, _, Qprobes, Vmeffprobes, _ = model.simulate(tstim, toffset, PRF, DC, None, None)
    t_1D *= 1e3  # ms
    Qprobes *= 1e5  # nC/cm2

    # Get stimulus patches
    npatches, tpatch_on, tpatch_off = getStimPulses(t_0D, stimon)

    # Add onset to signals
    tonset = -0.05 * (t_0D[-1] - t_0D[0])
    t_0D = np.hstack((np.array([tonset, 0.]), t_0D))
    Qm_0D = np.hstack((np.ones(2) * Qm0, Qm_0D))
    Vmeff_0D = np.hstack((np.ones(2) * Vm0, Vmeff_0D))

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    for ax in axes:
        for key in ['right', 'top']:
            ax.spines[key].set_visible(False)
        ax.set_xlim(tonset, (tstim + toffset) * 1e3)
        ax.set_ylim(-100, 50)
        ax.set_yticks(ax.get_ylim())
        for i in range(npatches):
            ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                       facecolor='#8A8A8A', alpha=0.2)


    colors = plt.get_cmap('Paired').colors[:2]

    # Plot membrane potential profiles
    ax = axes[0]
    ax.set_ylabel('$V_m^*\ (mV)$', fontsize=fontsize)
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.plot(t_1D, Vmeffprobes[0], c=colors[0], label='bilayer sonophore')
    ax.plot(t_1D, Vmeffprobes[1], '--', c=colors[1], label='surrounding membrane')
    ax.plot(t_0D, Vmeff_0D, c='dimgrey', label='spatially-averaged model')
    ax.legend(frameon=False, fontsize=fontsize, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=3, mode='expand', borderaxespad=0.)

    # Plot membrane charge density profiles
    ax = axes[1]
    ax.set_xlabel('time (ms)', fontsize=fontsize)
    ax.set_ylabel('$Q_m\ (nC/cm^2)$', fontsize=fontsize)
    ax.set_xticks([0, (tstim + toffset) * 1e3])
    ax.set_xticklabels(['{:.0f}'.format(x) for x in ax.get_xticks()])
    ax.plot(t_1D, Qprobes[0], c=colors[0])
    ax.plot(t_1D, Qprobes[1], '--', c=colors[1])
    ax.plot(t_0D, Qm_0D, c='dimgrey')

    for ax in axes:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fontsize)

    fig.canvas.set_window_title(figbase + 'a')

    return fig


def computeAthr0D(neuron, a, Fdrive, tstim, toffset, PRF, DC, fs_range, verbose=False):

    Athr = np.empty(fs_range.size)
    for i, fs in enumerate(fs_range):
        print('computing threshold amplitude for fs = {:.0f}%'.format(fs * 1e2))

        # Create extended SONIC model with specific US frequency and coverage fraction
        model = Sonic0D(neuron, a=a, Fdrive=Fdrive, fs=fs, verbose=verbose)

        # Compute threshold excitation amplitude
        Athr[i] = model.titrateUS(tstim, toffset, PRF, DC, None, None)

    return Athr


def computeAthr1D(neuron, a, Fdrive, tstim, toffset, PRF, DC, rs, deff, fs_range, verbose=False):

    connector = SeriesConnector(vref='Vmeff_{}'.format(neuron.name))

    Athr = np.empty(fs_range.size)
    for i, fs in enumerate(fs_range):
        print('computing threshold amplitude for deff = {}m, fs = {:.0f}%'.format(
            si_format(deff * 1e-6), fs * 1e2))
        nodeD, nodeL = radialGeometry(deff, a * 1e-3, fc=fs)

        # Create extended SONIC model with specific US frequency and connection scheme
        model = Sonic1D(neuron, rs, nodeD, nodeL, interD=1., interL=0.,
                        a=a, Fdrive=Fdrive, connector=connector, verbose=verbose)

        # Compute threshold excitation amplitude
        Athr[i] = model.titrateUS(tstim, toffset, PRF, DC, None, None, 'first')

    return Athr


def getData(fpath, func, args, fs_range, verbose):
    fs_key = 'fs (%)'
    Athr_key = 'Athr (kPa)'
    if os.path.isfile(fpath):
        print('loading data from file: "{}"'.format(fpath))
        df = pd.read_csv(fpath, sep=',')
        assert np.allclose(df[fs_key].values, fs_range * 1e2), 'fs range not matching'
        Athr = df[Athr_key].values
    else:
        print('computing data')
        Athr = func(*args, fs_range, verbose=verbose)
        df = pd.DataFrame({fs_key: fs_range * 1e2, Athr_key: Athr})
        df.to_csv(fpath, sep=',', index=False)
    return Athr


def thresholds_vs_fs(inputdir, neuron, rs, deff, a, fs, Fdrive, tstim, toffset, PRF, DC, verbose,
                     fontsize=12):

    # Determine naming convention for intermediate files
    fcode = 'Athr_vs_fs_{}s'.format(si_format(tstim, 0, space=''))
    fname_0D = '{}_0D.csv'.format(fcode)
    fname_1D = '{}_1D.csv'.format(fcode)

    # Compute threshold amplitudes with point-neuron model
    # assuming spatially averaged electrical system
    args0D = [neuron, a, Fdrive, tstim, toffset, PRF, DC]
    Athr0D = getData(os.path.join(inputdir, fname_0D), computeAthr0D, args0D, fs, verbose)

    # Compute threshold amplitudes with compartmental neuron model
    # assuming spatially distributed electrical systems with intracelular currents
    args1D = args0D + [rs, deff]
    Athr1D = getData(os.path.join(inputdir, fname_1D), computeAthr1D, args1D, fs, verbose)

    # Plot threshold curves as a function of coverage fraction, for various sub-membrane depths
    fig, ax = plt.subplots()
    ax.set_xlabel('sonophore membrane coverage (%)', fontsize=fontsize)
    ax.set_ylabel('threshold excitation amplitude (kPa)', fontsize=fontsize)
    ax.plot(fs * 100, Athr0D, c='dimgrey', label='spatially-averaged model')
    ax.plot(fs * 100, Athr1D, c='C0', label='spatially-distributed model')
    ax.set_yscale('log')
    ax.set_ylim(1e1, 6e2)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fontsize)
    # ax.legend(frameon=False, fontsize=fontsize)
    ax.legend(frameon=False, fontsize=fontsize, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=2, mode='expand', borderaxespad=0.)

    fig.canvas.set_window_title(figbase + 'b')

    return fig



def main():

    # Define argument parser
    ap = ArgumentParser()

    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-i', '--inputdir', type=str, default=None, help='Input directory')
    ap.add_argument('-f', '--figset', type=str, nargs='+', help='Figure set', default='all')
    ap.add_argument('-s', '--save', default=False, action='store_true',
                    help='Save output figures as pdf')

    # Parse arguments
    args = ap.parse_args()
    loglevel = logging.DEBUG if args.verbose is True else logging.INFO
    logger.setLevel(loglevel)
    inputdir = selectDirDialog() if args.inputdir is None else args.inputdir
    if inputdir == '':
        logger.error('No input directory chosen')
        return
    figset = args.figset
    if figset is 'all':
        figset = ['a', 'b']

    logger.info('Generating panels {} of {}'.format(figset, figbase))

    # Model parameters
    neuron = getNeuronsDict()['RS']()
    rs = 1e2  # Ohm.cm
    deff = 1e-1  # um

    # Stimulation parameters
    a = 32.  # nm
    Fdrive = 500.  # kHz
    PRF = 100.  # Hz
    DC = 1.

    figs = []

    if 'a' in figset:
        fs = 0.5
        Adrive = 50.  # kPa
        tstim = 100e-3  # s
        toffset = 50e-3  # s
        figs.append(plotResponse(
            neuron, rs, deff, a, fs, Fdrive, Adrive, tstim, toffset, PRF, DC, args.verbose))

    if 'b' in figset:
        fs_range = np.linspace(0.01, 0.99, 99)
        tstim = 1000e-3  # s
        toffset = 0e-3  # s
        figs.append(thresholds_vs_fs(
            inputdir, neuron, rs, deff, a, fs_range, Fdrive, tstim, toffset, PRF, DC, args.verbose))

    if args.save:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(inputdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()

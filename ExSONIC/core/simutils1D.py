# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 11:29:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-04 19:43:07

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

from PySONIC.utils import si_format, pow10_format
from ..plt import plotSignals
from ..utils import getSpikesTimings, getConductionSpeeds
from .sonic1D import Sonic1D, computeVext


def compareEStim(neuron, rs, connector, nodeD, nodeL, interD, interL, Iinj, tstim, toffset,
                 PRF, DC, nnodes=None, dt=None, atol=None, cmode='qual', verbose=False, z0=None,
                 config=None):
    ''' Compare results of E-STIM simulations of the 1D extended SONIC model with different sections
        connection schemes.
    '''

    # Initialize output variables
    keys = ['classic', 'custom']
    t = {}
    stimon = {}
    Vmeffprobes = {}
    tcomp = {}

    # Plot parameters
    fs = 10
    cmap = plt.cm.jet_r
    ls = {'classic': '-', 'custom': '--'}
    colors = {'classic': 'dimgrey', 'custom': 'silver'}
    indices = [1, 2]

    # Build model simulation with classic connect scheme
    k = 'classic'
    model = Sonic1D(neuron, rs, nodeD, nodeL, interD=interD, interL=interL, nnodes=nnodes,
                    connector=None, verbose=verbose)
    xcoords = model.getNodeCoordinates()  # um

    # If electrode z-coordinate is provided, compute extracellular potentials (mV) from Iinj (uA).
    # Otherwise, Iinj is interpreted as intracellular injections (mA/m2)
    if z0 is None:
        # If not current given, perform titration to find threshold excitation current
        if Iinj is None:
            Iinj = model.titrateIinjIntra(tstim, toffset, PRF, DC, dt, atol, config)
        model.setIinj(Iinj, config)
    else:
        # If not current given, perform titration to find threshold excitation current
        if Iinj is None:
            Iinj = model.titrateIinjExtra(z0, tstim, toffset, PRF, DC, dt, atol)
        Vexts = computeVext(model, Iinj, z0)
        model.setVext(Vexts)

    # Simulate model with determined stimulus current amplitude
    tstart = time.time()
    t[k], stimon[k], _, Vmeffprobes[k], *_ = model.simulate(tstim, toffset, PRF, DC, dt, atol)
    tcomp[k] = time.time() - tstart

    # Run model simulation with custom connect scheme
    k = 'custom'
    model = Sonic1D(neuron, rs, nodeD, nodeL, interD=interD, interL=interL, nnodes=nnodes,
                    connector=connector, verbose=verbose)
    if z0 is None:
        model.setIinj(Iinj, config)
    else:
        model.setVext(Vexts)
    tstart = time.time()
    t[k], stimon[k], _, Vmeffprobes[k], *_ = model.simulate(tstim, toffset, PRF, DC, dt, atol)
    tcomp[k] = time.time() - tstart

    print('----------------------------- output metrics ------------------------------------')

    # Compute action potentials depolarization amplitudes
    dV_amps = {k: np.mean(np.ptp(val, axis=1)) for k, val in Vmeffprobes.items()}
    print('average depolarization amplitude')
    for k, val in dV_amps.items():
        print('dV_{} = {:.2f} mV'.format(k, val))

    # Compute conduction velocity distributions
    spiketimings = {k: getSpikesTimings(t[k], Vmeffprobes[k]) for k in keys}  # ms
    vconds = {
        k: getConductionSpeeds(xcoords, spiketimings[k]) if spiketimings[k] is not None else np.nan
        for k in keys
    }  # m/s
    print('conduction speed range:')
    for k, val in vconds.items():
        print('vcond_{} = {:.2f} - {:.2f} m/s'.format(k, val.min(), val.max()))


    # Correct states vectors onset and rescale time vectors to ms units
    for k in keys:
        stimon[k][0] = 1
        t[k] *= 1e3

    # Create comparative figure
    fig = plt.figure(figsize=(12, 3))
    gs = gridspec.GridSpec(1, 4, width_ratios=[4, 0.2, 1, 1], wspace=0.3)
    axes = list(map(plt.subplot, gs))
    fig.subplots_adjust(top=0.8, left=0.05, right=0.95)
    fig.suptitle('{} - {}'.format(
        model.pprint(),
        'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))
    ), fontsize=13)
    tonset = -0.05 * (t['classic'][-1] - t['classic'][0])

    # Plot Vmeff-traces for classic and custom schemes
    ax = axes[0]
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_ylim(-100, 50)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$V_m$ (mV)', fontsize=fs)
    handles = {
        k: plotSignals(
            t[k], Vmeffprobes[k], states=stimon[k], ax=ax, onset=(tonset, neuron.Vm0),
            fs=fs, linestyle=ls[k], cmode=cmode, cmap=cmap)
        for k in keys
    }
    ax.legend([handles[k][-1] for k in keys], keys, fontsize=fs, frameon=False)

    # Plot node index reference
    cbar_ax = axes[1]
    bounds = np.arange(nnodes + 1) + 1
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    mpl.colorbar.ColorbarBase(
        cbar_ax,
        cmap=cmap,
        norm=norm,
        spacing='proportional',
        ticks=bounds[:-1] + 0.5,
        ticklocation='left',
        boundaries=bounds,
        format='%1i'
    )
    cbar_ax.tick_params(axis='both', which='both', length=0)
    cbar_ax.set_title('node index', size=fs)

    # Plot conduction speeds distribution
    vconds1D = {k: np.squeeze(np.reshape(val, (val.size, 1))) for k, val in vconds.items()}
    ax = axes[2]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)
    ax.set_title('conduction speed (m/s)', fontsize=fs)
    ax.set_xticks(indices)
    ax.set_xticklabels(keys)
    violin_parts = ax.violinplot(vconds1D.values(), indices, points=60, widths=0.5, showextrema=False)
    for c, pc in zip(colors.values(), violin_parts['bodies']):
        pc.set_facecolor(c)
        pc.set_edgecolor('k')
    for idx, data in zip(indices, vconds1D.values()):
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        wmin = np.clip(q1 - (q3 - q1) * 1.5, min(data), q1)
        wmax = np.clip(q3 + (q3 - q1) * 1.5, q3, max(data))
        ax.scatter(idx, med, marker='o', color='k', s=30, zorder=3)
        ax.vlines(idx, q1, q3, color='grey', linestyle='-', lw=5)
        ax.vlines(idx, wmin, wmax, color='grey', linestyle='-', lw=1)
        ax.text(idx, 0.5 * data.min(), '{:.1f} m/s'.format(med), horizontalalignment='center')
    ax.set_ylim(0, 300.)

    # Plot computation times
    ax = axes[3]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)
    ax.set_title('comp. time (s)', fontsize=fs)
    ax.set_xticks(indices)
    ax.set_xticklabels(keys)
    for i, k in enumerate(keys):
        ax.bar(indices[i], tcomp[k], align='center', color=colors[k])
        ax.text(indices[i], 1.5 * tcomp[k], '{}s'.format(si_format(tcomp[k], 2, space=' ')),
                horizontalalignment='center')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e2)

    # fig.tight_layout()

    return fig


def runPlotAStim(neuron, a, Fdrive, rs, connector, nodeD, nodeL, interD, interL, Adrive, tstim, toffset,
                 PRF, DC, nnodes=None, dt=None, atol=None, cmode='qual', verbose=False, config=None):
    ''' Run A-STIM simulation of 1D extended SONIC model and plot results. '''

    # Plot parameters
    fs = 10
    cmap = plt.cm.jet_r

    tstart = time.time()

    # Create extended SONIC model with specific US frequency and connection scheme
    model = Sonic1D(neuron, rs, nodeD, nodeL, interD=interD, interL=interL, nnodes=nnodes,
                    a=a, Fdrive=Fdrive, connector=connector, verbose=verbose)

    # Set node-specifc acoustic amplitudes
    if Adrive is None:
        Adrive = model.titrateUS(tstim, toffset, PRF, DC, dt, atol, config)

    lbls = model.setUSdrive(Adrive, config)

    # Run model simulation
    t, stimon, Qprobes, Vmeffprobes, _ = model.simulate(tstim, toffset, PRF, DC, dt, atol)
    tcomp = time.time() - tstart

    # Rescale vectors to appropriate units
    t *= 1e3  # ms
    Qprobes *= 1e5  # nC/cm2

    # Create figure
    fig = plt.figure(figsize=(12, 3))
    wratios = [4, 4, 0.2, 1] if cmode == 'seq' else [4, 4, 1]
    gs = gridspec.GridSpec(1, len(wratios), width_ratios=wratios, wspace=0.3)
    axes = list(map(plt.subplot, gs))
    fig.subplots_adjust(top=0.8, left=0.05, right=0.95)
    fig.suptitle('{} - {}'.format(
        model.pprint(),
        'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))
    ), fontsize=13)

    # for ax in axes[:2]:
    #     ax.set_ylim(-150, 50)
    tonset = -0.05 * (t[-1] - t[0])
    Vm0 = model.neuron.Vm0

    # Plot charge density profiles
    ax = axes[0]
    ax.set_title('membrane charge density', fontsize=fs)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$\\rm Qm\ (nC/cm^2)$', fontsize=fs)
    plotSignals(t, Qprobes, states=stimon, ax=ax, onset=(tonset, Vm0 * neuron.Cm0 * 1e2),
                fs=fs, cmode=cmode, cmap=cmap)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)

    # Plot effective potential profiles
    if cmode == 'seq':
        lbls = None
    ax = axes[1]
    ax.set_title('effective membrane potential', fontsize=fs)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$\\rm V_{m,eff}\ (mV)$', fontsize=fs)
    plotSignals(t, Vmeffprobes, states=stimon, ax=ax, onset=(tonset, Vm0),
                fs=fs, cmode=cmode, cmap=cmap, lbls=lbls)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)

    # Plot node index reference
    if cmode == 'seq':
        cbar_ax = axes[2]
        bounds = np.arange(nnodes + 1) + 1
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        mpl.colorbar.ColorbarBase(
            cbar_ax,
            cmap=cmap,
            norm=norm,
            spacing='proportional',
            ticks=bounds[:-1] + 0.5,
            ticklocation='left',
            boundaries=bounds,
            format='%1i'
        )
        cbar_ax.tick_params(axis='both', which='both', length=0)
        cbar_ax.set_title('node index', size=fs)
        iax = 3
    else:
        iax = 2

    # Plot computation time
    ax = axes[iax]
    ax.set_title('comp. time (s)', fontsize=fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)
    ax.set_xticks([])
    ax.bar(1, tcomp, align='center', color='dimgrey')
    ax.text(1, 1.5 * tcomp, '{}s'.format(si_format(tcomp, 2, space=' ')),
            horizontalalignment='center')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e2)

    return fig

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-27 16:41:08
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-30 14:21:07

import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PySONIC.utils import logger, si_format, pow10_format
from PySONIC.solvers import findPeaks, SolverUS, xlslog
from PySONIC.constants import *
from PySONIC.plt import getPatchesLoc

from .sonic0D import Sonic0D


def runPlotEStim(neuron, Astim, tstim, toffset, PRF, DC, dt=None, atol=None):
    ''' Run E-STIM simulation of point-neuron SONIC model and plot results. '''

    # Create NEURON point-neuron SONIC model and run simulation
    tstart = time.time()
    model = Sonic0D(neuron)
    model.setAstim(Astim)
    t, y, stimon = model.simulate(tstim, toffset, PRF, DC, dt, atol)
    tcomp = time.time() - tstart
    print('Simulation completed in {:.2f} ms'.format(tcomp * 1e3))
    Vm = y[0, :]

    # Rescale vectors to appropriate units
    t *= 1e3

    # Get pulses timing
    npatches, tpatch_on, tpatch_off = getPatchesLoc(t, stimon)

    # Add onset to signals
    t0 = -10.0
    t = np.hstack((np.array([t0, 0.]), t))
    Vm = np.hstack((np.ones(2) * neuron.Vm0, Vm))

    # Create figure and plot membrane potential profile
    fs = 10
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)
    fig.subplots_adjust(top=0.8)
    fig.suptitle('{} point-neuron, A = {}A/m2, {}s'.format(
        neuron.name, *si_format([Astim * 1e-3, tstim], space=' '),
        'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))),
        fontsize=18)
    ax.plot(t, Vm)
    for i in range(npatches):
        ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                   facecolor='#8A8A8A', alpha=0.2)
    ax.set_xlim(t0, (tstim + toffset) * 1e3)
    ax.set_ylim(-150, 70)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$V_{m, eff}$ (mV)', fontsize=fs)
    ax.set_title('membrane potential', fontsize=fs + 2)

    return fig


def compareAStim(neuron, a, Fdrive, Adrive, tstim, toffset, PRF, DC, dt=None, atol=None):
    ''' Compare results of NEURON and Python based A-STIM simulations of the point-neuron
        SONIC model.
    '''

    # Create NEURON point-neuron SONIC model and run simulation
    tstart = time.time()
    model = Sonic0D(neuron, a=a, Fdrive=Fdrive)
    model.setAdrive(Adrive * 1e-3)
    t_NEURON, y_NEURON, stimon_NEURON = model.simulate(tstim, toffset, PRF, DC, dt, atol)
    tcomp_NEURON = time.time() - tstart
    Qm_NEURON, Vm_NEURON = y_NEURON[0:2, :]

    # Run Python stimulation
    tstart = time.time()
    t_Python, y_Python, stimon_Python = SolverUS(a, neuron, Fdrive).run(neuron, Fdrive, Adrive,
                                                                        tstim, toffset, PRF, DC)
    tcomp_Python = time.time() - tstart
    Qm_Python, Vm_Python = y_Python[2:4, :]

    # Rescale vectors to appropriate units
    t_Python, t_NEURON = [t * 1e3 for t in [t_Python, t_NEURON]]
    Qm_Python, Qm_NEURON = [Qm * 1e5 for Qm in [Qm_Python, Qm_NEURON]]

    # Get pulses timing
    npatches, tpatch_on, tpatch_off = getPatchesLoc(t_Python, stimon_Python)

    # Add onset to signals
    t0 = -10.0
    y0 = neuron.Vm0
    t_Python, t_NEURON = [np.hstack((np.array([t0, 0.]), t)) for t in [t_Python, t_NEURON]]
    Qm_Python, Qm_NEURON = [np.hstack((np.ones(2) * y0, Qm)) for Qm in [Qm_Python, Qm_NEURON]]
    Vm_Python, Vm_NEURON = [np.hstack((np.ones(2) * y0, Vm)) for Vm in [Vm_Python, Vm_NEURON]]

    # Create comparative figure
    fs = 10
    fig = plt.figure(figsize=(16, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1])
    axes = list(map(plt.subplot, gs))
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for item in ax.get_xticklabels() + ax.get_xticklabels():
            item.set_fontsize(fs)
    fig.subplots_adjust(top=0.8, left=0.05, right=0.95)
    fig.suptitle('{} point-neuron, a = {}m, f = {}Hz, A = {}Pa, {}s'
                 .format(neuron.name, *si_format([a, Fdrive, Adrive, tstim], space=' '),
                         'adaptive time step' if dt is None else 'dt = ${}$ ms'
                         .format(pow10_format(dt * 1e3))),
                 fontsize=18)

    # Plot charge density profiles
    ax = axes[0]
    ax.plot(t_Python, Qm_Python, label='Python')
    ax.plot(t_NEURON, Qm_NEURON, label='NEURON')
    for i in range(npatches):
        ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                   facecolor='#8A8A8A', alpha=0.2)
    ax.legend(fontsize=fs, frameon=False)
    ax.set_xlim(t0, (tstim + toffset) * 1e3)
    ax.set_ylim(-100, 50)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('Qm (nC/cm2)', fontsize=fs)
    ax.set_title('membrane charge density', fontsize=fs + 2)

    # Plot effective potential profiles
    ax = axes[1]
    ax.plot(t_Python, Vm_Python, label='Python')
    ax.plot(t_NEURON, Vm_NEURON, label='NEURON')
    for i in range(npatches):
        ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                   facecolor='#8A8A8A', alpha=0.2)
    ax.legend(fontsize=fs, frameon=False)
    ax.set_xlim(t0, (tstim + toffset) * 1e3)
    ax.set_ylim(-150, 70)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$V_{m, eff}$ (mV)', fontsize=fs)
    ax.set_title('membrane potential', fontsize=fs + 2)

    # Plot comparative time histogram
    ax = axes[2]
    ax.set_ylabel('comp. time (s)', fontsize=fs)
    indices = [1, 2]
    tcomps = [tcomp_Python, tcomp_NEURON]
    ax.set_xticks(indices)
    ax.set_xticklabels(['Python', 'NEURON'])
    for idx, tcomp in zip(indices, tcomps):
        ax.bar(idx, tcomp, align='center')
        ax.text(idx, 1.5 * tcomp, '{}s'.format(si_format(tcomp, 2, space=' ')),
                horizontalalignment='center')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e2)

    return fig



class EStimWorker():
    ''' Worker class that runs a single E-STIM simulation a given neuron for specific
        stimulation parameters, and save the results in a PKL file. '''

    def __init__(self, wid, batch_dir, log_filepath, neuron, Astim, tstim, toffset,
                 PRF, DC, dt=None, atol=None, nsims=1):
        ''' Class constructor.

            :param wid: worker ID
            :param neuron: neuron object
            :param Astim: electrical stimulus amplitude (mA/m2)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param nsims: total number or simulations
        '''

        self.id = wid
        self.batch_dir = batch_dir
        self.log_filepath = log_filepath
        self.neuron = neuron
        self.Astim = Astim
        self.tstim = tstim
        self.toffset = toffset
        self.PRF = PRF
        self.DC = DC
        self.dt = dt
        self.atol = atol
        self.nsims = nsims

    def __call__(self):
        ''' Method that runs the simulation. '''

        # Determine simulation code
        simcode = 'ESTIM_{}_{}_{:.1f}mA_per_m2_{:.0f}ms_{}NEURON'.format(
            self.neuron.name,
            'CW' if self.DC == 1 else 'PW',
            self.Astim,
            self.tstim * 1e3,
            'PRF{:.2f}Hz_DC{:.2f}%_'.format(self.PRF, self.DC * 1e2) if self.DC < 1. else ''
        )

        # Get date and time info
        date_str = time.strftime("%Y.%m.%d")
        daytime_str = time.strftime("%H:%M:%S")

        # Run simulation
        tstart = time.time()
        model = Sonic0D(self.neuron)
        model.setAstim(self.Astim)
        (t, y, stimon) = model.simulate(self.tstim, self.toffset, self.PRF, self.DC,
                                        self.dt, self.atol)
        Qm, Vm, *states = y
        tcomp = time.time() - tstart
        logger.debug('completed in %ss', si_format(tcomp, 2))

        # Store dataframe and metadata
        df = pd.DataFrame({'t': t, 'states': stimon, 'Qm': Qm, 'Vm': Vm})
        for j in range(len(self.neuron.states_names)):
            df[self.neuron.states_names[j]] = states[j]
        meta = {'neuron': self.neuron.name, 'Astim': self.Astim, 'phi': np.pi,
                'tstim': self.tstim, 'toffset': self.toffset, 'PRF': self.PRF, 'DC': self.DC,
                'tcomp': tcomp}
        if self.dt is not None:
            meta['dt'] = self.dt
        if self.atol is not None:
            meta['atol'] = self.atol

        # Export into to PKL file
        output_filepath = '{}/{}.pkl'.format(self.batch_dir, simcode)
        with open(output_filepath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': df}, fh)
        logger.debug('simulation data exported to "%s"', output_filepath)

        # Detect spikes on Qm signal
        dt = t[1] - t[0]
        ipeaks, *_ = findPeaks(Qm, SPIKE_MIN_QAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                               SPIKE_MIN_QPROM)
        n_spikes = ipeaks.size
        lat = t[ipeaks[0]] if n_spikes > 0 else 'N/A'
        sr = np.mean(1 / np.diff(t[ipeaks])) if n_spikes > 1 else 'N/A'
        logger.debug('%u spike%s detected', n_spikes, "s" if n_spikes > 1 else "")

        # Export key metrics to log file
        log = {
            'A': date_str,
            'B': daytime_str,
            'C': self.neuron.name,
            'D': self.Astim,
            'E': self.tstim * 1e3,
            'F': self.PRF * 1e-3 if self.DC < 1 else 'N/A',
            'G': self.DC,
            'H': t.size,
            'I': round(tcomp, 4),
            'J': n_spikes,
            'K': lat * 1e3 if isinstance(lat, float) else 'N/A',
            'L': sr * 1e-3 if isinstance(sr, float) else 'N/A'
        }

        if xlslog(self.log_filepath, 'Data', log) == 1:
            logger.debug('log exported to "%s"', self.log_filepath)
        else:
            logger.error('log export to "%s" aborted', self.log_filepath)

        return output_filepath

    def __str__(self):
        worker_str = 'E-STIM {} simulation {}/{}: {} neuron, A = {}A/m2, t = {}s'\
            .format('NEURON', self.id, self.nsims, self.neuron.name,
                    si_format(self.Astim * 1e-3, 2, space=' '),
                    si_format(self.tstim, 1, space=' '))
        if self.DC < 1.0:
            worker_str += ', PRF = {}Hz, DC = {:.2f}%'\
                .format(si_format(self.PRF, 2, space=' '), self.DC * 1e2)
        return worker_str



class AStimWorker():
    ''' Worker class that runs a single A-STIM simulation a given neuron for specific
        stimulation parameters, and save the results in a PKL file. '''

    def __init__(self, wid, batch_dir, log_filepath, neuron, a, Fdrive, Adrive, tstim, toffset,
                 PRF, DC, dt=None, atol=None, nsims=1):
        ''' Class constructor.

            :param wid: worker ID
            :param neuron: neuron object
            :param a: sonophore diameter (m)
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param nsims: total number or simulations
        '''

        self.id = wid
        self.batch_dir = batch_dir
        self.log_filepath = log_filepath
        self.neuron = neuron
        self.a = a
        self.Fdrive = Fdrive
        self.Adrive = Adrive
        self.tstim = tstim
        self.toffset = toffset
        self.PRF = PRF
        self.DC = DC
        self.dt = dt
        self.atol = atol
        self.nsims = nsims

    def __call__(self):
        ''' Method that runs the simulation. '''

        # Determine simulation code
        simcode = 'ASTIM_{}_{}_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.0f}ms_{}NEURON'.format(
            self.neuron.name,
            'CW' if self.DC == 1 else 'PW',
            self.a * 1e9,
            self.Fdrive * 1e-3,
            self.Adrive * 1e-3,
            self.tstim * 1e3,
            'PRF{:.2f}Hz_DC{:.2f}%_'.format(self.PRF, self.DC * 1e2) if self.DC < 1. else ''
        )

        # Get date and time info
        date_str = time.strftime("%Y.%m.%d")
        daytime_str = time.strftime("%H:%M:%S")

        # Run simulation
        tstart = time.time()
        model = Sonic0D(self.neuron, a=self.a, Fdrive=self.Fdrive)
        model.setAdrive(self.Adrive * 1e-3)
        (t, y, stimon) = model.simulate(self.tstim, self.toffset, self.PRF, self.DC,
                                        self.dt, self.atol)
        Qm, Vm, *states = y
        tcomp = time.time() - tstart
        logger.debug('completed in %ss', si_format(tcomp, 2))

        # Store dataframe and metadata
        df = pd.DataFrame({'t': t, 'states': stimon, 'Qm': Qm, 'Vm': Vm})
        for j in range(len(self.neuron.states_names)):
            df[self.neuron.states_names[j]] = states[j]
        meta = {'neuron': self.neuron.name, 'a': self.a,
                'Fdrive': self.Fdrive, 'Adrive': self.Adrive, 'phi': np.pi,
                'tstim': self.tstim, 'toffset': self.toffset, 'PRF': self.PRF, 'DC': self.DC,
                'tcomp': tcomp}
        if self.dt is not None:
            meta['dt'] = self.dt
        if self.atol is not None:
            meta['atol'] = self.atol

        # Export into to PKL file
        output_filepath = '{}/{}.pkl'.format(self.batch_dir, simcode)
        with open(output_filepath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': df}, fh)
        logger.debug('simulation data exported to "%s"', output_filepath)

        # Detect spikes on Qm signal
        dt = t[1] - t[0]
        ipeaks, *_ = findPeaks(Qm, SPIKE_MIN_QAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                               SPIKE_MIN_QPROM)
        n_spikes = ipeaks.size
        lat = t[ipeaks[0]] if n_spikes > 0 else 'N/A'
        sr = np.mean(1 / np.diff(t[ipeaks])) if n_spikes > 1 else 'N/A'
        logger.debug('%u spike%s detected', n_spikes, "s" if n_spikes > 1 else "")

        # Export key metrics to log file
        log = {
            'A': date_str,
            'B': daytime_str,
            'C': self.neuron.name,
            'D': self.a * 1e9,
            'E': 0.0,
            'F': self.Fdrive * 1e-3,
            'G': self.Adrive * 1e-3,
            'H': self.tstim * 1e3,
            'I': self.PRF * 1e-3 if self.DC < 1 else 'N/A',
            'J': self.DC,
            'K': 'NEURON',
            'L': t.size,
            'M': round(tcomp, 4),
            'N': n_spikes,
            'O': lat * 1e3 if isinstance(lat, float) else 'N/A',
            'P': sr * 1e-3 if isinstance(sr, float) else 'N/A'
        }

        if xlslog(self.log_filepath, 'Data', log) == 1:
            logger.debug('log exported to "%s"', self.log_filepath)
        else:
            logger.error('log export to "%s" aborted', self.log_filepath)

        return output_filepath

    def __str__(self):
        worker_str = 'A-STIM {} simulation {}/{}: {} neuron, a = {}m, f = {}Hz, A = {}Pa, t = {}s'\
            .format('NEURON', self.id, self.nsims, self.neuron.name,
                    *si_format([self.a, self.Fdrive], 1, space=' '),
                    si_format(self.Adrive, 2, space=' '), si_format(self.tstim, 1, space=' '))
        if self.DC < 1.0:
            worker_str += ', PRF = {}Hz, DC = {:.2f}%'\
                .format(si_format(self.PRF, 2, space=' '), self.DC * 1e2)
        return worker_str

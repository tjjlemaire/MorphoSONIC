# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-08-15 15:08:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-27 22:04:25


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neuron import h

from PySONIC.neurons import *
from PySONIC.solvers import SolverUS
from PySONIC.utils import getLookups2D, si_format, pow10_format
from PySONIC.plt import getPatchesLoc

from pyhoc import *


class Sonic0D:
    ''' Point-neuron SONIC model in NEURON. '''

    def __init__(self, neuron, a=32e-9, Fdrive=500e3, verbose=False):
        ''' Initialization.

            :param neuron: neuron object
            :param a: sonophore diameter (nm)
            :param Fdrive: ultrasound frequency (Hz)
            :param Ra: cytoplasmic resistivity (Ohm.cm)
            :param diam: section diameter (m)
            :param L: section length (m)
            :param verbose: boolean stating whether to print out details
        '''
        self.neuron = neuron
        self.a = a  # m
        self.Fdrive = Fdrive  # Hz
        self.verbose = verbose
        self.mechname = neuron.name


        # Load mechanisms DLL file
        nmodl_dir = os.path.join(os.getcwd(), 'nmodl')
        mod_file = nmodl_dir + '/{}.mod'.format(self.mechname)
        dll_file = nmodl_dir + '/nrnmech.dll'
        if not os.path.isfile(dll_file) or os.path.getmtime(mod_file) > os.path.getmtime(dll_file):
            raise Warning('"{}.mod" file more recent than compiled dll'.format(self.mechname))
        if not isAlreadyLoaded(dll_file):
            h.nrn_load_dll(dll_file)

        # Load and set fnction tables of membrane mechanism
        self.setFuncTables(self.a, self.Fdrive)

        # Create section and set geometry
        self.section = self.createSection('point')

        # Set section membrane mechanism
        self.defineBiophysics()

        if self.verbose:
            print('Creating model: {}'.format(self))

    def __str__(self):
        ''' Explicit naming of the model instance. '''
        return 'SONIC0D_{}_{}m_{}Hz'.format(self.neuron.name, *si_format([self.a, self.Fdrive], 2))

    def createSection(self, id):
        ''' Create morphological section.

            :param id: name of the section.
        '''
        return h.Section(name=id, cell=self)

    def setFuncTables(self, a, Fdrive):
        ''' Set neuron-specific, sonophore diameter and US frequency dependent 2D interpolation tables
            in the (amplitude, charge) space, and link them to FUNCTION_TABLEs in the MOD file of the
            corresponding membrane mechanism.

            :param a: sonophore diameter (m)
            :param Fdrive: US frequency (Hz)
        '''

        # Get lookups
        Aref, Qref, lookups2D = getLookups2D(self.mechname, a, Fdrive)

        # Rescale rate constants to ms-1
        for k in lookups2D.keys():
            if 'alpha' in k or 'beta' in k:
                lookups2D[k] *= 1e-3

        # Convert lookups independent variables to hoc vectors
        self.Aref = h.Vector(Aref * 1e-3)  # kPa
        self.Qref = h.Vector(Qref * 1e5)  # nC/cm2

        # Convert lookups dependent variables to hoc matrices
        self.lookups2D = {key: array2Matrix(value) for key, value in lookups2D.items()}

        # Assign hoc lookups to as interpolation tables in membrane mechanism
        setFuncTable(self.mechname, 'V', self.lookups2D['V'], self.Aref, self.Qref)
        for gate in self.neuron.getGates():
            gate = gate.lower()
            for rate in ['alpha', 'beta']:
                rname = '{}{}'.format(rate, gate)
                setFuncTable(self.mechname, rname, self.lookups2D[rname], self.Aref, self.Qref)

    def defineBiophysics(self, sections=None):
        ''' Set neuron-specific active membrane properties. '''
        sections = [self.section] if sections is None else sections
        for sec in sections:
            sec.insert(self.mechname)

    def setAdrive(self, Adrive):
        ''' Set US stimulation amplitude (and set modality to "US").

            :param Adrive: acoustic pressure amplitude (Pa)
        '''
        if self.verbose:
            print('Setting acoustic stimulus amplitude: Adrive = {}Pa'
                  .format(si_format(Adrive * 1e3, space=' ')))
        setattr(self.section, 'Adrive_{}'.format(self.mechname), Adrive)
        self.modality = 'US'

    def setAstim(self, Astim):
        ''' Set electrical stimulation amplitude (and set modality to "elec")

            :param Astim: injected current density (mA/m2).
        '''
        self.Iinj = Astim * self.section(0.5).area() * 1e-6  # nA
        if self.verbose:
            print('Setting electrical stimulus amplitude: Iinj = {}A'
                  .format(si_format(self.Iinj * 1e-9, 2, space=' ')))
        self.iclamp = h.IClamp(self.section(0.5))
        self.iclamp.delay = 0  # we want to exert control over amp starting at 0 ms
        self.iclamp.dur = 1e9  # dur must be long enough to span all our changes
        self.modality = 'elec'

    def setStimON(self, value):
        ''' Set US or electrical stimulation ON or OFF by updating the appropriate
            mechanism/object parameter.

            :param value: new stimulation state (0 = OFF, 1 = ON)
            :return: new stimulation state
        '''
        setattr(self.section, 'stimon_{}'.format(self.mechname), value)
        if self.modality == 'elec':
            self.iclamp.amp = value * self.Iinj
        return value

    def toggleStim(self):
        ''' Toggle US or electrical stimulation and set appropriate next toggle event. '''
        # OFF -> ON at pulse onset
        if self.stimon == 0:
            # print('t = {:.2f} ms: switching stim ON and setting next OFF event at {:.2f} ms'
            #       .format(h.t, min(self.tstim, h.t + self.Ton)))
            self.stimon = self.setStimON(1)
            self.cvode.event(min(self.tstim, h.t + self.Ton), self.toggleStim)
        # ON -> OFF at pulse offset
        else:
            self.stimon = self.setStimON(0)
            if (h.t + self.Toff) < self.tstim - h.dt:
                # print('t = {:.2f} ms: switching stim OFF and setting next ON event at {:.2f} ms'
                #       .format(h.t, h.t + self.Toff))
                self.cvode.event(h.t + self.Toff, self.toggleStim)
            # else:
            #     print('t = {:.2f} ms: switching stim OFF'.format(h.t))

        # Re-initialize cvode if active
        if self.cvode.active():
            self.cvode.re_init()
        else:
            h.fcurrent()

    def integrate(self, tstop, tstim, PRF, DC, dt, atol):
        ''' Integrate the model differential variables for a given duration, while updating the
            value of the boolean parameter stimon during ON and OFF periods throughout the numerical
            integration, according to stimulus parameters.

            Integration uses an adaptive time step method by default.

            :param tstop: duration of numerical integration (s)
            :param tstim: stimulus duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle
            :param dt: integration time step (s). If provided, the fixed time step method is used.
            :param atol: absolute error tolerance (default = 1e-3). If provided, the adaptive
                time step method is used.
        '''
        # Convert input parameters to NEURON units
        tstim *= 1e3
        tstop *= 1e3
        PRF /= 1e3
        if dt is not None:
            dt *= 1e3

        # Update PRF for CW stimuli to optimize integration
        if DC == 1.0:
            PRF = 1 / tstim

        # Set pulsing parameters used in CVODE events
        self.Ton = DC / PRF
        self.Toff = (1 - DC) / PRF
        self.tstim = tstim

        # Set integration parameters
        h.secondorder = 2
        self.cvode = h.CVode()
        if dt is not None:
            h.dt = dt
            self.cvode.active(0)
            print('fixed time step integration (dt = {} ms)'.format(h.dt))
        else:
            self.cvode.active(1)
            if atol is not None:
                self.cvode.atol(atol)
                print('adaptive time step integration (atol = {})'.format(self.cvode.atol()))

        # Initialize
        h.finitialize(self.neuron.Vm0)
        self.stimon = self.setStimON(1)
        self.cvode.event(self.Ton, self.toggleStim)

        # Integrate
        while h.t < tstop:
            h.fadvance()

        return 0

    def simulate(self, tstim, toffset, PRF, DC, dt, atol):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param tstim: stimulus duration (s)
            :param toffset: stimulus offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method (default = 1e-3)
        '''
        # Set recording vectors
        tprobe = setTimeProbe()
        stimprobe = setStimProbe(self.section, self.mechname)
        vprobe = setRangeProbe(self.section, 'v')
        Vmeffprobe = setRangeProbe(self.section, 'Vmeff_{}'.format(self.mechname))
        statesprobes = [setRangeProbe(self.section, '{}_{}'.format(alias(state), self.mechname))
                        for state in self.neuron.states_names]

        # Integrate model
        self.integrate(tstim + toffset, tstim, PRF, DC, dt, atol)

        # Retrieve output variables
        t = Vec2array(tprobe) * 1e-3  # s
        stimon = Vec2array(stimprobe)
        vprobe = Vec2array(vprobe)  # mV or nC/cm2
        if self.modality == 'US':
            vprobe *= 1e-5  # C/m2
        Vmeffprobe = Vec2array(Vmeffprobe)  # mV
        statesprobes = list(map(Vec2array, statesprobes))
        y = np.vstack([vprobe, Vmeffprobe, np.array(statesprobes)])

        # return output variables
        return (t, y, stimon)



def runAStim(neuron, a, Fdrive, Adrive, tstim, toffset, PRF, DC, dt=None, atol=None):
    ''' Create NEURON point-neuron SONIC model and run acoustic simulation. '''
    model = Sonic0D(neuron, a=a, Fdrive=Fdrive)
    model.setAdrive(Adrive * 1e-3)
    return model.simulate(tstim, toffset, PRF, DC, dt, atol)


def runEStim(neuron, Astim, tstim, toffset, PRF, DC, dt=None, atol=None):
    ''' Create NEURON point-neuron SONIC model and run electrical simulation. '''
    model = Sonic0D(neuron, a=a, Fdrive=Fdrive)
    model.setAstim(Astim)
    return model.simulate(tstim, toffset, PRF, DC, dt, atol)


def runPlotEStim(neuron, Astim, tstim, toffset, PRF, DC, dt=None, atol=None):
    ''' Plot results of NEURON point-neuron SONIC model electrical simulation. '''

    # Create NEURON point-neuron SONIC model and run simulation
    tstart = time.time()
    t, y, stimon = runEStim(neuron, Astim, tstim, toffset, PRF, DC, dt, atol)
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
    ''' Compare NEURON and Python based simulations of the point-neuron SONIC model. '''

    # Create NEURON point-neuron SONIC model and run simulation
    tstart = time.time()
    t_NEURON, y_NEURON, stimon_NEURON = runAStim(neuron, a, Fdrive, Adrive, tstim, toffset, PRF, DC,
                                                 dt, atol)
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

    # Plot charge profiles
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


if __name__ == '__main__':

    # Model parameters
    neuron = CorticalRS()
    a = 32e-9  # sonophore diameter

    # Stimulation parameters
    Fdrive = 500e3  # Hz
    Adrive = 50e3  # kPa
    Astim = 25.0  # mA/m2
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    PRF = 100.  # Hz
    DC = 0.7

    # fig1 = compareAStim(neuron, a, Fdrive, Adrive, tstim, toffset, PRF, DC)
    fig2 = runPlotEStim(neuron, Astim, tstim, toffset, PRF, DC)

    plt.show()

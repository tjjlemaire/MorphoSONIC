# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-06-29 18:11:24
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-07-03 12:22:36

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from PySONIC.core import EffectiveVariablesLookup
from PySONIC.threshold import threshold
from PySONIC.utils import logger, timer, isWithin

from ..constants import THR_QM_DIV


class SonicBenchmark:
    ''' Interface allowing to run benchmark simulations of the SONIC paradigm
        for various multi-compartmental models, with a simplified sinusoidal
        capacitive drive.
    '''

    npc = 40  # number of samples per cycle
    varunits = {
        't': 'ms',
        'Cm': 'uF/cm2',
        'Vm': 'mV',
        'Qm': 'nC/cm2'
    }
    nodelabels = ['node 1', 'node 2']
    ga_bounds = [1e-5, 1e5]  # mS/cm2

    def __init__(self, pneuron, ga, f, rel_amps, passive=False):
        ''' Initialization.

            :param pneuron: point-neuron object
            :param ga: axial conductance (mS/cm2)
            :param f: US frequency (kHz)
            :param rel_amps: pair of relative capacitance oscillation amplitudes
        '''
        self.pneuron = pneuron
        self.states = self.pneuron.statesNames()
        self.ga = ga
        self.f = f
        self.rel_amps = rel_amps
        self.passive = passive

        # Compute effective capacitances over 1 cycle
        Cmeff = []
        self.lkps = []
        for A in self.rel_amps:
            Cm_cycle = self.capct(A, self.tcycle)    # uF/cm2
            Cmeff.append(1 / np.mean(1 / Cm_cycle))  # uF/cm2
            if not passive:
                self.lkps.append(self.getLookup(Cm_cycle))
        self.Cmeff = np.array(Cmeff)

    def __repr__(self):
        params = [
            f'ga = {self.ga:.2f} mS/cm2',
            f'f = {self.f:.0f} kHz',
            f'relative amps = {self.rel_amps}'
        ]
        dynamics = 'passive' if self.passive else 'full'
        mech = f'{dynamics} {self.pneuron.name} dynamics'
        return f'{self.__class__.__name__}({mech}, {", ".join(params)})'

    @property
    def ga(self):
        return self._ga

    @ga.setter
    def ga(self, value):
        assert isWithin('ga', value, self.ga_bounds)
        self._ga = value

    @property
    def Cm0(self):
        ''' Resting capacitance (uF/cm2). '''
        return self.pneuron.Cm0 * 1e2

    @property
    def Qref(self):
        ''' Reference charge linear space. '''
        return np.arange(*self.pneuron.Qbounds, 1e-5) * 1e5  # nC/cm2

    @property
    def gPas(self):
        ''' Passive membrane conductance (mS/cm2). '''
        return self.pneuron.gLeak * 1e-1

    @property
    def Vm0(self):
        ''' Resting membrane potential (mV). '''
        return self.pneuron.Vm0

    def capct(self, A, t):
        ''' Time-varying capacitance (in uF/cm2) '''
        return self.Cm0 * (1 + A * np.sin(2 * np.pi * self.f * t))

    def vCapct(self, t):
        ''' Vector of time-varying capacitance (in uF/cm2) '''
        return np.array([self.capct(A, t) for A in self.rel_amps])

    def getLookup(self, Cm):
        ''' Get a lookup object of effective variables for a given capacitance cycle vector. '''
        Vmarray = np.array([Q / Cm for Q in self.Qref])
        tables = {
            k: np.array([np.mean(np.vectorize(v)(Vmvec)) for Vmvec in Vmarray])
            for k, v in self.pneuron.effRates().items()
        }
        return EffectiveVariablesLookup({'Q': self.Qref}, tables)

    @property
    def tau(self):
        ''' membrane time constant (ms).

            [Cm0/gPas] = uF.cm-2 / mS.cm-2 = uF/mS = 1e-3 F/S = 1e-3 s = 1 ms
        '''
        return self.Cm0 / self.gPas

    @property
    def dt_full(self):
        ''' Full time step (ms). '''
        return 1 / (self.npc * self.f)

    @property
    def dt_sparse(self):
        ''' Sparse time step (ms). '''
        return 1 / self.f

    @property
    def tcycle(self):
        ''' Time vector over 1 acoustic cycle. '''
        return np.arange(0, 1 / self.f, self.dt_full)

    def getCmeff(self, A):
        ''' Compute effective capacitance over 1 cycle. '''
        return 1 / np.mean(1 / self.capct(A, self.tcycle))  # uF/cm2

    def iax(self, Vm, Vmother):
        ''' Axial current (in mA/cm2).

            [iax] = mS/cm2 * mV = 1e-6 SV/cm2 = 1 uA/cm2 = 1e-3 mA/cm2
        '''
        return self.ga * (Vmother - Vm) * 1e-3

    def vIax(self, Vm):
        return np.array([self.iax(*Vm), self.iax(*Vm[::-1])])  # mA/cm2

    def deserialize(self, y):
        return np.reshape(y.copy(), (2, self.npernode))

    def serialize(self, y):
        return np.reshape(y.copy(), (self.npernode * 2))

    def derivatives(self, t, y, Cm, dstates_func):
        ''' Generic derivatives method. '''
        y = self.deserialize(y)  # reshape 1D input per node
        dydt = np.empty(y.shape)
        Qm = y[:, 0]  # nC/cm2
        Vm = y[:, 0] / Cm  # mV
        states_array = y[:, 1:]
        for i, (qm, vm, states) in enumerate(zip(Qm, Vm, states_array)):
            if not self.passive:
                states_dict = dict(zip(self.states, states))
                dydt[i, 1:] = dstates_func(i, qm, vm, states_dict) * 1e-3  # ms-1
                im = self.pneuron.iNet(vm, states_dict)  # mA/m2
            else:
                im = self.pneuron.iLeak(vm)  # mA/m2
            dydt[i, 0] = -im * 1e-4  # mA/cm2
        dydt[:, 0] += self.vIax(Vm)  # mA/cm2
        dydt[:, 0] *= 1e3  # nC/cm2.ms
        return self.serialize(dydt)

    def dstatesFull(self, i, qm, vm, states):
        return self.pneuron.getDerStates(vm, states)

    def dfull(self, t, y):
        ''' Full derivatives. '''
        return self.derivatives(t, y, self.vCapct(t), self.dstatesFull)

    def dstatesEff(self, i, qm, vm, states):
        lkp0d = self.lkps[i].interpolate1D(qm)
        return np.array([self.pneuron.derEffStates()[k](lkp0d, states) for k in self.states])

    def deff(self, t, y):
        ''' Effective derivatives. '''
        return self.derivatives(t, y, self.Cmeff, self.dstatesEff)

    @property
    def Qm0(self):
        ''' Resting membrane charge density. '''
        return self.Vm0 * self.Cm0

    @property
    def y0node(self):
        if self.passive:
            return [self.Qm0]
        else:
            return [self.Qm0, *[self.pneuron.steadyStates()[k](self.Vm0) for k in self.states]]

    @property
    def y0(self):
        self.npernode = len(self.y0node)
        return self.y0node + self.y0node

    def integrate(self, dfunc, t):
        ''' Integrate over a time vector and return charge density arrays. '''
        y = odeint(dfunc, self.y0, t, tfirst=True).T
        sol = {'Qm': y[::self.npernode]}
        if not self.passive:
            for i, k in enumerate(self.states):
                sol[k] = y[i + 1::self.npernode]
        return sol

    @timer
    def simFull(self, tstop):
        ''' Simulate the full system until a specific stop time (us). '''
        t = np.arange(0, tstop, self.dt_full)
        sol = self.integrate(self.dfull, t)
        sol['Cm'] = self.vCapct(t)
        sol['Vm'] = sol['Qm'] / sol['Cm']
        return t, sol

    @timer
    def simEff(self, tstop):
        t = np.arange(0, tstop, self.dt_sparse)
        sol = self.integrate(self.deff, t)
        sol['Cm'] = np.array([np.ones(t.size) * Cmeff for Cmeff in self.Cmeff])
        sol['Vm'] = sol['Qm'] / sol['Cm']
        return t, sol

    @property
    def methods(self):
        return {'full': self.simFull, 'effective': self.simEff}

    def simulate(self, mtype, tstop):
        # Cast tstop as a multiple of acoustic period
        tstop = int(np.ceil(tstop * self.f)) / self.f
        try:
            method = self.methods[mtype]
        except KeyError:
            raise ValueError(f'"{mtype}" is not a valid method type')
        logger.debug(f'running {mtype} {tstop:.2f} ms simulation')
        output, tcomp = method(tstop)
        logger.debug(f'completed in {tcomp:.2f} s')
        return output

    def cycleAvg(self, t, sol):
        ''' Cycle-average a time vector and a solution dictionary. '''
        solavg = {}
        for k, y in sol.items():
            yavg = []
            for yy in y:
                ys = np.reshape(yy, (int(yy.shape[0] / self.npc), self.npc))
                yavg.append(np.mean(ys, axis=1))
            solavg[k] = np.array(yavg)
        tavg = t[::self.npc]  # + 0.5 / self.f
        return tavg, solavg

    def benchmarkSim(self, tstop):
        ''' Run benchmark simulations of the model. '''
        logger.info(self)
        # Simulate with full and effective systems
        t, sol = {}, {}
        for method in ['full', 'effective']:
            t[method], sol[method] = self.simulate(method, tstop)
        # Cycle average full solution
        t['cycle-avg'], sol['cycle-avg'] = self.cycleAvg(t['full'], sol['full'])
        return t, sol

    def benchmarkPlot(self, t, sol):
        ''' Plot results of benchmark simulations of the model. '''
        colors = ['C0', 'C1']
        markers = ['-', '--', '-.']
        alphas = [0.5, 1., 1.]
        naxes = 3
        if not self.passive:
            naxes += len(self.states)
        fig, axes = plt.subplots(naxes, 1, sharex=True, figsize=(10, min(3 * naxes, 10)))
        axes[0].set_title(f'{self} - {t[list(t.keys())[0]][-1]:.2f} ms simulation')
        axes[-1].set_xlabel(f'time ({self.varunits["t"]})')
        for ax, k in zip(axes, sol[list(sol.keys())[0]].keys()):
            ax.set_ylabel(f'{k} ({self.varunits.get(k, "-")})')
        for m, alpha, (key, varsdict) in zip(markers, alphas, sol.items()):
            for ax, (k, v) in zip(axes, varsdict.items()):
                for y, c, lbl in zip(v, colors, self.nodelabels):
                    ax.plot(t[key], y, m, alpha=alpha, c=c, label=f'{lbl} - {key}')
        fig.subplots_adjust(bottom=0.2)
        axes[-1].legend(
            bbox_to_anchor=(0., -0.7, 1., .1), loc='upper center',
            ncol=3, mode="expand", borderaxespad=0.)
        return fig

    def benchmark(self, *args, **kwargs):
        ''' Rune benchmark simulation and plot results. '''
        return self.benchmarkPlot(*self.benchmarkSim(*args, **kwargs))

    def divergence(self, t, sol, tobs=None):
        ''' Evaluate the divergence between the effective and full, cycle-averaged solutions
            at a specific point in time, computing per-node differences in charge density values.
        '''
        Qmobs = np.empty((2, 2))
        for i, k in enumerate(['effective', 'cycle-avg']):  # for both solution variants
            for j, Qm in enumerate(sol[k]['Qm']):  # for each node
                if tobs is not None:  # interpolate observed Qm at given time
                    Qmobs[i, j] = np.interp(tobs, t[k], Qm)
                else:  # or take last observed Qm value
                    Qmobs[i, j] = Qm[-1]
        # Compute dictionary of per-node diffeerences
        Qdiff = dict(zip(self.nodelabels, np.squeeze(np.diff(Qmobs, axis=0))))
        logger.debug(f'Qm differences per node: {Qdiff}')
        return Qdiff

    def isDivergent(self, ga, tstop, *args, **kwargs):
        ''' Function evaluating whether max abs charge divergence is above a given threshold. '''
        self.ga = ga
        t, sol = self.benchmarkSim(tstop)
        Qdiff = self.divergence(t, sol, *args, **kwargs)
        Qdiff_absmax = max(np.abs(list(Qdiff.values())))
        return Qdiff_absmax >= THR_QM_DIV

    def findThresholdAxialCoupling(self, tstop):
        ''' Find threshold ga creating a significant charge divergence. '''
        assert self.passive, 'Procedure optimized only for passive benchmark'
        return threshold(lambda ga: self.isDivergent(ga, tstop), self.ga_bounds)

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-06-29 18:11:24
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-07-02 22:59:11

import logging
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from PySONIC.core import EffectiveVariablesLookup
from PySONIC.utils import logger
from ExSONIC.core import SennFiber, UnmyelinatedFiber

logger.setLevel(logging.DEBUG)


class SonicBenchmark:
    ''' Interface allowing to run benchmark simulations of the SONIC paradigm
        for various multi-compartmental models.
    '''

    npc = 40  # number of samples per cycle
    varunits = {
        't': 'ms',
        'Cm': 'uF/cm2',
        'Vm': 'mV',
        'Qm': 'nC/cm2'
    }
    nodelabels = ['node 1', 'node 2']

    def __init__(self, pneuron, ga, f, rel_amps, passive=True):
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
        return f'{self.__class__.__name__}({self.pneuron.name}, {", ".join(params)})'

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

    def simFull(self, tstop):
        ''' Simulate the full system until a specific stop time (us). '''
        logger.debug('running full simulation')
        t = np.arange(0, tstop, self.dt_full)
        sol = self.integrate(self.dfull, t)
        sol['Cm'] = self.vCapct(t)
        sol['Vm'] = sol['Qm'] / sol['Cm']
        return t, sol

    def simEff(self, tstop):
        logger.debug('running effective simulation')
        t = np.arange(0, tstop, self.dt_sparse)
        sol = self.integrate(self.deff, t)
        sol['Cm'] = np.array([np.ones(t.size) * Cmeff for Cmeff in self.Cmeff])
        sol['Vm'] = sol['Qm'] / sol['Cm']
        return t, sol

    def simulate(self, method, tstop):
        # Cast tstop as a multiple of acoustic period
        tstop = int(np.ceil(tstop * self.f)) / self.f
        if method == 'full':
            return self.simFull(tstop)
        elif method == 'effective':
            return self.simEff(tstop)
        raise ValueError(f'"{method}" is not a valid method type')

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

    @property
    def default_tstop(self):
        ''' Default simulation time (ms). '''
        if self.passive:
            return 5 * self.tau
        else:
            return 10.

    def benchmarkSim(self):
        ''' Run benchmark simulations of the model. '''
        # Simulate with full and effective systems
        t, sol = {}, {}
        for method in ['full', 'effective']:
            t[method], sol[method] = self.simulate(method, self.default_tstop)
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


if __name__ == '__main__':

    # Fiber models
    fibers = [
        SennFiber(10e-6, 11),                   # 10 um diameter SENN fiber
        UnmyelinatedFiber(0.8e-6, fiberL=5e-3)  # 0.8 um diameter unmylinated fiber
    ]

    # Stimulation parameters
    f = 500.             # US frequency (kHz)
    rel_amps = (0.8, 0)  # relative capacitance oscillation amplitudes

    for fiber in fibers:
        Ga = 1 / fiber.R_node_to_node    # S
        Anode = fiber.nodes['node0'].Am  # cm2
        ga = Ga / Anode * 1e3            # mS/cm2
        logger.info(f'Node-to-node coupling in {fiber}: ga = {ga:.2f} mS/cm2')
        for passive in [True, False]:
            # Create SONIC benchmarker
            sb = SonicBenchmark(fiber.pneuron, ga, f, rel_amps, passive=passive)
            logger.info(f'{sb} -> tau = {sb.tau:.2f} ms')
            # Run benchmark simulations and plot results
            fig = sb.benchmarkPlot(*sb.benchmarkSim())

    plt.show()

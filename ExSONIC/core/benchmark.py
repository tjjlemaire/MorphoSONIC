# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-06-29 18:11:24
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-08-04 17:53:30

import os
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from PySONIC.core import EffectiveVariablesLookup
from PySONIC.utils import logger, timer, isWithin, bounds
from PySONIC.plt import XYMap
from PySONIC.neurons import passiveNeuron
from PySONIC.threshold import threshold


class SonicBenchmark:
    ''' Interface allowing to run benchmark simulations of the SONIC paradigm
        for various multi-compartmental models, with a simplified sinusoidal
        capacitive drive.
    '''

    npc = 1000  # number of samples per cycle
    varunits = {
        't': 'ms',
        'Cm': 'uF/cm2',
        'Vm': 'mV',
        'Qm': 'nC/cm2'
    }
    nodelabels = ['node 1', 'node 2']
    ga_bounds = [1e-10, 1e10]  # mS/cm2

    def __init__(self, pneuron, ga, f, gamma_pair, passive=False):
        ''' Initialization.

            :param pneuron: point-neuron object
            :param ga: axial conductance (mS/cm2)
            :param f: US frequency (kHz)
            :param gamma_pair: pair of relative capacitance oscillation amplitudes
        '''
        self.pneuron = pneuron
        self.ga = ga
        self.f = f
        self.gamma_pair = gamma_pair
        self.passive = passive
        self.computeLookups()

    def copy(self):
        return self.__class__(self.pneuron, self.ga, self.f, self.gamma_pair, passive=self.passive)

    @property
    def strAmps(self):
        s = ', '.join([f'{x:.2f}' for x in self.gamma_pair])
        return f'({s})'

    def __repr__(self):
        params = [
            f'ga = {self.ga:.2e} mS/cm2',
            f'f = {self.f:.0f} kHz',
            f'rel A_Cm = {self.strAmps}'
        ]
        dynamics = 'passive ' if self.passive else ''
        mech = f'{dynamics}{self.pneuron.name} dynamics'
        return f'{self.__class__.__name__}({mech}, {", ".join(params)})'

    @property
    def pneuron(self):
        return self._pneuron

    @pneuron.setter
    def pneuron(self, value):
        self._pneuron = value.copy()
        self.states = self._pneuron.statesNames()
        if hasattr(self, 'lkps'):
            self.computeLookups()

    def isPassive(self):
        return self.pneuron.name.startswith('pas_')

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self._f = value
        if hasattr(self, 'lkps'):
            self.computeLookups()

    @property
    def gamma_pair(self):
        return self._gamma_pair

    @gamma_pair.setter
    def gamma_pair(self, value):
        self._gamma_pair = value
        if hasattr(self, 'lkps'):
            self.computeLookups()

    @property
    def gamma_str(self):
        return [f'{x:.2f}' for x in self.gamma_pair]

    @property
    def passive(self):
        return self._passive

    @passive.setter
    def passive(self, value):
        assert isinstance(value, bool), 'passive must be boolean typed'
        self._passive = value
        if hasattr(self, 'lkps'):
            self.computeLookups()

    @property
    def ga(self):
        return self._ga

    @ga.setter
    def ga(self, value):
        if value != 0.:
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
    def Cmeff(self):
        ''' Analytical solution for effective membrane capacitance (uF/cm2). '''
        return np.sqrt(1 - np.array(self.gamma_pair)**2)

    @property
    def Qminf_SONIC(self):
        ''' Analytical solution for steady-state charge density (nC/cm2). '''
        return self.pneuron.ELeak * self.Cmeff

    @property
    def Vm0(self):
        ''' Resting membrane potential (mV). '''
        return self.pneuron.Vm0

    def capct(self, gamma, t):
        ''' Time-varying capacitance (in uF/cm2) '''
        return self.Cm0 * (1 + gamma * np.sin(2 * np.pi * self.f * t))

    def vCapct(self, t):
        ''' Vector of time-varying capacitance (in uF/cm2) '''
        return np.array([self.capct(gamma, t) for gamma in self.gamma_pair])

    def getLookup(self, Cm):
        ''' Get a lookup object of effective variables for a given capacitance cycle vector. '''
        Vmarray = np.array([Q / Cm for Q in self.Qref])
        tables = {
            k: np.array([np.mean(np.vectorize(v)(Vmvec)) for Vmvec in Vmarray])
            for k, v in self.pneuron.effRates().items()
        }
        return EffectiveVariablesLookup({'Q': self.Qref}, tables)

    def computeLookups(self):
        ''' Compute benchmark lookups. '''
        self.lkps = []
        if not self.passive:
            self.lkps = [self.getLookup(Cm_cycle) for Cm_cycle in self.vCapct(self.tcycle)]

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
        return np.linspace(0, 1 / self.f, self.npc)

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

    def orderedKeys(self, varkeys):
        mainkeys = ['Qm', 'Vm', 'Cm']
        otherkeys = list(set(varkeys) - set(mainkeys))
        return mainkeys + otherkeys

    def orderedSol(self, sol):
        return {k: sol[k] for k in self.orderedKeys(sol.keys())}

    @timer
    def simFull(self, tstop):
        ''' Simulate the full system until a specific stop time (us). '''
        t = np.linspace(0, tstop, self.getNCycles(tstop) * self.npc)
        sol = self.integrate(self.dfull, t)
        sol['Cm'] = self.vCapct(t)
        sol['Vm'] = sol['Qm'] / sol['Cm']
        return t, self.orderedSol(sol)

    @timer
    def simEff(self, tstop):
        t = np.linspace(0, tstop, self.getNCycles(tstop))
        sol = self.integrate(self.deff, t)
        sol['Cm'] = np.array([np.ones(t.size) * Cmeff for Cmeff in self.Cmeff])
        sol['Vm'] = sol['Qm'] / sol['Cm']
        return t, self.orderedSol(sol)

    @property
    def methods(self):
        return {'full': self.simFull, 'effective': self.simEff}

    def getNCycles(self, tstop):
        return int(np.ceil(tstop * self.f))

    def simulate(self, mtype, tstop):
        # Cast tstop as a multiple of acoustic period
        tstop = self.getNCycles(tstop) / self.f
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

    def computeGradient(self, sol):
        ''' compute the gradient of a solution array. '''
        return {k: np.vstack((y, np.diff(y, axis=0))) for k, y in sol.items()}

    def g2tau(self, g):
        ''' Convert conductance per unit membrane area (mS/cm2) to time constant (ms). '''
        return (self.pneuron.Cm0 * 1e2) / g  # ms

    def tau2g(self, tau):
        ''' Convert time constant (ms) to conductance per unit membrane area (mS/cm2). '''
        return (self.pneuron.Cm0 * 1e2) / tau  # ms

    @property
    def taum(self):
        ''' Passive membrane time constant (ms). '''
        return self.pneuron.tau_pas * 1e3

    @taum.setter
    def taum(self, value):
        ''' Update point-neuron leakage conductance to match time new membrane time constant. '''
        if not self.isPassive():
            raise ValueError('taum can only be set for passive neurons')
        self.pneuron = passiveNeuron(
            self.pneuron.Cm0,
            self.tau2g(value) * 1e1,  # S/m2
            self.pneuron.ELeak)

    @property
    def tauax(self):
        ''' Axial time constant (ms). '''
        return self.g2tau(self.ga)

    @tauax.setter
    def tauax(self, value):
        ''' Update axial conductance per unit area to match time new axial time constant. '''
        self.ga = self.tau2g(value)  # mS/cm2

    def setTimeConstants(self, taum, tauax):
        ''' Update benchmark according to pair of time constants. '''
        self.taum = taum  # ms
        self.tauax = tauax  # ms

    def setDrive(self, f, gamma_pair):
        ''' Update benchmark drive to a new frequency and amplitude. '''
        self.f = f
        self.gamma_pair = gamma_pair

    def getPassiveTstop(self, f_US):
        ''' Compute minimum simulation time for a passive model (ms). '''
        return 5 * max(self.taum, self.tauax, 2 / f_US)

    @property
    def passive_tstop(self):
        return self.getPassiveTstop(self.f)

    def sim(self, tstop):
        ''' Run benchmark simulations of the model. '''
        logger.info(f'{self}: {tstop:.2f} ms simulation')
        # Simulate with full and effective systems
        t, sol = {}, {}
        for method in ['full', 'effective']:
            t[method], sol[method] = self.simulate(method, tstop)
        # Cycle average full solution
        t['cycle-avg'], sol['cycle-avg'] = self.cycleAvg(t['full'], sol['full'])

        # Compute gradients
        new_sol = {}
        for key in ['full', 'cycle-avg', 'effective']:
            new_sol[key] = self.computeGradient(sol[key])
            # for ykey, yarray in sol[key].items():
            # new_sol[key] = sol[key]
        # sol = {k: sol[k] for k in ['full', 'cycle-avg', 'effective']}
        return t, new_sol

    def isExcited(self, gamma, tstop, method):
        ''' Simulate with SONIC paradigm for a given gamma-tstop combination, and check
            excitation of the passive node.
        '''
        self.gamma_pair = (gamma, 0.)
        t, sol = self.simulate(method, tstop)
        if method == 'full':
            t, sol = self.cycleAvg(t, sol)
        Qm_passive = sol['Qm'][1]
        return Qm_passive.max() > 0.

    def titrate(self, tstop, method='effective'):
        logger.info(f'running {method} titration for f = {self.f:.0f} kHz')
        return threshold(lambda x: self.isExcited(x, tstop, method), (0., 1.))

    def plot(self, t, sol, Qonly=False, gradient=False):
        ''' Plot results of benchmark simulations of the model. '''
        colors = ['C0', 'C1', 'darkgrey']
        markers = ['-', '--', '-']
        alphas = [0.5, 1., 1.]
        if Qonly:
            sol = {key: {'Qm': value['Qm']} for key, value in sol.items()}
        varkeys = list(sol[list(sol.keys())[0]].keys())
        naxes = len(varkeys)
        # if not self.passive:
        #     naxes += len(self.states)
        fig, axes = plt.subplots(naxes, 1, sharex=True, figsize=(10, min(3 * naxes, 10)))
        if naxes == 1:
            axes = [axes]
        axes[0].set_title(f'{self} - {t[list(t.keys())[0]][-1]:.2f} ms simulation')
        axes[-1].set_xlabel(f'time ({self.varunits["t"]})')
        for ax, k in zip(axes, sol[list(sol.keys())[0]].keys()):
            ax.set_ylabel(f'{k} ({self.varunits.get(k, "-")})')
            if k == 'Qm':
                for Qm, c in zip(self.Qminf_SONIC, colors):
                    ax.axhline(Qm, c=c, linestyle=':')
                if gradient:
                    ax.axhline(np.diff(self.Qminf_SONIC), c=colors[-1], linestyle=':')
            # if k == 'Qm':
            #     ax.set_ylim(-250.0, 50.)

        lbls = self.nodelabels
        if gradient:
            lbls.append('gradient')
        for m, alpha, (key, varsdict) in zip(markers, alphas, sol.items()):
            for ax, (k, v) in zip(axes, varsdict.items()):
                for y, c, lbl in zip(v, colors, lbls):
                    ax.plot(t[key], y, m, alpha=alpha, c=c, label=f'{lbl} - {key}')
        fig.subplots_adjust(bottom=0.2)
        axes[-1].legend(
            bbox_to_anchor=(0., -0.7, 1., .1), loc='upper center',
            ncol=3, mode="expand", borderaxespad=0.)
        return fig

    def plotV(self, t, sol):
        ''' Plot results of benchmark simulations of the model. '''
        colors = ['C0', 'C1']
        markers = ['-', '--', '-']
        alphas = [0.5, 1., 1.]
        V = {key: value['Qm'] / self.Cm0 for key, value in sol.items()}
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_title(f'{self} - {t[list(t.keys())[0]][-1]:.2f} ms simulation')
        ax.set_xlabel(f'time ({self.varunits["t"]})')
        ax.set_ylabel(f'Qm / Cm0 (mV)')
        ax.set_ylim(-100.0, 50.)
        for m, alpha, (key, varsdict) in zip(markers, alphas, sol.items()):
            for y, c, lbl in zip(V[key], colors, self.nodelabels):
                ax.plot(t[key], y, m, alpha=alpha, c=c, label=f'{lbl} - {key}')
        fig.subplots_adjust(bottom=0.2)
        ax.legend(bbox_to_anchor=(0., -0.7, 1., .1), loc='upper center', ncol=3,
                  mode="expand", borderaxespad=0.)
        return fig

    def simplot(self, *args, **kwargs):
        ''' Run benchmark simulation and plot results. '''
        return self.plot(*self.sim(*args, **kwargs))

    def divergencePerNode(self, t, sol, eval_mode='avg'):
        ''' Evaluate the divergence between the effective and full, cycle-averaged solutions
            at a specific point in time, computing per-node differences in charge density values
            divided by resting capacitance.
        '''
        # Compute matrix of charge density differences, normalized by resting capacitance
        dV_mat = (sol['effective']['Qm'] - sol['cycle-avg']['Qm']) / self.Cm0  # mV

        # Keep only the first two rows (3rd one, if any, is a gradient)
        dV_mat = dV_mat[:2, :]

        # Remove first index and take absolute value
        dV_mat = np.abs(dV_mat[:, 1:])

        # Compute summary metrics of difference per node according to evaluation mode
        if eval_mode == 'end':  # final index
            dV_vec = dV_mat[:, -1]
        elif eval_mode == 'avg':  # average
            dV_vec = np.mean(dV_mat, axis=1)
        elif eval_mode == 'max':  # max absolute difference
            dV_vec = np.max(dV_mat, axis=1)
        else:
            raise ValueError(f'{eval_mode} evaluation mode is not supported')

        # Cast into dictionary and return
        dV_dict = dict(zip(self.nodelabels, np.squeeze(dV_vec)))
        logger.debug(f'Vm differences per node: ', {k: f'{v:.2e}' for k, v in dV_dict.items()})
        return dV_dict

    def divergence(self, *args, **kwargs):
        dV_dict = self.divergencePerNode(*args, **kwargs)  # mV
        return max(list(dV_dict.values()))                 # mV


class DivergenceMap(XYMap):
    ''' Interface to a 2D map showing divergence of the SONIC output from a
        cycle-averaged NICE output, for various combinations of parameters.
    '''

    zkey = 'dV'
    zunit = 'mV'
    zfactor = 1e0
    suffix = 'Vdiff'

    def __init__(self, root, sb, eval_mode, *args, tstop=None, **kwargs):
        self.sb = sb.copy()
        self.eval_mode = eval_mode
        self.tstop = tstop
        super().__init__(root, *args, **kwargs)

    @property
    def tstop(self):
        if self._tstop is None:
            return self.sb.passive_tstop
        return self._tstop

    @tstop.setter
    def tstop(self, value):
        self._tstop = value

    def descPair(self, x1, x2):
        raise NotImplementedError

    def updateBenchmark(self, x):
        raise NotImplementedError

    def compute(self, x):
        self.updateBenchmark(x)
        t, sol = self.sb.sim(self.tstop)
        dV = self.sb.divergence(t, sol, eval_mode=self.eval_mode)  # mV
        logger.info(f'{self.descPair(*x)}, dV = {dV:.2e} mV')
        return dV

    def onClick(self, event):
        ''' Execute action when the user clicks on a cell in the 2D map. '''
        x = self.getOnClickXY(event)
        self.updateBenchmark(x)
        ix, iy = [np.where(vec == val)[0][0] for vec, val in zip([self.xvec, self.yvec], x)]
        dV_log = self.getOutput()[iy, ix]
        t, sol = self.sb.sim(self.tstop)
        dV = self.sb.divergence(t, sol, eval_mode=self.eval_mode)  # mV
        if not np.isclose(dV_log, dV):
            raise ValueError(
                f'computed divergence ({dV:.2e} mV) does not match log reference ({dV_log:.2e} mV)')
        logger.info(f'{self.descPair(*x)}, dV = {dV:.2e} mV')
        fig = self.sb.plot(t, sol)
        fig.axes[0].set_title(self.descPair(*x))
        plt.show()

    def render(self, zscale='log', dVmax=None, zbounds=(1e-1, 1e1),
               extend_under=True, extend_over=True, cmap='Spectral_r', figsize=(6, 4), fs=12,
               **kwargs):
        fig = super().render(
            zscale=zscale, zbounds=zbounds, extend_under=extend_under, extend_over=extend_over,
            cmap=cmap, figsize=figsize, fs=fs, **kwargs)
        if dVmax is not None:
            ax = fig.axes[0]
            if zscale == 'log':
                levels = [dVmax / 10, dVmax, dVmax * 10]
            else:
                levels = [dVmax / 2, dVmax, dVmax * 2]
            #lstyles = ['dashed', 'solid', 'dashdot']
            fmt = lambda x: f'{x:g}'  # ' mV'
            CS = ax.contour(
                self.xvec, self.yvec, self.getOutput(), levels, colors='k')
            ax.clabel(CS, fontsize=fs, fmt=fmt, inline_spacing=2)
        return fig


class TauDivergenceMap(DivergenceMap):
    ''' Divergence map of a passive model for various combinations of
        membrane time constants (taum) and axial time constant (tauax)
    '''

    xkey = 'tau_m'
    xfactor = 1e0
    xunit = 'ms'
    ykey = 'tau_ax'
    yfactor = 1e0
    yunit = 'ms'
    ga_default = 1e0  # mS/cm2

    @property
    def title(self):
        return f'Tau div map (f = {self.sb.f:.0f} kHz, gamma = ({", ".join(self.sb.gamma_str)})'

    def corecode(self):
        return f'tau_divmap_f{self.sb.f:.0f}kHz_gamma{"_".join(self.sb.gamma_str)}'

    def descPair(self, taum, tauax):
        return f'taum = {taum:.2e} ms, tauax = {tauax:.2e} ms'

    def updateBenchmark(self, x):
        self.sb.setTimeConstants(*x)

    def render(self, xscale='log', yscale='log', **kwargs):
        fig = super().render(xscale=xscale, yscale=yscale, **kwargs)
        fig.canvas.set_window_title(self.corecode())
        return fig


class OldDriveDivergenceMap(DivergenceMap):
    ''' Divergence map of a specific (membrane model, axial coupling) pairfor various
        combinations of drive frequencies and drive amplitudes.
    '''

    xkey = 'f_US'
    xfactor = 1e0
    xunit = 'kHz'
    ykey = 'gamma'
    yfactor = 1e0
    yunit = '-'

    @property
    def title(self):
        return f'Drive divergence map - {self.sb.pneuron.name}, tauax = {self.sb.tauax:.2e} ms)'

    def corecode(self):
        if self.sb.isPassive():
            neuron_desc = f'passive_taum_{self.sb.taum:.2e}ms'
        else:
            neuron_desc = self.sb.pneuron.name
            if self.sb.passive:
                neuron_desc = f'passive_{neuron_desc}'
        code = f'drive_divmap_{neuron_desc}_tauax_{self.sb.tauax:.2e}ms'
        if self._tstop is not None:
            code = f'{code}_tstop{self.tstop:.2f}ms'
        return code

    def descPair(self, f_US, A_Cm):
        return f'f = {f_US:.2f} kHz, gamma = {A_Cm:.2f}'

    def updateBenchmark(self, x):
        f, gamma = x
        self.sb.setDrive(f, (gamma, 0.))

    def threshold_filename(self, method):
        fmin, fmax = bounds(self.xvec)
        return f'{self.corecode()}_f{fmin:.0f}kHz_{fmax:.0f}kHz_{self.xvec.size}_gammathrs_{method}.txt'

    def threshold_filepath(self, *args, **kwargs):
        return os.path.join(self.root, self.threshold_filename(*args, **kwargs))

    def addThresholdCurves(self, ax):
        ls = ['--', '-.']
        for j, method in enumerate(['effective', 'full']):
            fpath = self.threshold_filepath(method)
            if os.path.isfile(fpath):
                gamma_thrs = np.loadtxt(fpath)
            else:
                gamma_thrs = np.empty(self.xvec.size)
                for i, f in enumerate(self.xvec):
                    self.sb.f = f
                    gamma_thrs[i] = self.sb.titrate(self.tstop, method=method)
                np.savetxt(fpath, gamma_thrs)
            ylims = ax.get_ylim()
            ax.plot(self.xvec * self.xfactor, gamma_thrs * self.yfactor, ls[j], color='k')
            ax.set_ylim(ylims)

    def render(self, xscale='log', thresholds=False, **kwargs):
        fig = super().render(xscale=xscale, **kwargs)
        if thresholds:
            self.addThresholdCurves(fig.axes[0])
        return fig

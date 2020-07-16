# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-06-29 18:11:24
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-07-16 10:39:22

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from PySONIC.core import EffectiveVariablesLookup
from PySONIC.utils import logger, timer, isWithin, bounds
from PySONIC.plt import XYMap, setNormalizer
from PySONIC.neurons import passiveNeuron


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
    ga_bounds = [1e-10, 1e10]  # mS/cm2

    def __init__(self, pneuron, ga, f, rel_amps, passive=False):
        ''' Initialization.

            :param pneuron: point-neuron object
            :param ga: axial conductance (mS/cm2)
            :param f: US frequency (kHz)
            :param rel_amps: pair of relative capacitance oscillation amplitudes
        '''
        self.pneuron = pneuron
        self.ga = ga
        self.f = f
        self.rel_amps = rel_amps
        self.passive = passive
        self.computeLookups()

    def copy(self):
        return self.__class__(self.pneuron, self.ga, self.f, self.rel_amps, passive=self.passive)

    @property
    def strAmps(self):
        s = ', '.join([f'{x:.2f}' for x in self.rel_amps])
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
    def rel_amps(self):
        return self._rel_amps

    @rel_amps.setter
    def rel_amps(self, value):
        self._rel_amps = value
        if hasattr(self, 'lkps'):
            self.computeLookups()

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

    def computeLookups(self):
        # Compute lookups over 1 cycle
        Cmeff = []
        self.lkps = []
        for A in self.rel_amps:
            Cm_cycle = self.capct(A, self.tcycle)    # uF/cm2
            Cmeff.append(1 / np.mean(1 / Cm_cycle))  # uF/cm2
            if not self.passive:
                self.lkps.append(self.getLookup(Cm_cycle))
        self.Cmeff = np.array(Cmeff)

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

    def setDrive(self, f_US, A_Cm):
        ''' Update benchmark drive to a new frequency and amplitude. '''
        self.f = f_US
        self.rel_amps = (A_Cm, 0.)

    def getPassiveTstop(self, f_US):
        ''' Compute minimum simulation time for a passive model (ms). '''
        return 5 * max(self.taum, self.tauax, 1 / f_US)

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
        return t, sol

    def plot(self, t, sol):
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

    def invexp(self, x, x0, a, b):
        ''' Inverse exponential. '''
        return np.exp(a * (x - x0)) + b

    def invexpFit(self, x, y, logify=False):
        ''' Inverse exponential fit. '''
        if logify:
            x, y = np.log10(x), np.log10(y)
        inds = np.where(~np.isnan(y))[0]
        pguess = (np.nanmin(x) + 1, -1, np.nanmin(y))
        popt, _ = curve_fit(self.invexp, x[inds], y[inds], pguess)
        yinterp = self.invexp(x, *popt)
        if logify:
            yinterp = np.power(10, yinterp)
        return yinterp

    def addThresholdCurve(self, fig, dVmax, fs, fit_method=None, logify=False):
        ax, cbar_ax = fig.axes
        levels = [dVmax / 10, dVmax, dVmax * 10]
        lstyles = ['dashed', 'solid', 'dashdot']
        CS = ax.contour(
            self.xvec, self.yvec, self.getOutput(), levels, colors='k', linestyles=lstyles)
        ax.clabel(CS, fontsize=fs, fmt='%1.1f', inline_spacing=0)

    def render1D(self, xscale='log', yscale='log', zscale='log', cmap='viridis',
                 figsize=(6, 4), fs=10, dVmax=None):
        mymap = plt.get_cmap(cmap)
        norm, sm = setNormalizer(mymap, bounds(self.xvec), zscale)
        colors = sm.to_rgba(self.xvec)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title('SONIC divergence 1D plot', fontsize=fs)
        ax.set_xlabel(f'{self.ykey} ({self.yunit})', fontsize=fs)
        ax.set_ylabel(f'{self.zkey} ({self.zunit})', fontsize=fs)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if dVmax is not None:
            ax.axhline(dVmax, color='k', linestyle='--')
        for i, dV in enumerate(self.getOutput().T):
            ax.plot(self.yvec, dV, c=colors[i])
            if dVmax is not None:
                if np.all(np.diff(dV) < 0):
                    dV, dVmax = -dV, -dVmax
                ax.axvline(np.interp(dVmax, dV, self.yvec), linestyle='--', color=colors[i])
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)
        fig.subplots_adjust(left=0.1, right=0.8, bottom=0.15, top=0.95, hspace=0.5)
        cbarax = fig.add_axes([0.85, 0.15, 0.03, 0.8])
        fig.colorbar(sm, cax=cbarax)
        cbarax.set_xlabel(f'{self.xkey} ({self.xunit})', fontsize=fs, labelpad=20)
        for item in cbarax.get_yticklabels():
            item.set_fontsize(fs)
        return fig

    def render(self, zscale='log', logify=False, dVmax=None, mode='2d', zbounds=(1e-1, 1e1),
               extend_under=True, extend_over=True, cmap='Spectral_r', figsize=(6, 4), fs=12,
               **kwargs):
        if mode == '2d':
            fig = super().render(
                zscale=zscale, zbounds=zbounds, extend_under=extend_under, extend_over=extend_over,
                cmap=cmap, figsize=figsize, fs=fs, **kwargs)
            if dVmax is not None:
                self.addThresholdCurve(
                    fig, dVmax, fs, fit_method=self.fit_method, logify=logify)
        else:
            fig = self.render1D(zscale=zscale, dVmax=dVmax, figsize=figsize, fs=fs, **kwargs)
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

    def fit_method(self, *args, **kwargs):
        return self.invexpFit(*args, **kwargs)

    @property
    def title(self):
        return f'Tau divergence map (f = {self.sb.f:.0f} kHz, gamma = {self.sb.rel_amps[0]:.2f})'

    def corecode(self):
        return f'tau_divmap_f{self.sb.f:.0f}kHz_gamma{self.sb.rel_amps[0]:.2f}'

    def descPair(self, taum, tauax):
        return f'taum = {taum:.2e} ms, tauax = {tauax:.2e} ms'

    def updateBenchmark(self, x):
        self.sb.setTimeConstants(*x)

    def render(self, xscale='log', yscale='log', logify=True, **kwargs):
        return super().render(xscale=xscale, yscale=yscale, logify=logify, **kwargs)


class DriveDivergenceMap(DivergenceMap):
    ''' Divergence map of a specific (membrane model, axial coupling) pairfor various
        combinations of drive frequencies and drive amplitudes.
    '''

    xkey = 'f_US'
    xfactor = 1e0
    xunit = 'kHz'
    ykey = 'gamma'
    yfactor = 1e0
    yunit = '-'
    fit_method = None

    @property
    def title(self):
        return f'Drive divergence map - (taum = {self.sb.taum:.2e} ms, tauax = {self.sb.tauax:.2e} ms)'

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
        self.sb.setDrive(*x)

    def render(self, xscale='log', **kwargs):
        return super().render(xscale=xscale, **kwargs)

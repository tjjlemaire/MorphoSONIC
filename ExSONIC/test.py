# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 11:34:09
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-14 16:35:57

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PySONIC.core import NeuronalBilayerSonophore, PulsedProtocol, AcousticDrive, ElectricDrive
from PySONIC.test import TestBase
from PySONIC.neurons import getNeuronsDict
from PySONIC.plt import GroupedTimeSeries
from PySONIC.utils import logger, si_format, pow10_format, isIterable

from .constants import *
from .core import *
from .models import *


class TestComp(TestBase):
    ''' Generic test interface to run simulations with 2 different methods and compare
        resulting membrane potential and membrane charge density profiles, along with
        computation times. '''

    @staticmethod
    def runSims(pneuron, drive, pp, a, Fdrive, dt, atol):
        raise NotImplementedError

    @staticmethod
    def createFigureBackbone():
        fig = plt.figure(figsize=(16, 3))
        gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1])
        axes = list(map(plt.subplot, gs))
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for item in ax.get_xticklabels() + ax.get_xticklabels():
                item.set_fontsize(10)
        fig.subplots_adjust(top=0.8, left=0.05, right=0.95)
        for ax in axes[:2]:
            ax.set_xlabel('time (ms)', fontsize=10)
        return fig, axes

    @staticmethod
    def plotStimPatches(axes, tbounds, pulses):
        tstart, tend, x = zip(*pulses)
        colors = GroupedTimeSeries.getPatchesColors(x)
        for ax in axes:
            ax.legend(fontsize=10, frameon=False)
            ax.set_xlim(tbounds[0] * S_TO_MS, tbounds[1] * S_TO_MS)
            for i in range(len(colors)):
                ax.axvspan(tstart[i] * S_TO_MS, tend[i] * S_TO_MS,
                           edgecolor='none', facecolor=colors[i], alpha=0.2)

    @classmethod
    def plotTcompHistogram(cls, ax, tcomp):

        # Plot comparative time histogram
        ax.set_ylabel('comp. time (s)', fontsize=10)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(cls.comp_keys)
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 1e2)
        for i, k in enumerate(cls.comp_keys):
            tc = tcomp[k]
            idx = i + 1
            ax.bar(idx, tc, align='center')
            ax.text(idx, 1.5 * tc, '{}s'.format(si_format(tc, 2, space=' ')),
                    horizontalalignment='center')

    @classmethod
    def compare(cls, pneuron, drive, pp, a=None, dt=None, atol=None):
        raise NotImplementedError


class TestCompNode(TestComp):

    fiber_mechs = ['FHnode', 'SWnode', 'MRGnode', 'SUseg']
    fiber_pp = PulsedProtocol(100e-6, 3e-3)
    cortical_pp = PulsedProtocol(100e-3, 50e-3)

    @staticmethod
    def getNeurons():
        pneurons = {}
        for name, neuron_class in getNeuronsDict().items():
            if name not in ('template', 'LeechP', 'LeechT', 'LeechR', 'SW', 'sundt'):
                pneurons[name] = neuron_class()
        return pneurons

    @classmethod
    def compare(cls, pneuron, drive, pp, a=None, dt=None, atol=None):

        # Run simulations
        data, meta, tcomp, comp_title = cls.runSims(pneuron, drive, pp, a, dt, atol)

        # Extract ref time and stim state
        ref_t = data[cls.def_key]['t'].values
        ref_state = data[cls.def_key]['stimstate'].values

        # Determine time onset and stim patches
        tonset = 0.05 * (np.ptp(ref_t))
        pulses = GroupedTimeSeries.getStimPulses(ref_t, ref_state)

        # Create comparative figure backbone
        fig, axes = cls.createFigureBackbone()

        # Plot charge density and membrane potential profiles
        ax = axes[0]
        ax.set_ylim(pneuron.Qbounds * C_M2_TO_NC_CM2)
        ax.set_ylabel('Qm (nC/cm2)', fontsize=10)
        ax.set_title('membrane charge density', fontsize=12)
        ax = axes[1]
        ax.set_ylim(-150, 70)
        ax.set_ylabel('$V_{m, eff}$ (mV)', fontsize=10)
        ax.set_title('effective membrane potential', fontsize=12)
        for k in cls.comp_keys:
            tplt = np.insert(data[k]['t'].values, 0, -tonset) * S_TO_MS
            axes[0].plot(
                tplt, np.insert(data[k]['Qm'].values, 0, data[k]['Qm'].values[0]) * C_M2_TO_NC_CM2,
                label=k)
            axes[1].plot(
                tplt, np.insert(data[k]['Vm'].values, 0, data[k]['Vm'].values[0]),
                label=k)

        # Plot stim patches on both graphs
        tbounds = (-tonset, pp.tstim + pp.toffset)
        cls.plotStimPatches(axes[:-1], tbounds, pulses)

        # Plot comparative histogram of computation times
        cls.plotTcompHistogram(axes[2], tcomp)

        # Add figure title
        if isIterable(comp_title):
            comp_title = ' - '.join(comp_title)
        fig.suptitle(comp_title, fontsize=18)

        return fig

    def test_ESTIM(self, is_profiled=False):
        for name, pneuron in self.getNeurons().items():
            if pneuron.name in self.fiber_mechs:
                Astim = 1e4     # mA/m2
                pp = self.fiber_pp
            else:
                Astim = 20.0     # mA/m2
                pp = self.cortical_pp
            ELdrive = ElectricDrive(Astim)
            fig = self.compare(pneuron, ELdrive, pp)

    def test_ASTIM(self, is_profiled=False):
        a = 32e-9        # nm
        Fdrive = 500e3   # kHz
        for name, pneuron in self.getNeurons().items():
            if pneuron.name in self.fiber_mechs:
                Adrive = 300e3  # kPa
                pp = self.fiber_pp
            else:
                Adrive = 100e3  # kPa
                pp = self.cortical_pp
            USdrive = AcousticDrive(Fdrive, Adrive)
            fig = self.compare(pneuron, USdrive, pp, a=a)


class TestNodePythonVsNeuron(TestCompNode):
    ''' Run Node ESTIM and ASTIM simulations and compare results with those obtained
        with pure-Python implementations (PointNeuron and NeuronalBilayerSonophore classes). '''

    comp_keys = ['Python', 'NEURON']
    def_key = 'Python'

    @staticmethod
    def runSims(pneuron, drive, pp, a, dt, atol):

        # Initialize Python and NEURON models
        if a is not None:
            py_model = NeuronalBilayerSonophore(a, pneuron)
        else:
            py_model = pneuron
        nrn_model = Node(pneuron, a=a)

        # Run NEURON and Python simulations
        data, meta = {}, {}
        data['NEURON'], meta['NEURON'] = nrn_model.simulate(drive, pp, dt, atol)
        data['Python'], meta['Python'] = py_model.simulate(drive, pp)
        tcomp = {k: v['tcomp'] for k, v in meta.items()}

        comp_title = (
            f'{nrn_model}, {drive.desc}, {pp.desc}',
            'adaptive time step' if dt is None else f'dt = ${pow10_format(dt * S_TO_MS)}$ ms')

        return data, meta, tcomp, comp_title


class TestCompExtended(TestComp):

    @classmethod
    def compare(cls, *args, **kwargs):

        # Run simulations
        data, meta, tcomp, comp_title = cls.runSims(*args, **kwargs)

        # Extract ref time and stim state
        ref_key = list(data[cls.def_key].keys())[0]
        ref_t = data[cls.def_key][ref_key]['t'].values
        ref_state = data[cls.def_key][ref_key]['stimstate'].values

        # Determine time onset and stim patches
        tonset = 0.05 * (np.ptp(ref_t))
        pulses = GroupedTimeSeries.getStimPulses(ref_t, ref_state)

        # Create comparative figure backbone
        fig, axes = cls.createFigureBackbone()

        # Plot membrane potential profiles for both conditions
        for i, key in enumerate(cls.comp_keys):
            ax = axes[i]
            df = data[key]
            ax.set_ylim(-150, 70)
            ax.set_ylabel('$V_m^*$ (mV)', fontsize=10)
            ax.set_title(key, fontsize=12)
            for k in df.keys():
                tplt = np.insert(df[k]['t'].values, 0, -tonset) * S_TO_MS
                ax.plot(tplt, np.insert(df[k]['Vm'].values, 0, df[k]['Vm'].values[0]), label=k)

        # Plot stim patches on both graphs
        pp = args[-1]
        tbounds = (-tonset, pp.tstim + pp.toffset)
        cls.plotStimPatches(axes[:-1], tbounds, pulses)

        # Plot comparative histogram of computation times
        cls.plotTcompHistogram(axes[2], tcomp)

        # Add figure title
        if isIterable(comp_title):
            comp_title = ' - '.join(comp_title)
        fig.suptitle(comp_title, fontsize=18)

        return fig


class TestConnectClassicVsCustom(TestCompExtended):
    ''' Run simulations with a fiber object in which sections are connected using either
        NEURON's classic built-in scheme or a custom-made scheme, and compare results.
    '''

    comp_keys = ['classic', 'custom']
    def_key = 'classic'

    def test_senn(self, is_profiled=False):

        # Fiber model
        fiber = SennFiber(10e-6, nnodes=3)

        # Intracellular stimulation parameters
        Istim = None
        pp = PulsedProtocol(1e-3, 3e-3)

        return self.compare(fiber, Istim, pp)

    @classmethod
    def runSims(cls, fiber, Istim, pp):

        # Create fiber objects with both classic and custom connection schemes
        fibers = fiber.compdict(original_key='classic', sonic_key='custom')

        # Create extracellular current source
        source = IntracellularCurrent('node0', mode='anode')

        # Titrate classic model for a specific duration
        if Istim is None:
            logger.info(f'Running titration for {si_format(pp.tstim)}s pulse')
            Istim = fibers[cls.def_key].titrate(source, pp)  # A
            source.I = Istim

        # Simulate both fiber models above threshold current
        data, meta = {}, {}
        for k, f in fibers.items():
            data[k], meta[k] = f.simulate(source, pp)
        tcomp = {k: v['tcomp'] for k, v in meta.items()}

        comp_title = '{}, I = {}A, {}s'.format(
            fibers[cls.def_key],
            *si_format([Istim, pp.tstim], space=' '))

        return data, meta, tcomp, comp_title


class TestFiber(TestBase):

    @staticmethod
    def relativeChange(x, xref):
        if isIterable(x):
            x, xref = np.asarray(x), np.asarray(xref)
        return np.mean((x - xref) / xref)

    @classmethod
    def logOutputMetrics(cls, sim_metrics, ref_metrics=None):
        ''' Log output metrics. '''
        logfmts = {
            'Q0': {
                'name': 'rheobase charge',
                'fmt': lambda x: f'{si_format(np.abs(x), 1)}C'
            },
            'I0': {
                'name': 'rheobase current',
                'fmt': lambda x: f'{si_format(np.abs(x), 1)}A'
            },
            'Ithr': {
                'name': 'threshold current',
                'fmt': lambda x: f'{si_format(np.abs(x), 1)}A'
            },
            'Vthr': {
                'name': 'threshold extracellular voltage',
                'fmt': lambda x: f'{np.abs(x):.2f}mV'
            },
            'cv': {
                'name': 'conduction velocity',
                'fmt': lambda x: f'{x:.2f} m/s'
            },
            'dV': {
                'name': 'spike amplitude',
                'fmt': lambda x: '{:.1f}-{:.1f} mV'.format(*x)
            },
            'Athr': {
                'name': 'threshold US amplitude',
                'fmt': lambda x: f'{si_format(x, 2)}Pa'
            },
            'uthr': {
                'name': 'threshold transducer particle velocity',
                'fmt': lambda x: f'{si_format(x, 2)}m/s'
            },
            'chr': {
                'name': 'chronaxie',
                'fmt': lambda x: f'{si_format(x, 1)}s'
            },
            'tau_e': {
                'name': 'S/D time constant',
                'fmt': lambda x: f'{si_format(x, 1)}s'
            }
        }
        for t in [1e0, 1e1, 1e3, 1e4]:
            key = f'P{si_format(t / S_TO_US, 0, space="")}s'
            desc = f'{si_format(t / S_TO_US, 0)}s'
            logfmts[key] = {
                'name': f'polarity selectivity ratio @ {desc}',
                'fmt': lambda x: f'{x:.2f}'
            }

        for k, v in logfmts.items():
            warn = False
            if k in sim_metrics:
                log = f"--- {v['name']}: {k} = {v['fmt'](sim_metrics[k])}"
                if ref_metrics is not None and k in ref_metrics:
                    rel_change = cls.relativeChange(sim_metrics[k], ref_metrics[k])
                    if np.abs(rel_change) > 0.05:
                        warn = True
                    log += f" (ref = {v['fmt'](ref_metrics[k])}, {rel_change * 100:.2f}% change)"
                if warn:
                    logger.warning(log)
                else:
                    logger.info(log)

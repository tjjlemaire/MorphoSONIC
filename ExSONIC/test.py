# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 11:34:09
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-22 18:06:19

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.test import TestBase
from PySONIC.neurons import getNeuronsDict
from PySONIC.plt import GroupedTimeSeries
from PySONIC.utils import logger, si_format, pow10_format, isIterable

from .core import IintraNode, SonicNode


class TestComp(TestBase):
    ''' Generic test interface to run simulations with 2 different methods and compare
        resulting membrane potential and membrane charge density profiles, along with
        computation times. '''

    @staticmethod
    def getNeurons():
        pneurons = {}
        for name, neuron_class in getNeuronsDict().items():
            if name not in ('template', 'LeechP', 'LeechT', 'LeechR', 'SW'):
                pneurons[name] = neuron_class()
        return pneurons

    def test_ESTIM(self, is_profiled=False):
        for name, pneuron in self.getNeurons().items():
            if pneuron.name == 'FH':
                Astim = 1e4      # mA/m2
                tstim = 0.12e-3  # s
                toffset = 3e-3   # s
            else:
                Astim = 20.0  # mA/m2
                tstim = 100e-3   # s
                toffset = 50e-3  # s
            fig = self.compare(pneuron, Astim, tstim, toffset)

    def test_ASTIM(self, is_profiled=False):
        a = 32e-9        # nm
        Fdrive = 500e3   # kHz
        tstim = 100e-3   # s
        toffset = 50e-3  # s
        DC = 0.5         # (-)
        for name, pneuron in self.getNeurons().items():
            if pneuron.name == 'FH':
                Adrive = 300e3  # kPa
            else:
                Adrive = 100e3  # kPa
            fig = self.compare(pneuron, Adrive, tstim, toffset, DC=DC, a=a, Fdrive=Fdrive)

    @staticmethod
    def runSims(pneuron, A, tstim, toffset, PRF, DC, a, Fdrive, dt, atol):
        return NotImplementedError

    @classmethod
    def compare(cls, pneuron, A, tstim, toffset, PRF=100., DC=1., a=None, Fdrive=None,
                dt=None, atol=None):

        data, meta, tcomp, comp_title = cls.runSims(
            pneuron, A, tstim, toffset, PRF, DC, a, Fdrive, dt, atol)

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
        for ax in axes[:2]:
            ax.set_xlabel('time (ms)', fontsize=fs)
        ax = axes[0]
        ax.set_ylim(pneuron.Qbounds() * 1e5)
        ax.set_ylabel('Qm (nC/cm2)', fontsize=fs)
        ax.set_title('membrane charge density', fontsize=fs + 2)
        ax = axes[1]
        ax.set_ylim(-150, 70)
        ax.set_ylabel('$V_{m, eff}$ (mV)', fontsize=fs)
        ax.set_title('effective membrane potential', fontsize=fs + 2)
        ax = axes[2]
        ax.set_ylabel('comp. time (s)', fontsize=fs)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(cls.comp_keys)
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 1e2)

        # Get pulses timing
        tpatch_on, tpatch_off = GroupedTimeSeries.getStimPulses(
            data[cls.def_key]['t'].values, data[cls.def_key]['stimstate'].values)

        # Plot charge density and membrane potential profiles
        tonset = 0.05 * (np.ptp(data[cls.def_key]['t'].values))
        for k in cls.comp_keys:
            tplt = np.insert(data[k]['t'].values, 0, -tonset) * 1e3
            axes[0].plot(tplt, np.insert(data[k]['Qm'].values, 0, data[k]['Qm'].values[0]) * 1e5, label=k)
            axes[1].plot(tplt, np.insert(data[k]['Vm'].values, 0, data[k]['Vm'].values[0]), label=k)

        # Plot stim patches on both graphs
        for ax in axes[:2]:
            ax.legend(fontsize=fs, frameon=False)
            ax.set_xlim(-tonset * 1e3, (tstim + toffset) * 1e3)
            for ton, toff in zip(tpatch_on, tpatch_off):
                ax.axvspan(ton * 1e3, toff * 1e3, edgecolor='none', facecolor='#8A8A8A', alpha=0.2)

        # Plot comparative time histogram
        ax = axes[2]
        for i, k in enumerate(cls.comp_keys):
            tc = tcomp[k]
            idx = i + 1
            ax.bar(idx, tc, align='center')
            ax.text(idx, 1.5 * tc, '{}s'.format(si_format(tc, 2, space=' ')),
                    horizontalalignment='center')

        # Add figure title
        fig.suptitle(comp_title, fontsize=18)

        return fig


class TestNodePythonVsNeuron(TestComp):
    ''' Run Node (ESTIM) and SonicNode (ASTIM) simulations and compare results with those obtained
        with pure-Python implementations (PointNeuron and NeuronalBilayerSonophore classes). '''

    comp_keys = ['Python', 'NEURON']
    def_key = 'Python'

    @staticmethod
    def runSims(pneuron, A, tstim, toffset, PRF, DC, a, Fdrive, dt, atol):

        # Initialize Python and NEURON models
        args = [A, tstim, toffset, PRF, DC]
        if a is not None:
            nrn_model = SonicNode(pneuron, a=a, Fdrive=Fdrive)
            py_model = NeuronalBilayerSonophore(a, pneuron, Fdrive=Fdrive)
            py_args = [Fdrive] + args
        else:
            nrn_model = IintraNode(pneuron)
            py_model = pneuron
            py_args = args

        # Run NEURON and Python simulations
        data, meta = {}, {}
        data['NEURON'], meta['NEURON'] = nrn_model.simulate(*args, dt, atol)
        data['Python'], meta['Python'] = py_model.simulate(*py_args)
        tcomp = {k: v['tcomp'] for k, v in meta.items()}

        comp_title = '{}, A = {}{}, {}s'.format(
            nrn_model.strBiophysics(),
            si_format(A * nrn_model.modality['factor'], space=' '), nrn_model.modality['unit'],
            si_format(tstim, space=' '),
            'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))
        )

        return data, meta, tcomp, comp_title


class TestNmodlAutoVsManual(TestComp):
    ''' Run IintraNode (ESTIM) and SonicNode (ASTIM) simulations with manually and automatically created
        NMODL membrane mechanisms, and compare results.
    '''

    comp_keys = ['manual', 'auto']
    def_key = 'manual'

    @staticmethod
    def runSims(pneuron, A, tstim, toffset, PRF, DC, a, Fdrive, dt, atol):

        # Initialize models and run simulations
        args = [A, tstim, toffset, PRF, DC]
        data, meta = {}, {}
        if a is not None:
            manual_model = SonicNode(pneuron, a=a, Fdrive=Fdrive, auto_nmodl=False)
            auto_model = SonicNode(pneuron, a=a, Fdrive=Fdrive, auto_nmodl=True)
        else:
            manual_model = IintraNode(pneuron, auto_nmodl=False)
            auto_model = IintraNode(pneuron, auto_nmodl=True)

        data['auto'], meta['auto'] = auto_model.simulate(*args, dt, atol)
        data['manual'], meta['manual'] = manual_model.simulate(*args, dt, atol)
        tcomp = {k: v['tcomp'] for k, v in meta.items()}

        comp_title = '{}, A = {}{}, {}s'.format(
            manual_model.strBiophysics(),
            si_format(A * manual_model.modality['factor'], space=' '), manual_model.modality['unit'],
            si_format(tstim, space=' '),
            'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3)))

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
            'cv': {
                'name': 'conduction velocity',
                'fmt': lambda x: f'{x:.1f} m/s'
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
            key = f'P{si_format(t * 1e-6, 0, space="")}s'
            desc = f'{si_format(t * 1e-6, 0)}s'
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
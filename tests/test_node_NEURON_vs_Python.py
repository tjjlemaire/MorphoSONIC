# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-30 11:26:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-19 06:57:53

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PySONIC.test import TestBase
from PySONIC.neurons import getNeuronsDict
from PySONIC.utils import si_format, pow10_format
from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.plt import GroupedTimeSeries

from ExSONIC.core import IintraNode, SonicNode


class TestNode(TestBase):
    ''' Run Node (ESTIM) and SonicNode (ASTIM) simulations and compare results with those obtained
        with pure-Python implementations (PointNeuron and NeuronalBilayerSonophore classes). '''

    def test_ESTIM(self, is_profiled=False):
        for name, neuron_class in getNeuronsDict().items():
            if name not in ('template', 'LeechP', 'LeechT', 'LeechR'):
                pneuron = neuron_class()
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
        for name, neuron_class in getNeuronsDict().items():
            if name not in ('template', 'LeechP', 'LeechT', 'LeechR'):
                pneuron = neuron_class()
                if pneuron.name == 'FH':
                    Adrive = 300e3  # kPa
                else:
                    Adrive = 100e3  # kPa
                fig = self.compare(pneuron, Adrive, tstim, toffset, DC=DC, a=a, Fdrive=Fdrive)

    @staticmethod
    def compare(pneuron, A, tstim, toffset, PRF=100., DC=1., a=None, Fdrive=None,
                dt=None, atol=None):

        comp_keys = ['Python', 'NEURON']

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
        ax.set_xticklabels(comp_keys)
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 1e2)

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

        # Get pulses timing
        tpatch_on, tpatch_off = GroupedTimeSeries.getStimPulses(
            data['Python']['t'].values, data['Python']['stimstate'].values)

        # Plot charge density and membrane potential profiles
        tonset = -0.05 * (np.ptp(data['Python']['t']))
        for k in comp_keys:
            tplt = np.hstack((np.array([tonset, 0.]), data[k]['t'].values)) * 1e3
            axes[0].plot(tplt, np.hstack((np.ones(2) * pneuron.Qm0(), data[k]['Qm'])) * 1e5, label=k)
            axes[1].plot(tplt, np.hstack((np.ones(2) * pneuron.Vm0, data[k]['Vm'])), label=k)

        # Plot stim patches on both graphs
        for ax in axes[:2]:
            ax.legend(fontsize=fs, frameon=False)
            ax.set_xlim(tonset * 1e3, (tstim + toffset) * 1e3)
            for ton, toff in zip(tpatch_on, tpatch_off):
                ax.axvspan(ton * 1e3, toff * 1e3, edgecolor='none', facecolor='#8A8A8A', alpha=0.2)

        # Plot comparative time histogram
        ax = axes[2]
        for i, k in enumerate(comp_keys):
            tc = tcomp[k]
            idx = i + 1
            ax.bar(idx, tc, align='center')
            ax.text(idx, 1.5 * tc, '{}s'.format(si_format(tc, 2, space=' ')),
                    horizontalalignment='center')

        # Add figure title
        fig.suptitle('{}, A = {}{}, {}s'.format(
            nrn_model.strBiophysics(),
            si_format(A * nrn_model.modality['factor'], space=' '), nrn_model.modality['unit'],
            si_format(tstim, space=' '),
            'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))),
            fontsize=18)

        return fig


if __name__ == '__main__':
    tester = TestNode()
    tester.main()
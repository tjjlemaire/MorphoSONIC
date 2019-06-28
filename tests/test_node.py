# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-30 11:26:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-28 12:29:35

from PySONIC.test import TestBase
from PySONIC.neurons import getNeuronsDict
from ExSONIC._0D import compare


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
                fig = compare(pneuron, Astim, tstim, toffset)

    def test_ASTIM(self, is_profiled=False):
        a = 32.          # nm
        Fdrive = 500.    # kHz
        tstim = 100e-3   # s
        toffset = 50e-3  # s
        DC = 0.5         # (-)
        for name, neuron_class in getNeuronsDict().items():
            if name not in ('template', 'LeechP', 'LeechT', 'LeechR'):
                pneuron = neuron_class()
                if pneuron.name == 'FH':
                    Adrive = 300.    # kPa
                else:
                    Adrive = 100.    # kPa
                fig = compare(pneuron, Adrive, tstim, toffset, DC=DC, a=a, Fdrive=Fdrive)


if __name__ == '__main__':
    tester = TestNode()
    tester.main()
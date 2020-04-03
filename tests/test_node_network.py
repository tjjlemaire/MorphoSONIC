# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-13 19:51:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-02 16:40:39

import logging

from PySONIC.core import PulsedProtocol
from PySONIC.neurons import getPointNeuron
from PySONIC.test import TestBase
from PySONIC.utils import logger

from ExSONIC.plt import SectionCompTimeSeries, SectionGroupedTimeSeries
from ExSONIC.core.node import Node, DrivenNode
from ExSONIC.core.synapses import Exp2Synapse, FExp2Synapse, FDExp2Synapse
from ExSONIC.core.network import NodeCollection, NodeNetwork
from ExSONIC.parsers import TestNodeNetworkParser

''' Create and simulate a small network of nodes. '''

logger.setLevel(logging.INFO)


class TestNodeNetwork(TestBase):

    parser_class = TestNodeNetworkParser

    def runTests(self, testsets, args):
        ''' Run appropriate tests. '''
        for s in args['subset']:
            testsets[s](args['connect'])

    def __init__(self):
        ''' Initialize network components. '''

        # Point-neuron models
        self.pneurons = {k: getPointNeuron(k) for k in ['RS', 'FS', 'LTS']}

        # Synapse models
        RS_syn_base = Exp2Synapse(tau1=0.1, tau2=3.0, E=0.0)
        RS_LTS_syn = FExp2Synapse(
            tau1=RS_syn_base.tau1, tau2=RS_syn_base.tau2, E=RS_syn_base.E, f=0.2, tauF=200.0)
        RS_FS_syn = FDExp2Synapse(
            tau1=RS_syn_base.tau1, tau2=RS_syn_base.tau2, E=RS_syn_base.E, f=0.5, tauF=94.0,
            d1=0.46, tauD1=380.0, d2=0.975, tauD2=9200.0)
        FS_syn = Exp2Synapse(tau1=0.5, tau2=8.0, E=-85.0)
        LTS_syn = Exp2Synapse(tau1=0.5, tau2=50.0, E=-85.0)

        # Synaptic connections
        self.connections = {
            'RS': {
                'RS': (0.002, RS_syn_base),
                'FS': (0.04, RS_FS_syn),
                'LTS': (0.09, RS_LTS_syn)
            },
            'FS': {
                'RS': (0.015, FS_syn),
                'FS': (0.135, FS_syn),
                'LTS': (0.86, FS_syn)
            },
            'LTS': {
                'RS': (0.135, LTS_syn),
                'FS': (0.02, LTS_syn)
            }
        }

        # Driving currents
        I_Th_RS = 0.17  # nA
        Idrives = {  # nA
            'RS': I_Th_RS,
            'FS': 1.4 * I_Th_RS,
            'LTS': 0.0}
        self.idrives = {k: (v * 1e-6) / self.pneurons[k].area for k, v in Idrives.items()}  # mA/m2

        # Pulsing parameters
        tstim = 2.0    # s
        toffset = 1.0  # s
        PRF = 100.0    # Hz
        DC = 1.0       # (-)
        self.pp = PulsedProtocol(tstim, toffset, PRF, DC)

        # Sonophore parameters
        self.a = 32e-9
        self.fs = 1.0

        # US stimulation parameters
        self.Fdrive = 500e3  # Hz
        self.Adrive = 30e3  # Pa

    def simulate(self, nodes, amps, connect):
        # Create appropriate system
        if connect:
            system = NodeNetwork(nodes, self.connections)
        else:
            system = NodeCollection(nodes)

        # Simulate system
        data, meta = system.simulate(amps, self.pp)

        # Plot membrane potential traces and comparative firing rate profiles
        for id in system.ids:
            SectionGroupedTimeSeries(id, [(data, meta)], pltscheme={'Q_m': ['Qm']}).render()
        # SectionCompTimeSeries([(data, meta)], 'FR', system.ids).render()

    def test_nostim(self, connect):
        nodes = {k: Node(v) for k, v in self.pneurons.items()}
        amps = self.idrives
        self.simulate(nodes, amps, connect)

    def test_nodrive(self, connect):
        nodes = {k: Node(v, a=self.a, fs=self.fs) for k, v in self.pneurons.items()}
        amps = {k: self.Adrive for k in self.pneurons.keys()}
        self.simulate(nodes, amps, connect)

    def test_full(self, connect):
        nodes = {k: DrivenNode(v, self.idrives[k], Fdrive=self.Fdrive)
                 for k, v in self.pneurons.items()}
        amps = {k: self.Adrive for k in self.pneurons.keys()}
        self.simulate(nodes, amps, connect)


if __name__ == '__main__':
    tester = TestNodeNetwork()
    tester.main()

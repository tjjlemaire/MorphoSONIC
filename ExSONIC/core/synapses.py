# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-14 15:49:25
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-01-14 17:38:45

from neuron import h


class Synapse:

    def __init__(self, Vthr=0., delay=0.):
        ''' Constructor.

            :param Vthr: pre-synaptic voltage threshold triggering transmission (mV)
            :param delay: synaptic delay (ms)
        '''
        self.Vthr = Vthr
        self.delay = delay


class Exp2Synapse(Synapse):
    ''' Bi-exponential, two-state kinetic synapse model. '''

    def __init__(self, tau1, tau2, E, *args, **kwargs):
        ''' Constructor.

            :param tau1: rise time constant (ms)
            :param tau2: decay time constant (ms)
            :param E: reversal potential (mV)
        '''
        self.tau1 = tau1
        self.tau2 = tau2
        self.E = E
        super().__init__(*args, **kwargs)

    def assign(self, node):
        ''' Assign synapse model to a specific target node.

            :param node: target node
            :return: synapse object
        '''
        syn = h.Exp2Syn(node.section(0.5))
        syn.tau1 = self.tau1
        syn.tau2 = self.tau2
        syn.e = self.E
        return syn


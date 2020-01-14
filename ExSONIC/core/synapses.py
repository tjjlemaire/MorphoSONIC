# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-14 15:49:25
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-01-14 17:07:04

from neuron import h

class Exp2Synapse:
    ''' Bi-exponential, two-state kinetic synapse model. '''

    def __init__(self, tau1, tau2, E):
        ''' Constructor.

            :param tau1: rise time constant (ms)
            :param tau2: decay time constant (ms)
            :poram E: reversal potential (mV)
        '''
        self.tau1 = tau1
        self.tau2 = tau2
        self.E = E

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


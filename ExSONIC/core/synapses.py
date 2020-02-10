# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-14 15:49:25
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-05 18:18:40

import abc
from neuron import h

from ..utils import getNmodlDir
from .pyhoc import load_mechanisms

nmodl_dir = getNmodlDir()
for syn_mod_file in ['fexp2syn.mod', 'fdexp2syn.mod']:
    load_mechanisms(nmodl_dir, syn_mod_file)


class Synapse(metaclass=abc.ABCMeta):
    '''Generic synapse model. '''

    @property
    @abc.abstractmethod
    def hoc_obj(self):
        raise NotImplementedError

    def __init__(self, Vthr=0., delay=1.):
        ''' Constructor.

            :param Vthr: pre-synaptic voltage threshold triggering transmission (mV)
            :param delay: synaptic delay (ms)
        '''
        self.Vthr = Vthr
        self.delay = delay

    def attach(self, node):
        ''' Attach synapse model to a specific target node.

            :param node: target node
            :return: synapse object
        '''
        syn = self.hoc_obj(node.section(0.5))
        return syn


class Exp2Synapse(Synapse):
    ''' Bi-exponential, two-state kinetic synapse model. '''

    hoc_obj = h.Exp2Syn

    def __init__(self, tau1=1., tau2=1., E=0., **kwargs):
        ''' Constructor.

            :param tau1: rise time constant (ms)
            :param tau2: decay time constant (ms)
            :param E: reversal potential (mV)
        '''
        self.tau1 = tau1
        self.tau2 = tau2
        self.E = E
        super().__init__(**kwargs)

    def attach(self, node):
        syn = super().attach(node)
        syn.tau1 = self.tau1
        syn.tau2 = self.tau2
        syn.e = self.E
        return syn


class FExp2Synapse(Exp2Synapse):
    ''' Bi-exponential synapse model with short-term facilitation. '''

    hoc_obj = h.FExp2Syn

    def __init__(self, f=1., tauF=1.0, **kwargs):
        ''' Constructor.

            :param f: short-term synaptic facilitation factor (-)
            :param tauF: short-term synaptic facilitation time constant (ms)
        '''
        self.f = f
        self.tauF = tauF
        super().__init__(**kwargs)

    def attach(self, node):
        syn = super().attach(node)
        syn.f = self.f
        syn.tauF = self.tauF
        return syn


class FDExp2Synapse(FExp2Synapse):
    ''' Bi-exponential synapse model with short-term facilitation
        and short-term and long-term depression.
    '''

    hoc_obj = h.FDExp2Syn

    def __init__(self, d1=1.0, tauD1=1.0, d2=1.0, tauD2=1.0, **kwargs):
        ''' Constructor.

            :param d1: short-term synaptic depression factor (-)
            :param tauD1: short-term synaptic depression time constant (ms)
            :param d2: long-term synaptic depression factor (-)
            :param tauD2: short-term synaptic depression time constant (ms)
        '''
        self.d1 = d1
        self.tauD1 = tauD1
        self.d2 = d2
        self.tauD2 = tauD2
        super().__init__(**kwargs)

    def attach(self, node):
        syn = super().attach(node)
        syn.d1 = self.d1
        syn.tauD1 = self.tauD1
        syn.d2 = self.d2
        syn.tauD2 = self.tauD2
        return syn


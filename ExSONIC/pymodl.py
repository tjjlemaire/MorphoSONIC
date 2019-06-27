# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-03-18 21:17:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-17 10:56:34

import inspect
import re
from time import gmtime, strftime

from PySONIC.constants import FARADAY, Rg


def escaped_pow(x):
    return ' * '.join([x.group(1)] * int(x.group(2)))


class NmodlGenerator:

    tabreturn = '\n   '
    NEURON_protected_vars = ['O', 'C']

    def __init__(self, pneuron):
        self.pneuron = pneuron
        self.translated_states = [self.translateState(s) for s in self.pneuron.states]

    def allBlocks(self):
        return '\n\n'.join([
            self.title(),
            self.description(),
            self.constants(),
            self.tscale(),
            self.neuron_block(),
            self.parameter_block(),
            self.state_block(),
            self.assigned_block(),
            self.function_tables(),
            self.initial_block(),
            self.breakpoint_block(),
            self.derivative_block()
        ])

    def print(self):
        print(self.allBlocks())

    def dump(self, outfile):
        with open(outfile, "w") as fh:
            fh.write(self.allBlocks())

    def translateState(self, state):
        return '{}{}'.format(state, '1' if state in self.NEURON_protected_vars else '')

    def title(self):
        return 'TITLE {} membrane mechanism'.format(self.pneuron.name)

    def description(self):
        return '\n'.join([
            'COMMENT',
            self.pneuron.getDesc(),
            '',
            '@Author: Theo Lemaire, EPFL',
            '@Date: {}'.format(strftime("%Y-%m-%d", gmtime())),
            '@Email: theo.lemaire@epfl.ch',
            'ENDCOMMENT'
        ])

    def constants(self):
        block = [
            'FARADAY = {:.5e}     (coul)     : moles do not appear in units'.format(FARADAY),
            'R = {:.5e}         (J/mol/K)  : Universal gas constant'.format(Rg)
        ]
        return 'CONSTANT {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def tscale(self):
        return 'INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}'

    def neuron_block(self):
        block = [
            'SUFFIX {}'.format(self.pneuron.name),
            '',
            ': Constituting currents',
            *['NONSPECIFIC_CURRENT {}'.format(i) for i in self.pneuron.getCurrentsNames()],
            '',
            ': RANGE variables',
            'RANGE Adrive, Vmeff : section specific',
            'RANGE stimon : common to all sections (but set as RANGE to be accessible from caller)'
        ]
        return 'NEURON {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def parameter_block(self):
        block = [
            ': Parameters set by python/hoc caller',
            'stimon : Stimulation state',
            'Adrive (kPa) : Stimulation amplitude',
            '',
            ': Membrane properties',
            'cm = {} (uF/cm2)'.format(self.pneuron.Cm0 * 1e2)
        ]

        # Reversal potentials
        possibles_E = list(set(['Na', 'K', 'Ca'] + [i[1:] for i in self.pneuron.getCurrentsNames()]))
        for x in possibles_E:
            nernst_pot = 'E{}'.format(x)
            if hasattr(self.pneuron, nernst_pot):
                block.append('{} = {} (mV)'.format(
                    nernst_pot, getattr(self.pneuron, nernst_pot)))

        # Conductances / permeabilities
        for i in self.pneuron.getCurrentsNames():
            suffix = '{}{}'.format(i[1:], '' if 'Leak' in i else 'bar')
            factors = {'g': 1e-4, 'p': 1e2}
            units = {'g': 'S/cm2', 'p': 'cm/s'}
            for prefix in ['g', 'p']:
                attr = '{}{}'.format(prefix, suffix)
                if hasattr(self.pneuron, attr):
                    val = getattr(self.pneuron, attr) * factors[prefix]
                    block.append('{} = {} ({})'.format(attr, val, units[prefix]))

        return 'PARAMETER {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def state_block(self):
        block = [': Standard gating states', *self.translated_states]
        return 'STATE {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def assigned_block(self):
        block = [
            ': Variables computed during the simulation and whose value can be retrieved',
            'Vmeff (mV)',
            'v (mV)',
            *['{} (mA/cm2)'.format(i) for i in self.pneuron.getCurrentsNames()]
        ]
        return 'ASSIGNED {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def function_tables(self):
        block = [
            ': Function tables to interpolate effective variables',
            'FUNCTION_TABLE V(A(kPa), Q(nC/cm2)) (mV)',
            *['FUNCTION_TABLE {}(A(kPa), Q(nC/cm2)) (mV)'.format(r) for r in self.pneuron.rates]
        ]
        return '\n'.join(block)

    def initial_block(self):
        block = [': Set initial states values']
        for s in self.pneuron.states:
            if s in self.pneuron.getGates():
                block.append('{0} = alpha{1}(0, v) / (alpha{1}(0, v) + beta{1}(0, v))'.format(
                    self.translateState(s), s.lower()))
            else:
                block.append('{} = ???'.format(self.translateState(s)))

        return 'INITIAL {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def breakpoint_block(self):
        block = [
            ': Integrate states',
            'SOLVE states METHOD cnexp',
            '',
            ': Compute effective membrane potential',
            'Vmeff = V(Adrive * stimon, v)',
            '',
            ': Compute ionic currents'
        ]
        for i in self.pneuron.getCurrentsNames():
            func_exp = inspect.getsource(getattr(self.pneuron, i)).splitlines()[-1]
            func_exp = func_exp[func_exp.find('return') + 7:]
            func_exp = func_exp.replace('self.', '').replace('Vm', 'Vmeff')
            func_exp = re.sub(r'([A-Za-z][A-Za-z0-9]*)\*\*([0-9])', escaped_pow, func_exp)
            block.append('{} = {}'.format(i, func_exp))

        return 'BREAKPOINT {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def derivative_block(self):
        block = [': Gating states derivatives']
        for s in self.pneuron.states:
            if s in self.pneuron.getGates():
                block.append(
                    '{0}\' = alpha{1}{2} * (1 - {0}) - beta{1}{2} * {0}'.format(
                        self.translateState(s), s.lower(), '(Adrive * stimon, v)')
                )
            else:
                block.append('{}\' = ???'.format(self.translateState(s)))

        return 'DERIVATIVE states {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

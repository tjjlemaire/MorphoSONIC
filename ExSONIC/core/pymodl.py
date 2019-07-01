# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-03-18 21:17:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-01 18:30:44

import pprint
import inspect
import re
from time import gmtime, strftime

from PySONIC.constants import getConstantsDict
from PySONIC.core import PointNeuronTranslator


class NmodlTranslator(PointNeuronTranslator):

    tabreturn = '\n   '
    NEURON_protected_vars = ['O', 'C']
    func_table_pattern = 'FUNCTION_TABLE {}(A(kPa), Q(nC/cm2)) ({})'
    current_pattern = 'NONSPECIFIC_CURRENT {} : {}'
    conductance_pattern = '(g)([A-Za-z0-9_]*)(Leak|bar)'
    reversal_potential_pattern = '(E)([A-Za-z0-9_]+)'
    time_constant_pattern = '(tau)([A-Za-z0-9_]+)'
    rate_constant_pattern = '(k)([0-9_]+)'
    ion_concentration_pattern = '(Cai|Nai)([A-Za-z0-9_]*)'

    def __init__(self, pclass):
        super().__init__(pclass, verbose=False)
        self.params = {}
        self.constants = {}
        self.ftables_dict = {'V': self.func_table_pattern.format('V', 'mV')}
        self.dstates_str = self.parseDerStates()
        self.sstates_str = self.parseSteadyStates()
        self.currents_desc = {}
        self.currents = self.parseCurrents()

    @classmethod
    def translateState(cls, state):
        return '{}{}'.format(state, '1' if state in cls.NEURON_protected_vars else '')

    @staticmethod
    def escapedPow(x):
        return ' * '.join([x.group(1)] * int(x.group(2)))

    @classmethod
    def replacePowerExponents(cls, expr):
        return re.sub(r'([A-Za-z][A-Za-z0-9]*)\*\*([0-9])', cls.escapedPow, expr)

    @staticmethod
    def getDocstring(func):
        return inspect.getdoc(func).replace('\n', ' ').strip()

    @staticmethod
    def funcTableExpr(fname, fargs):
        return '{}({})'.format(fname, fargs)

    def addToFuncTables(self, expr):
        ''' Add function table corresponding to function expression '''
        for pattern in [self.alphax_pattern, self.betax_pattern]:
            if pattern.match(expr):
                self.ftables_dict[expr] = self.func_table_pattern.format(expr, '/ms')
        if self.taux_pattern.match(expr):
            self.ftables_dict[expr] = self.func_table_pattern.format(expr, 'ms')
        if self.xinf_pattern.match(expr):
            self.ftables_dict[expr] = self.func_table_pattern.format(expr, '')

    def addToParameters(self, s):
        ''' Add MOD parameters for each class attribute and constants used in Python expression. '''
        class_attr_matches  = self.getClassAttributeCalls(s)
        attrs = {}
        for m in class_attr_matches:
            attr_name = m.group(2)
            attrs[attr_name] = getattr(self.pclass, attr_name)
        for attr_name, attr_val in attrs.items():
            if not inspect.isroutine(attr_val):
                if re.match(self.conductance_pattern, attr_name):
                    self.params[attr_name] = {'val': attr_val * 1e-4, 'unit': 'S/cm2'}
                elif re.match(self.reversal_potential_pattern, attr_name):
                    self.params[attr_name] = {'val': attr_val, 'unit': 'mV'}
                elif re.match(self.time_constant_pattern, attr_name):
                    self.params[attr_name] = {'val': attr_val * 1e3, 'unit': 'ms'}
                elif re.match(self.rate_constant_pattern, attr_name):
                    self.params[attr_name] = {'val': attr_val * 1e-3, 'unit': '/ms'}
                elif re.match(self.ion_concentration_pattern, attr_name):
                    self.params[attr_name] = {'val': attr_val, 'unit': 'M'}
                else:
                    self.params[attr_name] = {'val': attr_val, 'unit': ''}

    def addToConstants(self, s):
        for k, v in getConstantsDict().items():
            if k in s:
                self.constants[k] = v

    def translateExpr(self, expr, lkp_args=None, desc_dict=None):
        ''' Translate Python expression into MOD expression, by parsing all
            internal function calls recursively.
        '''

        # Add potential MOD parameters if class atributes were used in expression
        self.addToParameters(expr)
        self.addToConstants(expr)

        # Replace states getters (x['...']) by MOD translated states
        matches = re.finditer(r"(x\[')([A-Za-z0-9_]+)('\])", expr)
        for m in matches:
            left, state, right = m.groups()
            expr = expr.replace("{}{}{}".format(left, state, right), self.translateState(state))

        # Get all function calls in expression
        matches = self.getFuncCalls(expr)

        # For each function call
        for m in matches:

            # Get function name and arguments
            fcall, fname, fargs = self.getFuncArgs(m)

            # If sole argument is Vm and lookup replacement mode is active
            if lkp_args is not None and len(fargs) == 1 and fargs[0] == 'Vm':
                # Add lookup to the list of function tables
                self.addToFuncTables(fname)

                # Replace function call by equivalent function table call
                new_fcall = self.funcTableExpr(fname, lkp_args)
                expr = expr.replace(fcall, new_fcall)

            # If numpy function, remove "np" prefix assuming that function is in MOD library
            elif fcall.startswith('np.'):
                expr = expr.replace(fcall, fcall.split('np.')[1])

            # Otherwise
            else:
                # Get function object
                func = getattr(self.pclass, fname)

                # If description mode active
                if desc_dict is not None:
                    # Add function docstring fo description dictionary
                    desc_dict[fname] = self.getDocstring(func)

                # Get function source code lines
                func_lines = inspect.getsource(func).split("'''", 2)[-1].splitlines()
                code_lines = []
                for line in func_lines:
                    stripped_line = line.strip()
                    if len(stripped_line) > 0:
                        if not any(stripped_line.startswith(x) for x in ['@', 'def']):
                            code_lines.append(stripped_line)

                # If function contains multiple statements, raise error
                if len(code_lines) > 1 and not code_lines[0].startswith('return'):
                    raise ValueError('cannot parse multi-statement function {}'.format(fname))

                # Join lines into new nested expression and remove comments
                func_exp = ''.join(code_lines).split('return ', 1)[1].split('#', 1)[0].strip()

                # Replace arguments from function signature by their caler name in return expression
                sig_fargs = list(inspect.signature(func).parameters.keys())
                if len(sig_fargs) != len(fargs):
                    raise ValueError(
                        f'number of argumens not matching function signature: {fargs} {sig_fargs}')
                for arg, sig_arg in zip(fargs, sig_fargs):
                    if arg != sig_arg:
                        print(f'replacing {sig_arg} by {arg} in {fname} expression')
                        func_exp = func_exp.replace(sig_arg, arg)

                # Translate internal calls in nested expression recursively
                expr = expr.replace(fcall, '({})'.format(self.translateExpr(func_exp, lkp_args=lkp_args)))

        # Add potential MOD parameters if class atributes were used in expression
        self.addToParameters(expr)

        # Remove comments and strip off all references to class or instance
        expr = expr.replace('self.', '').replace('cls.', '').split('#', 1)[0].strip()

        # Replace integer power exponents by multiplications
        expr = self.replacePowerExponents(expr)

        # Return expression
        return expr

    def parseDerStates(self):
        ''' Parse neuron's derStates method to construct adapted DERIVATIVE block. '''
        dstates_str = self.parseLambdaDict(
            self.pclass.derStates(),
            lambda *args: self.translateExpr(*args, lkp_args='Adrive * stimon, v'))
        if self.verbose:
            print('---------- derStates ----------')
            pprint.PrettyPrinter(indent=4).pprint(dstates_str)
            print('---------- function tables ----------')
            pprint.PrettyPrinter(indent=4).pprint(self.ftables_dict)
        return dstates_str

    def parseSteadyStates(self):
        ''' Parse neuron's steadyStates method to construct adapted INITIAL block. '''
        sstates_str = self.parseLambdaDict(
            self.pclass.steadyStates(),
            lambda *args: self.translateExpr(*args, lkp_args='0, v'))
        if self.verbose:
            print('---------- steadyStates ----------')
            pprint.PrettyPrinter(indent=4).pprint(sstates_str)
        return sstates_str

    def parseCurrents(self):
        ''' Parse neuron's currents method to construct adapted BREAKPOINT block. '''
        currents_str = self.parseLambdaDict(
            self.pclass.currents(),
            lambda *args: self.translateExpr(*args, desc_dict=self.currents_desc))
        if self.verbose:
            print('---------- currents ----------')
            pprint.PrettyPrinter(indent=4).pprint(currents_str)
        return currents_str

    def title(self):
        return 'TITLE {} membrane mechanism'.format(self.pclass.name)

    def description(self):
        return '\n'.join([
            'COMMENT',
            'Equations governing the effective membrane dynamics of a {}'.format(self.pclass.description()),
            'upon electrical / ultrasonic stimulation, based on the SONIC model.',
            '',
            'Reference: Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019).',
            'Understanding ultrasound neuromodulation using a computationally efficient',
            'and interpretable model of intramembrane cavitation. J. Neural Eng.',
            '',
            '@Author: Theo Lemaire, EPFL',
            '@Date: {}'.format(strftime("%Y-%m-%d", gmtime())),
            '@Email: theo.lemaire@epfl.ch',
            'ENDCOMMENT'
        ])

    def constants_block(self):
        block = [f'{k} = {v}' for k , v in self.constants.items()]
        return 'CONSTANT {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def tscale(self):
        return 'INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}'

    def neuronBlock(self):
        block = [
            'SUFFIX {}'.format(self.pclass.name),
            '',
            *[self.current_pattern.format(k, v) for k, v in self.currents_desc.items()],
            '',
            'RANGE Adrive, Vm : section specific',
            'RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)'
        ]
        return 'NEURON {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def parameterBlock(self):
        block = [
            'stimon       : Stimulation state',
            'Adrive (kPa) : Stimulation amplitude',
            '',
            'cm = {} (uF/cm2)'.format(self.pclass.Cm0 * 1e2)
        ]
        for k, v in self.params.items():
            block.append('{} = {} ({})'.format(k, v['val'], v['unit']))

        return 'PARAMETER {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def stateBlock(self):
        block = ['{} : {}'.format(self.translateState(name), desc)
                 for name, desc in self.pclass.states.items()]
        return 'STATE {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def assignedBlock(self):
        block = [
            'v  (nC/cm2)',
            'Vm (mV)',
            *['{} (mA/cm2)'.format(k) for k in self.currents_desc.keys()]
        ]
        return 'ASSIGNED {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def function_tables(self):
        return '\n'.join(self.ftables_dict.values())

    def initialBlock(self):
        block = ['{} = {}'.format(k, v) for k, v in self.sstates_str.items()]
        return 'INITIAL {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def breakpointBlock(self):
        block = [
            'SOLVE states METHOD cnexp',
            'Vm = V(Adrive * stimon, v)'
        ]
        for k, v in self.currents.items():
            block.append('{} = {}'.format(k, v))

        return 'BREAKPOINT {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def derivativeBlock(self):
        block = ['{} = {}'.format(k, v) for k, v in self.dstates_str.items()]
        return 'DERIVATIVE states {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def allBlocks(self):
        return '\n\n'.join([
            self.title(),
            self.description(),
            self.constants_block(),
            self.tscale(),
            self.neuronBlock(),
            self.parameterBlock(),
            self.stateBlock(),
            self.assignedBlock(),
            self.function_tables(),
            self.initialBlock(),
            self.breakpointBlock(),
            self.derivativeBlock()
        ])

    def print(self):
        print(self.allBlocks())

    def dump(self, outfile):
        with open(outfile, "w") as fh:
            fh.write(self.allBlocks())

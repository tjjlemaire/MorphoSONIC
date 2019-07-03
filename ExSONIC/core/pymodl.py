# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-03-18 21:17:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-03 18:22:04

import pprint
import os
import inspect
import re
from time import gmtime, strftime

from PySONIC.constants import getConstantsDict
from PySONIC.core import PointNeuronTranslator


class NmodlTranslator(PointNeuronTranslator):

    tabreturn = '\n   '
    NEURON_protected_vars = ['O', 'C']
    func_pattern_short = '([a-z_A-Z]+\.)?([a-z_A-Z][a-z_A-Z0-9]*)\('
    AQ_func_table_pattern = 'FUNCTION_TABLE {}(A(kPa), Q(nC/cm2)) ({})'
    current_pattern = 'NONSPECIFIC_CURRENT {} : {}'
    conductance_pattern = '(g)([A-Za-z0-9_]*)(Leak|bar)'
    reversal_potential_pattern = '(E)([A-Za-z0-9_]+)'
    time_constant_pattern = '(tau)([A-Za-z0-9_]+)'
    rate_constant_pattern = '(k)([0-9_]+)'
    ion_concentration_pattern = '(Cai|Nai)([A-Za-z0-9_]*)'

    mod_functions = [
        'acos',
        'asin',
        'atan',
        'atan2',
        'ceil',
        'cos',
        'cosh',
        'exp',
        'fabs',
        'floor',
        'fmod',
        'log',
        'log10',
        'pow',
        'sin',
        'sinh',
        'sqrt',
        'tan',
        'tanh'
    ]

    def __init__(self, pclass):
        os.system('cls')
        super().__init__(pclass, verbose=False)
        self.conserved_funcs = []
        self.params = {}
        self.constants = {}
        self.currents_desc = {}
        self.ftables_dict = {'V': self.AQ_func_table_pattern.format('V', 'mV')}
        self.f_dict = {}
        print('------------ parsing currents ------------')
        self.currents_str = self.parseCurrents()
        print('------------ parsing derStates ------------')
        self.dstates_str = self.parseDerStates()
        print('------------ parsing steadyStates ------------')
        self.sstates_str = self.parseSteadyStates()
        print('------------ replacing func tables ------------')
        self.replaceFuncTables()

    @classmethod
    def translateState(cls, state):
        if state in cls.NEURON_protected_vars:
            print(f'-------------------------------------- REPLACING {state} by {state}1')
            state += '1'
        return state

    def replacePowerExponents(self, expr):
        # Replace explicit integer power exponents by multiplications
        expr = re.sub(
            r'([A-Za-z][A-Za-z0-9_]*)\*\*([0-9]+)',
            lambda x: ' * '.join([x.group(1)] * int(x.group(2))),
            expr)

        # Replace parametrized power exponents (not necessarily integers) by MOD notation
        if '**' in expr:
            self.f_dict['npow'] = {'args': ['x', 'n'], 'expr': 'x^n'}
            expr = re.sub(
                r'([A-Za-z][A-Za-z0-9_]*)\*\*([A-Za-z0-9][A-Za-z0-9_.]*)',
                lambda x: f'npow({x.group(1)}, {x.group(2)})',
                expr)

        return expr

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
                self.ftables_dict[expr] = self.AQ_func_table_pattern.format(expr, '/ms')
        if self.taux_pattern.match(expr):
            self.ftables_dict[expr] = self.AQ_func_table_pattern.format(expr, 'ms')
        if self.xinf_pattern.match(expr):
            self.ftables_dict[expr] = self.AQ_func_table_pattern.format(expr, '')

    def addToFunctions(self, fname, fargs, fexpr, level):
        ''' Add a function to the FUNCTION dictionary. '''
        if fname not in self.f_dict:
            print(f'adding {fname} to FUNCTIONS')
            self.f_dict [fname] = {
                'args': fargs,
                'expr': self.translateExpr(fexpr, level=level + 1)
            }
            self.conserved_funcs.append(fname)

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

    @staticmethod
    def getIndent(level):
        return ''.join(['   '] * level)

    @classmethod
    def getFuncCalls(cls, s):
        ''' Find 1st function call in expression. '''
        return [m for m in re.finditer(cls.func_pattern_short, s)]

    def getFuncArgs(self, m, expr, lkp_args, level):
        indent = self.getIndent(level)
        fprefix, fname = m.groups()
        fcall = fname
        if fprefix:
            fcall = '{}{}'.format(fprefix, fname)
        else:
            fprefix = ''
        print(fcall)
        fclosure = self.getClosure(expr[m.end():])
        fclosure = self.translateExpr(fclosure, lkp_args=lkp_args, level=level + 1)
        print(f'{indent}{fclosure}')
        fcall = f'{fcall}({fclosure})'
        fargs = [x.strip() for x in fclosure.split(',')]
        i = 0
        while i < len(fargs):
            j = fargs[i].find('(')
            if j == -1:
                i += 1
            else:
                try:
                    self.getClosure(fargs[i][j + 1:])
                    i += 1
                except ValueError:
                    fargs[i:i + 2] = [', '.join(fargs[i:i + 2])]

        return fcall, fname, fargs, fprefix


    def translateExpr(self, expr, lkp_args=None, is_current=False, level=0):
        ''' Translate Python expression into MOD expression, by parsing all
            internal function calls recursively.
        '''
        indent = self.getIndent(level)

        # Replace dictionary accessors (xxx['...']) by MOD translated states
        expr = re.sub(
            r"([A-Za-z0-9_]+\[')([A-Za-z0-9_]+)('\])",
            lambda x: self.translateState(x.group(2)),
            expr)

        ## Add potential MOD parameters / constants if they were used in expression
        self.addToParameters(expr)
        self.addToConstants(expr)

        # Remove comments and strip off all references to class or instance
        expr = expr.replace('self.', '').replace('cls.', '').split('#', 1)[0].strip()

        done = False
        while not done:
            print(f'{indent}expression: {expr}')

            # Get all function calls in expression
            matches = self.getFuncCalls(expr)

            # Check if functions are listed as functions that must be preserved in expression
            is_conserved = [m.group(2) in self.conserved_funcs for m in matches]

            if len(matches) == 0:
                print(f'{indent}no function call found -> done')
                done = True
            elif all(is_conserved):
                print(f'{indent}all functions must be preserved -> done')
                done = True
            else:
                # Get first match to function call that is not listed as conserved
                m = matches[next(i for i in range(len(is_conserved)) if not is_conserved[i])]

                # Get function information
                fcall, fname, fargs, fprefix = self.getFuncArgs(m, expr, lkp_args, level + 1)

                # If function belongs to the class
                if hasattr(self.pclass, fname):

                    # If function sole argument is Vm and lookup replacement mode is active
                    if lkp_args is not None and len(fargs) == 1 and fargs[0] == 'Vm':

                        # Add lookup to the list of function tables (and preserved functions)
                        if fname not in self.ftables_dict.keys():
                            print(f'{indent}adding {fname} to FUNCTION TABLES')
                            self.addToFuncTables(fname)
                        self.conserved_funcs.append(fname)

                    # Otherwise
                    else:
                        # Get function object
                        func = getattr(self.pclass, fname)

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
                        fexpr = ''.join(code_lines).split('return ', 1)[1].split('#', 1)[0].strip()

                        # Get arguments from function signature and check match with function call
                        sig_fargs = list(inspect.signature(func).parameters.keys())
                        if len(sig_fargs) != len(fargs):
                            raise ValueError(
                                f'differing number of arguments: {fargs} {sig_fargs}')

                        # Strip of potential starting underscore
                        if fname.startswith('_'):
                            new_fname = fname.split('_')[1]
                            print(f'{indent}replacing {fprefix}{fname} by {new_fname}')
                            expr = expr.replace(f'{fprefix}{fname}', new_fname)
                            fname = new_fname

                        # If function is a current
                        if fname in self.pclass.currents().keys():
                            print(f'{indent}current function')

                            # If in CURRENTS block -> replace by expression
                            # and add function docstring to currents description dictionary
                            if is_current:
                                expr = expr.replace(
                                    fcall, self.translateExpr(fexpr, level=level + 1))
                                self.currents_desc[fname] = self.getDocstring(func)

                            # Otherwise, replace current function by current
                            else:
                                # self.conserved_funcs.append(fname)
                                print(f'{indent}replacing {fcall} by {fname}')
                                expr = expr.replace(fcall, fname)

                        else:
                            # If entire call is a single function, replace by its expression
                            if level == 0 and fcall == expr:
                                expr = expr.replace(
                                    fcall, self.translateExpr(fexpr, level=level + 1))
                            else:
                                # Otherwise, add the function to FUNCTION dictionaries
                                self.addToFunctions(fname, sig_fargs, fexpr, level)

                                # Translate potential state arguments in function call
                                new_fargs = [self.translateState(arg) for arg in fargs]
                                new_fcall = f'{fprefix}{fname}({", ".join(new_fargs)})'
                                print(f'{indent}replacing {fcall} by {new_fcall}')
                                expr = expr.replace(fcall, new_fcall)

                # If function does not belong to the class
                else:

                    # If function in MOD library, keep it in expression
                    if fname in self.mod_functions:
                        if '.' in fcall:  # Remove prefix if any
                            stripped_fcall = fcall.split('.')[1]
                            print(f'{indent}replacing {fcall} by {stripped_fcall}')
                            expr = expr.replace(fcall, stripped_fcall)
                            fcall = stripped_fcall
                        print(f'{indent}{fname} in MOD library -> keeping {fcall} in expression')
                        self.conserved_funcs.append(fname)

                    # Otherwise, assume it is a formatting error, and keep it as a value
                    else:
                        # raise ValueError(f'{fname} not part of MOD library nor neuron class')
                        print(f'{fname} not in MOD library or neuron class -> keeping as value')
                        expr = expr.replace(fcall, fname)

        # Translate remaining states in arithmetic expressions for MOD
        expr = re.sub(
            r"([A-Za-z0-9_]+)(\+|-|/| )",
            lambda x: f'{self.translateState(x.group(1))}{x.group(2)}',
            expr)

        # Replace integer power exponents by multiplications
        expr = self.replacePowerExponents(expr)

        return expr

    def parseDerStates(self):
        ''' Parse neuron's derStates method to construct adapted DERIVATIVE block. '''
        dstates_str = self.parseLambdaDict(
            self.pclass.derStates(),
            lambda *args: self.translateExpr(*args, lkp_args='Adrive * stimon, v'))
        for k in dstates_str.keys():
            dstates_str[self.translateState(k)] = dstates_str.pop(k)
        if self.verbose:
            print('---------- derStates ----------')
            pprint.PrettyPrinter(indent=4).pprint(dstates_str)
        return dstates_str

    def parseSteadyStates(self):
        ''' Parse neuron's steadyStates method to construct adapted INITIAL block. '''
        sstates_str = self.parseLambdaDict(
            self.pclass.steadyStates(),
            lambda *args: self.translateExpr(*args, lkp_args='0, v'))
        for k in sstates_str.keys():
            sstates_str[self.translateState(k)] = sstates_str.pop(k)
        if self.verbose:
            print('---------- steadyStates ----------')
            pprint.PrettyPrinter(indent=4).pprint(sstates_str)
        return sstates_str

    def parseCurrents(self):
        ''' Parse neuron's currents method to construct adapted BREAKPOINT block. '''
        currents_str = self.parseLambdaDict(
            self.pclass.currents(),
            lambda *args: self.translateExpr(*args, is_current=True))
        if self.verbose:
            print('---------- currents ----------')
            pprint.PrettyPrinter(indent=4).pprint(currents_str)
        return currents_str

    def replaceFuncTables(self):
        lkp_args_off = '0, v'
        lkp_args_dynamic = 'Adrive * stimon, v'
        for d, lkp_args in zip(
            [self.dstates_str, self.sstates_str, self.currents_str],
            [lkp_args_dynamic, lkp_args_off, lkp_args_dynamic]):
            for k, expr in d.items():
                matches = self.getFuncCalls(expr)
                f_list = [self.getFuncArgs(m, expr, None, 0) for m in matches]
                for (fcall, fname, fargs, fprefix) in f_list:
                    if fname in self.ftables_dict.keys():
                        # Replace function call by equivalent function table call
                        ftable_call = self.funcTableExpr(fname, lkp_args)
                        print(f'replacing {fcall} by {ftable_call}')
                        expr = expr.replace(fcall, ftable_call)
                d[k] = expr

        for k, fcontent in self.f_dict.items():
            if 'inf' in k:
                lkp_args = '0, v'
            else:
                lkp_args = 'Adrive * stimon, v'
            expr = fcontent['expr']
            matches = self.getFuncCalls(expr)
            f_list = [self.getFuncArgs(m, expr, None, 0) for m in matches]
            for (fcall, fname, fargs, fprefix) in f_list:
                if fname in self.ftables_dict.keys():
                    # Replace function call by equivalent function table call
                    ftable_call = self.funcTableExpr(fname, lkp_args)
                    print(f'replacing {fcall} by {ftable_call}')
                    expr = expr.replace(fcall, ftable_call)
            self.f_dict[k]['expr'] = expr

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
        if len(self.constants) > 0:
            block = [f'{k} = {v}' for k , v in self.constants.items()]
            return 'CONSTANT {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))
        else:
            return None

    def tscale(self):
        return 'INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}'

    def neuronBlock(self):
        block = [
            'SUFFIX {}'.format(self.pclass.name),
            *[self.current_pattern.format(k, v) for k, v in self.currents_desc.items()],
            'RANGE Adrive, Vm : section specific',
            'RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)'
        ]
        return 'NEURON {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def parameterBlock(self):
        block = [
            'stimon       : Stimulation state',
            'Adrive (kPa) : Stimulation amplitude',
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

    def functions(self):
        block = []
        for name, content in self.f_dict.items():
            def_line = f'FUNCTION {name}({", ".join(content["args"])}) {{'
            return_line = f'    {name} = {content["expr"]}'
            closure_line = '}'
            block.append('\n'.join([def_line, return_line, closure_line]))
        return '\n\n'.join(block)

    def initialBlock(self):
        block = ['{} = {}'.format(k, v) for k, v in self.sstates_str.items()]
        return 'INITIAL {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def breakpointBlock(self):
        block = [
            'SOLVE states METHOD cnexp',
            'Vm = V(Adrive * stimon, v)'
        ]
        for k, v in self.currents_str.items():
            block.append('{} = {}'.format(k, v))

        return 'BREAKPOINT {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def derivativeBlock(self):
        block = ['{}\' = {}'.format(k, v) for k, v in self.dstates_str.items()]
        return 'DERIVATIVE states {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def allBlocks(self):
        return '\n\n'.join(filter(None, [
            self.title(),
            self.description(),
            self.constants_block(),
            self.tscale(),
            self.neuronBlock(),
            self.parameterBlock(),
            self.stateBlock(),
            self.assignedBlock(),
            self.function_tables(),
            self.functions(),
            self.initialBlock(),
            self.breakpointBlock(),
            self.derivativeBlock()
        ]))

    def print(self):
        print(self.allBlocks())

    def dump(self, outfile):
        with open(outfile, "w") as fh:
            fh.write(self.allBlocks())

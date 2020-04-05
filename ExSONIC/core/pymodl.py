# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-03-18 21:17:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-05 17:24:58

import logging
import pprint
import re
from time import gmtime, strftime

from PySONIC.constants import getConstantsDict
from PySONIC.core import PointNeuronTranslator
from PySONIC.utils import logger

from ..constants import *


class NmodlTranslator(PointNeuronTranslator):

    # MOD specific formatting
    tabreturn = '\n   '
    NEURON_protected_vars = ['O', 'C']
    AQ_func_table_pattern = 'FUNCTION_TABLE {}(A(kPa), Q(nC/cm2)) ({})'
    current_pattern = 'NONSPECIFIC_CURRENT {} : {}'

    # MOD library components
    mod_functions = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh', 'exp', 'fabs',
                     'floor', 'fmod', 'log', 'log10', 'pow', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']

    # Lookup FUNCTION TABLE arguments
    lkp_func_table_args = 'Adrive * stimon, v'

    # Protected functions that must not be inserted but replaced by their first argument
    protected_funcs = ['findModifiedEq']

    def __init__(self, pclass, verbose=False):
        super().__init__(pclass, verbose=verbose)

        # Initialize class containers
        self.translated_states = [self.translateState(x) for x in self.pclass.statesNames()]
        self.funcs_to_preserve = []
        self.params = {}
        self.constants = {}
        self.currents_desc = {}
        self.ftables_dict = {'V': self.AQ_func_table_pattern.format('V', 'mV')}
        self.functions_dict = {}
        self.initial_priors = {k: [] for k in self.translated_states}

        # Parse neuron class's key lambda dictionary functions
        self.currents_dict = self.parseLambdaDict('currents')
        self.dstates_dict = self.parseLambdaDict('derStates')
        self.sstates_dict = self.parseLambdaDict('steadyStates')

        # Replace function tables in parsed dictionaries
        self.adjustFuncTableCalls()

    @classmethod
    def replace(cls, s, old, new, level=0):
        ''' Replace substring in expression. '''
        logger.debug(f'{cls.getIndent(level)}replacing {old} by {new}')
        return s.replace(old, new)

    def replaceFuncCallByFuncExpr(self, s, fcall, fexpr, level=0):
        ''' Replace a function call by its translated return expression. '''
        return self.replace(
            s, fcall, self.translateExpr(fexpr, level=level + 1), level=level)

    def parseLambdaDict(self, pclass_method_name):
        pclass_method = getattr(self.pclass, pclass_method_name)
        logger.info(f'parsing {pclass_method_name}')
        parsed_lambda_dict = super().parseLambdaDict(
            pclass_method(),
            lambda *args: self.translateExpr(*args, method_name=pclass_method_name))
        for k in parsed_lambda_dict.keys():
            parsed_lambda_dict[self.translateState(k)] = parsed_lambda_dict.pop(k)
        if logger.getEffectiveLevel() <= logging.DEBUG:
            pprint.PrettyPrinter(indent=4).pprint(parsed_lambda_dict)
        return parsed_lambda_dict

    @classmethod
    def translateState(cls, state):
        ''' Translate a state name for MOD compatibility if needed. '''
        if state in cls.NEURON_protected_vars:
            logger.debug(f'REPLACING {state} by {state}1')
            state += '1'
        return state

    @classmethod
    def translateDictAccessedStates(cls, s):
        ''' Replace dictionary accessors (xxx['...']) by MOD translated states. '''
        return re.sub(cls.dict_accessor_pattern, lambda x: cls.translateState(x.group(3)), s)

    @classmethod
    def translateVariableStates(cls, s):
        ''' Translate states found as variables in expression by MOD aliases. '''

        # Identify correctly preceded variables in expression,
        # and replace states by MOD aliases if any
        s = re.sub(
            cls.preceded_variable_pattern,
            lambda x: f'{x.group(1)}{cls.translateState(x.group(2))}',
            s)

        # Do the exact same with correctly followed variables
        s = re.sub(
            cls.followed_variable_pattern,
            lambda x: f'{cls.translateState(x.group(1))}{x.group(2)}',
            s)

        # Return translated expression
        return s

    def translatePowerExponents(self, expr):
        ''' Replace power exponents in expression. '''

        # Replace explicit integer power exponents by multiplications
        expr = re.sub(
            r'({})\*\*({})'.format(self.variable_pattern, self.integer_pattern),
            lambda x: ' * '.join([x.group(1)] * int(x.group(2))),
            expr)

        # If power exponents remain in expression
        if '**' in expr:
            # Add npow function (with MOD power notation) to functions dictionary
            self.functions_dict['npow'] = {'args': ['x', 'n'], 'expr': 'x^n'}

            # Replace both parametrized anf float power exponents by call to npow function
            for pattern in [self.variable_pattern, self.float_pattern]:
                expr = re.sub(
                    r'({})\*\*({})'.format(self.variable_pattern, pattern),
                    lambda x: f'npow({x.group(1)}, {x.group(2)})',
                    expr)

        return expr

    @classmethod
    def getFuncReturnExpr(cls, func):
        ''' Get function return expression merged in one line and stripped from comments. '''

        # Get function source code
        code_lines = cls.getFuncSource(func)

        # Remove comments on all lines
        code_lines = [cls.removeLineComments(cl) for cl in code_lines]

        # If function contains no / multiple return statements, raise error
        n_returns = sum(s.count('return') for s in code_lines)
        if n_returns == 0:
            raise ValueError(f'{func.__name__} does not contain any return statement')
        elif n_returns > 1:
            raise ValueError(f'{func.__name__} contains multiple return statements')

        # Search for line containing the return statement
        iline_return = next(i for i, s in enumerate(code_lines) if s.startswith('return'))

        # Merge all lines below return statement
        new_return_line = ''.join(code_lines[iline_return:])
        code_lines = list(filter(None, code_lines[:iline_return] + [new_return_line]))

        # Remove return statement
        code_lines[-1] = code_lines[-1].split('return ')[-1]

        # Return lines separated by new line character
        return '\n'.join(code_lines)

    def addToFuncTables(self, fname, level=0):
        ''' Add a function table corresponding to function name '''
        logger.debug(f'{self.getIndent(level)}adding {fname} to FUNCTION TABLES')
        if self.alphax_pattern.match(fname):
            self.ftables_dict[fname] = self.AQ_func_table_pattern.format(fname, '/ms')
        elif self.betax_pattern.match(fname):
            self.ftables_dict[fname] = self.AQ_func_table_pattern.format(fname, '/ms')
        elif self.taux_pattern.match(fname):
            self.ftables_dict[fname] = self.AQ_func_table_pattern.format(fname, 'ms')
        elif self.xinf_pattern.match(fname):
            self.ftables_dict[fname] = self.AQ_func_table_pattern.format(fname, '')
        else:
            raise ValueError('expression is not a standard alpha-beta or tau-xinf function')

    def addToFunctions(self, fname, fargs, fexpr, level=0):
        ''' Add a function corresponding to the function name, arguments and expression. '''
        if fname not in self.functions_dict:
            logger.debug(f'adding {fname} to FUNCTIONS')

            # Detect potential local variables
            local_vars = []
            flines = fexpr.splitlines()
            for fl in flines:
                if '=' in fl:
                    local_vars.append(fl.split('=')[0].strip())

            self.functions_dict[fname] = {
                'args': fargs,
                'expr': self.translateExpr(fexpr, level=level + 1),
                'locals': local_vars
            }
            self.funcs_to_preserve.append(fname)

    def addToConstants(self, s):
        ''' Add fields to the MOD constants block. '''
        for k, v in getConstantsDict().items():
            if k in s:
                self.constants[k] = v

    def addToParameters(self, s):
        ''' Add MOD parameters for each class attribute and constants used in Python expression. '''
        for attr_name in self.getClassAttributes(s):
            try:
                attr_val = getattr(self.pclass, attr_name)
                if self.conductance_pattern.match(attr_name):
                    self.params[attr_name] = {'val': attr_val * 1e-4, 'unit': 'S/cm2'}
                elif self.permeability_pattern.match(attr_name):
                    self.params[attr_name] = {'val': attr_val * 1e-4, 'unit': '10 m/ms'}
                elif self.reversal_potential_pattern.match(attr_name):
                    self.params[attr_name] = {'val': attr_val, 'unit': 'mV'}
                elif self.time_constant_pattern.match(attr_name):
                    self.params[attr_name] = {'val': attr_val * S_TO_MS, 'unit': 'ms'}
                elif self.rate_constant_pattern.match(attr_name):
                    self.params[attr_name] = {'val': attr_val / S_TO_MS, 'unit': '/ms'}
                elif self.ion_concentration_pattern.match(attr_name):
                    self.params[attr_name] = {'val': attr_val, 'unit': 'M'}
                elif self.current_to_molar_rate_pattern.match(attr_name):
                    self.params[attr_name] = {'val': attr_val * 10., 'unit': '1e7 mol.m-1.C-1'}
                else:
                    self.params[attr_name] = {'val': attr_val, 'unit': ''}
            except AttributeError:
                raise AttributeError(f'{attr_name} is not a class attribute')

    def adjustFuncTableCalls(self):
        ''' Adjust function table calls to match the corresponding FUNCTION_TABLE signature. '''
        logger.info('adjusting function table calls')

        # Replace calls in dstates, sstates and currents dictionaries.
        for d in [self.dstates_dict, self.sstates_dict, self.currents_dict]:
            for k, expr in d.items():
                matches = self.getFuncCalls(expr)
                f_list = [self.parseFuncFields(m, expr, level=0) for m in matches]
                for (fcall, fname, fargs, fprefix) in f_list:
                    if fname in self.ftables_dict.keys():
                        ftable_call = '{}({})'.format(fname, self.lkp_func_table_args)
                        expr = self.replace(expr, fcall, ftable_call)
                d[k] = expr

        # Replace calls in dynamically defined functions
        for k, fcontent in self.functions_dict.items():
            expr = fcontent['expr']
            matches = self.getFuncCalls(expr)
            f_list = [self.parseFuncFields(m, expr, level=0) for m in matches]
            for (fcall, fname, fargs, fprefix) in f_list:
                if fname in self.ftables_dict.keys():
                    # Replace function call by equivalent function table call
                    ftable_call = '{}({})'.format(fname, self.lkp_func_table_args)
                    expr = self.replace(expr, fcall, ftable_call)
            self.functions_dict[k]['expr'] = expr

    def translateExpr(self, expr, level=0, method_name='none'):
        ''' Parse a Python expression and translate it into an equivalent MOD expression.
            Internal function calsl are parsed and translated recursively.
        '''
        # Get level indent
        indent = self.getIndent(level)

        # Replace dictionary accessors (xxx['...']) by MOD translated states
        expr = self.translateDictAccessedStates(expr)

        # Add potential MOD parameters / constants if they were used in expression
        self.addToParameters(expr)
        self.addToConstants(expr)

        # Remove comments and strip off all references to class or instance
        expr = self.removeClassReferences(expr)
        expr = self.removeLineComments(expr)

        # Get all function calls in expression, and check if they are listed as "preserved"
        matches = self.getFuncCalls(expr)
        to_preserve = [m.group(2) in self.funcs_to_preserve for m in matches]

        # As long as expression contains function calls that must be transformed
        while not (len(matches) == 0 or all(to_preserve)):
            logger.debug(f'{indent}expression: {expr}')

            # Get first match to function call that is not listed as conserved
            m = matches[next(i for i in range(len(to_preserve)) if not to_preserve[i])]

            # Get function information
            fcall, fname, fargs, fprefix = self.parseFuncFields(m, expr, level=level + 1)

            # If function belongs to the class
            if hasattr(self.pclass, fname):

                # If function sole argument is Vm and it is not a current function
                if self.isEffectiveVariable(fname, fargs):

                    # Add function to the list of function tables if not already there
                    if fname not in self.ftables_dict.keys():
                        self.addToFuncTables(fname, level=level)

                    # Add function to the list of preserved functions if not already there
                    if fname not in self.funcs_to_preserve:
                        self.funcs_to_preserve.append(fname)

                # If function has multiple arguments or sinle argument that is not Vm
                else:
                    # Get function object, docstring, signature arguments and expression
                    func = getattr(self.pclass, fname)
                    fdoc = self.getDocstring(func)
                    fexpr = self.getFuncReturnExpr(func)
                    sig_fargs = self.getFuncSignatureArgs(func)

                    # Check that number of signature arguments matches that of function call
                    if len(sig_fargs) != len(fargs):
                        raise ValueError(f'differing number of arguments: {fargs} {sig_fargs}')

                    # Strip off starting underscores in function name, if any
                    if fname.startswith('_'):
                        fname = self.removeStartingUnderscores(fname)

                    # If function is a current
                    if fname in self.pclass.currents().keys():
                        logger.debug(f'{indent}current function')

                        # If function contains multiple lines, raise error
                        if fexpr.count('\n') > 0:
                            raise ValueError('current function cannot contain multiple statements')

                        # Translate corresponding expression
                        fexpr = self.translateExpr(fexpr, level=level + 1)

                        # If currently parsing currents, replace function call by its expression,
                        # and add function to currents dictionary
                        if method_name == 'currents':
                            expr = self.replace(expr, fcall, fexpr, level=level)
                            self.currents_desc[fname] = fdoc

                        # Otherwise
                        else:
                            # Replace function by current assigned variable
                            expr = self.replace(expr, fcall, fname, level=level)

                            # If currently parsing steady states: add required current computation
                            # prior to state computation
                            if method_name == 'steadyStates':
                                self.initial_priors[self.current_key].append({
                                    'assigned': fname,
                                    'expr': fexpr
                                })

                    else:
                        # If entry level and entire call is a single line function
                        if level == 0 and fexpr.count('\n') == 0 and fcall == expr:
                            # Replace function call by its expression
                            expr = self.replaceFuncCallByFuncExpr(expr, fcall, fexpr, level=level)

                        # Otherwise
                        else:
                            # Add function to functions dictionary
                            self.addToFunctions(fname, sig_fargs, fexpr, level=level)

                            # Translate potential state arguments in function call by MOD aliases
                            new_fargs = [self.translateVariableStates(arg) for arg in fargs]
                            new_fcall = f'{fprefix}{fname}({", ".join(new_fargs)})'
                            expr = self.replace(expr, fcall, new_fcall, level=level)

            # If function does not belong to the class
            else:

                # If function in MOD library, keep it in expression
                if fname in self.mod_functions:
                    if '.' in fcall:  # Remove prefix if any
                        stripped_fcall = fcall.split('.')[1]
                        expr = self.replace(expr, fcall, stripped_fcall, level=level)
                        fcall = stripped_fcall
                    logger.debug(f'{indent}{fname} in MOD library -> keeping {fcall} in expression')
                    self.funcs_to_preserve.append(fname)

                # If function is part of protected function, replace it by its first argument
                elif fname in self.protected_funcs:
                    rpl_var = fargs[0]
                    logger.debug(f'{indent}{fname} function is protected -> replaced by {rpl_var}')
                    expr = expr.replace(fcall, rpl_var)

                # Otherwise, assume it is a formatting error, and keep it as a variable
                else:
                    logger.debug(f'{fname} not in MOD library or neuron class -> kept as variable')
                    expr = expr.replace(fcall, fname)

            # Get all function calls in new expression, and check if they are listed as "preserved"
            matches = self.getFuncCalls(expr)
            to_preserve = [m.group(2) in self.funcs_to_preserve for m in matches]

        # Translate remaining states in expression by MOD aliases if needed
        if level == 0:
            expr = self.translateVariableStates(expr)

        # Replace power exponents in expression
        expr = self.translatePowerExponents(expr)

        logger.debug(f'{indent}expression: {expr} -----> done')
        return expr

    def title(self):
        ''' Create the TITLE block of the MOD file. '''
        return 'TITLE {} membrane mechanism'.format(self.pclass.name)

    def description(self):
        ''' Create the description (COMMENT) block of the MOD file. '''
        return '\n'.join([
            'COMMENT',
            f'Equations governing the effective membrane dynamics of a {self.pclass.description()}',
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

    def constantsBlock(self):
        ''' Create the (CONSTANT) block of the MOD file. '''
        if len(self.constants) > 0:
            block = [f'{k} = {v}' for k, v in self.constants.items()]
            return 'CONSTANT {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))
        else:
            return None

    def tscale(self):
        ''' Create the time definition block of the MOD file. '''
        return 'INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}'

    def neuronBlock(self):
        ''' Create the NEURON block of the MOD file. '''
        block = [
            'SUFFIX {}auto'.format(self.pclass.name),
            *[self.current_pattern.format(k, v) for k, v in self.currents_desc.items()],
            'RANGE Adrive, Vm : section specific',
            'RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)'
        ]
        return 'NEURON {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def parameterBlock(self):
        ''' Create the PARAMETER block of the MOD file. '''
        block = [
            'stimon       : Stimulation state',
            'Adrive (kPa) : Stimulation amplitude'
        ]
        for k, v in self.params.items():
            block.append('{} = {} ({})'.format(k, v['val'], v['unit']))
        return 'PARAMETER {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def stateBlock(self):
        ''' Create the STATE block of the MOD file. '''
        block = ['{} : {}'.format(self.translateState(name), desc)
                 for name, desc in self.pclass.states.items()]
        if len(block) == 0:
            return ''
        return 'STATE {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def assignedBlock(self):
        ''' Create the ASSIGNED block of the MOD file. '''
        block = [
            'v  (nC/cm2)',
            'Vm (mV)',
            *['{} (mA/cm2)'.format(k) for k in self.currents_desc.keys()]
        ]
        return 'ASSIGNED {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def function_tables(self):
        ''' Create the FUNCTION_TABLE block of the MOD file. '''
        return '\n'.join(self.ftables_dict.values())

    def functions(self):
        ''' Create the FUNCTION block of the MOD file. '''
        block = []
        for name, content in self.functions_dict.items():
            def_line = f'FUNCTION {name}({", ".join(content["args"])}) {{'
            flines = content['expr'].splitlines()
            if len(content.get('locals', [])) > 0:
                locals_line = 'LOCAL ' + ', '.join(content['locals'])
                flines = [locals_line] + flines
            flines[-1] = f'{name} = {flines[-1]}'
            flines = [f'    {fl}' for fl in flines]
            closure_line = '}'
            block.append('\n'.join([def_line, *flines, closure_line]))
        return '\n\n'.join(block)

    def initialBlock(self):
        ''' Create the INITIAL block of the MOD file. '''
        block = []
        for k in self.translated_states:
            for prior in self.initial_priors[k]:
                block.append('{} = {}'.format(prior['assigned'], prior['expr']))
            block.append('{} = {}'.format(k, self.sstates_dict[k]))
        if len(block) == 0:
            return ''
        return 'INITIAL {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def breakpointBlock(self):
        ''' Create the BREAKPOINT block of the MOD file. '''
        block = ['SOLVE states METHOD cnexp'] if len(self.translated_states) > 0 else []
        block.append('Vm = V(Adrive * stimon, v)')
        for k in self.pclass.currents().keys():
            block.append('{} = {}'.format(k, self.currents_dict[k]))
        return 'BREAKPOINT {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def derivativeBlock(self):
        ''' Create the DERIVATIVE block of the MOD file. '''
        block = ['{}\' = {}'.format(k, self.dstates_dict[k]) for k in self.translated_states]
        if len(block) == 0:
            return ''
        return 'DERIVATIVE states {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def allBlocks(self):
        ''' Create all the blocks of the MOD file in the appropriate order. '''
        return '\n\n'.join(filter(None, [
            self.title(),
            self.description(),
            self.constantsBlock(),
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
        ''' Print the created MOD file on the console. '''
        print('---------------------------------------------------------------------')
        print(self.allBlocks())
        print('---------------------------------------------------------------------')
        print('\n')

    def dump(self, outfile):
        ''' Dump the created MOD file in an output file. '''
        with open(outfile, "w") as fh:
            fh.write(self.allBlocks())

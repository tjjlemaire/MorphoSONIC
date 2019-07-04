# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-03-18 21:17:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-04 17:42:15

import pprint
import inspect
import re
from time import gmtime, strftime

from PySONIC.constants import getConstantsDict
from PySONIC.core import PointNeuronTranslator


class NmodlTranslator(PointNeuronTranslator):

    # Generic regexp patterns
    variable_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*'
    class_attribute_pattern = r'(cls.)({})'.format(variable_pattern)
    strict_func_pattern = r'({})\('.format(variable_pattern)
    loose_func_pattern = r'({}\.)?{}'.format(variable_pattern, strict_func_pattern)

    # Neuron-specific regexp patterns
    conductance_pattern = r'(g)([A-Za-z0-9_]*)(Leak|bar)'
    reversal_potential_pattern = r'(E)([A-Za-z0-9_]+)'
    time_constant_pattern = r'(tau)([A-Za-z0-9_]+)'
    rate_constant_pattern = r'(k)([0-9_]+)'
    ion_concentration_pattern = r'(Cai|Nai)([A-Za-z0-9_]*)'

    # MOD specific formatting
    NEURON_protected_vars = ['O', 'C']
    tabreturn = '\n   '
    AQ_func_table_pattern = 'FUNCTION_TABLE {}(A(kPa), Q(nC/cm2)) ({})'
    current_pattern = 'NONSPECIFIC_CURRENT {} : {}'

    # MOD library components
    mod_functions = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh', 'exp', 'fabs',
                     'floor', 'fmod', 'log', 'log10', 'pow', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']

    def __init__(self, pclass, verbose=False):
        super().__init__(pclass, verbose=verbose)

        # Initialize class containers
        self.conserved_funcs = []
        self.params = {}
        self.constants = {}
        self.currents_desc = {}
        self.ftables_dict = {'V': self.AQ_func_table_pattern.format('V', 'mV')}
        self.functions_dict = {}

        # Parse neuron class's key lambda dictionary functions
        self.currents_dict = self.parseLambdaDict('currents')
        self.dstates_dict = self.parseLambdaDict('derStates')
        self.sstates_dict = self.parseLambdaDict('steadyStates')

        # Replace function tables in parsed dictionaries
        self.adjustFuncTableCalls()

    @classmethod
    def getClassAttributeCalls(cls, s):
        ''' Find attribute calls in expression. '''
        return re.finditer(cls.class_attribute_pattern, s)

    @classmethod
    def getFuncCalls(cls, s):
        ''' Get all function calls in expression. '''
        return [m for m in re.finditer(cls.loose_func_pattern, s)]

    def parseFuncFields(self, m, expr, level=0):
        ''' Parse a function call with all its relevant fields: name, arguments, and prefix. '''
        fprefix, fname = m.groups()
        fcall = fname
        if fprefix:
            fcall = '{}{}'.format(fprefix, fname)
        else:
            fprefix = ''
        fclosure = self.getClosure(expr[m.end():])
        fclosure = self.translateExpr(fclosure, level=level + 1)
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

    def parseLambdaDict(self, pclass_method_name):
        is_current_expr = pclass_method_name == 'currents'
        pclass_method = getattr(self.pclass, pclass_method_name)
        print(f'------------ parsing {pclass_method_name} ------------')
        parsed_lambda_dict = super().parseLambdaDict(
            pclass_method(),
            lambda *args: self.translateExpr(*args, is_current_expr=is_current_expr))
        for k in parsed_lambda_dict.keys():
            parsed_lambda_dict[self.translateState(k)] = parsed_lambda_dict.pop(k)
        if self.verbose:
            print(f'---------- {pclass_method_name} ----------')
            pprint.PrettyPrinter(indent=4).pprint(parsed_lambda_dict)
        return parsed_lambda_dict

    @classmethod
    def translateState(cls, state):
        ''' Translate a state name for MOD compatibility if needed. '''
        if state in cls.NEURON_protected_vars:
            print(f'REPLACING {state} by {state}1')
            state += '1'
        return state

    def replacePowerExponents(self, expr):
        ''' Replace power exponents in expression. '''

        # Replace explicit integer power exponents by multiplications
        expr = re.sub(
            r'([A-Za-z][A-Za-z0-9_]*)\*\*([0-9]+)',
            lambda x: ' * '.join([x.group(1)] * int(x.group(2))),
            expr)

        # If parametrized power exponents (not necessarily integers) are present
        if '**' in expr:
            # Replace them by call to npow function
            expr = re.sub(
                r'([A-Za-z][A-Za-z0-9_]*)\*\*([A-Za-z0-9][A-Za-z0-9_.]*)',
                lambda x: f'npow({x.group(1)}, {x.group(2)})',
                expr)

            # add npow function (with MOD power notation) to functions dictionary
            self.functions_dict['npow'] = {'args': ['x', 'n'], 'expr': 'x^n'}

        return expr

    def addToFuncTables(self, fname, level=0):
        ''' Add a function table corresponding to function name '''
        print(f'{self.getIndent(level)}adding {fname} to FUNCTION TABLES')
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
            print(f'adding {fname} to FUNCTIONS')
            self.functions_dict [fname] = {
                'args': fargs,
                'expr': self.translateExpr(fexpr, level=level + 1)
            }
            self.conserved_funcs.append(fname)

    def addToConstants(self, s):
        ''' Add fields to the MOD constants block. '''
        for k, v in getConstantsDict().items():
            if k in s:
                self.constants[k] = v

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
            else:
                print('aaa')
                # raise ValueError(f'{attr_name} is a class method')

    def adjustFuncTableCalls(self):
        ''' Adjust function table calls to match the corresponding FUNCTION_TABLE signature. '''
        print('------------ adjusting function table calls ------------')
        lkp_args_off = '0, v'
        lkp_args_dynamic = 'Adrive * stimon, v'

        # Replace calls in dstates, sstates and currents dictionaries.
        for d, lkp_args in zip(
            [self.dstates_dict, self.sstates_dict, self.currents_dict],
            [lkp_args_dynamic, lkp_args_off, lkp_args_dynamic]):
            for k, expr in d.items():
                matches = self.getFuncCalls(expr)
                f_list = [self.parseFuncFields(m, expr, level=0) for m in matches]
                for (fcall, fname, fargs, fprefix) in f_list:
                    if fname in self.ftables_dict.keys():
                        # Replace function call by equivalent function table call
                        ftable_call = '{}({})'.format(fname, lkp_args)
                        print(f'replacing {fcall} by {ftable_call}')
                        expr = expr.replace(fcall, ftable_call)
                d[k] = expr

        # Replace calls in dynamically defined functions
        for k, fcontent in self.functions_dict.items():
            if 'inf' in k:
                lkp_args = '0, v'
            else:
                lkp_args = 'Adrive * stimon, v'
            expr = fcontent['expr']
            matches = self.getFuncCalls(expr)
            f_list = [self.parseFuncFields(m, expr, level=0) for m in matches]
            for (fcall, fname, fargs, fprefix) in f_list:
                if fname in self.ftables_dict.keys():
                    # Replace function call by equivalent function table call
                    ftable_call = '{}({})'.format(fname, lkp_args)
                    print(f'replacing {fcall} by {ftable_call}')
                    expr = expr.replace(fcall, ftable_call)
            self.functions_dict[k]['expr'] = expr

    def translateExpr(self, expr, is_current_expr=False, level=0):
        ''' Parse a Python expression and translate it into an equivalent MOD expression.
            Internal function calsl are parsed and translated recursively.
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
                fcall, fname, fargs, fprefix = self.parseFuncFields(m, expr, level=level + 1)

                # If function belongs to the class
                if hasattr(self.pclass, fname):

                    # If function sole argument is Vm and it is not a current function
                    if len(fargs) == 1 and fargs[0] == 'Vm' and fname not in self.pclass.currents().keys():

                        # Add function to the list of function tables if not already there
                        if fname not in self.ftables_dict.keys():
                            self.addToFuncTables(fname, level=level)

                        # Add function to the list of preserved functions if not already there
                        if fname not in self.conserved_funcs:
                            self.conserved_funcs.append(fname)

                    # If function has multiple arguments or sinle argument that is not Vm
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

                        # Strip off starting underscore in function name, if any
                        if fname.startswith('_'):
                            new_fname = fname.split('_')[1]
                            print(f'{indent}replacing {fprefix}{fname} by {new_fname}')
                            expr = expr.replace(f'{fprefix}{fname}', new_fname)
                            fname = new_fname

                        # If function is a current
                        if fname in self.pclass.currents().keys():
                            print(f'{indent}current function')

                            # If current expression
                            if is_current_expr:

                                # Replace current function by its expression
                                expr = expr.replace(
                                    fcall, self.translateExpr(fexpr, level=level + 1))

                                # Add to list of currents, along with description parsed
                                # from function docstring
                                self.currents_desc[fname] = self.getDocstring(func)

                            # Otherwise, replace current function by current assigned value
                            else:
                                # self.conserved_funcs.append(fname)
                                print(f'{indent}replacing {fcall} by {fname}')
                                expr = expr.replace(fcall, fname)

                        else:
                            # If entry level and entire call is a single function
                            if level == 0 and fcall == expr:
                                # Replace funciton call by its expression
                                expr = expr.replace(
                                    fcall, self.translateExpr(fexpr, level=level + 1))

                            # Otherwise
                            else:
                                # Add the function to FUNCTION dictionaries
                                self.addToFunctions(fname, sig_fargs, fexpr, level=level)

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

    def title(self):
        ''' Create the TITLE block of the MOD file. '''
        return 'TITLE {} membrane mechanism'.format(self.pclass.name)

    def description(self):
        ''' Create the description (COMMENT) block of the MOD file. '''
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
        ''' Create the (CONSTANT) block of the MOD file. '''
        if len(self.constants) > 0:
            block = [f'{k} = {v}' for k , v in self.constants.items()]
            return 'CONSTANT {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))
        else:
            return None

    def tscale(self):
        ''' Create the time definition block of the MOD file. '''
        return 'INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}'

    def neuronBlock(self):
        ''' Create the NEURON block of the MOD file. '''
        block = [
            'SUFFIX {}'.format(self.pclass.name),
            *[self.current_pattern.format(k, v) for k, v in self.currents_desc.items()],
            'RANGE Adrive, Vm : section specific',
            'RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)'
        ]
        return 'NEURON {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def parameterBlock(self):
        ''' Create the PARAMETER block of the MOD file. '''
        block = [
            'stimon       : Stimulation state',
            'Adrive (kPa) : Stimulation amplitude',
            'cm = {} (uF/cm2)'.format(self.pclass.Cm0 * 1e2)
        ]
        for k, v in self.params.items():
            block.append('{} = {} ({})'.format(k, v['val'], v['unit']))
        return 'PARAMETER {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def stateBlock(self):
        ''' Create the STATE block of the MOD file. '''
        block = ['{} : {}'.format(self.translateState(name), desc)
                 for name, desc in self.pclass.states.items()]
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
            return_line = f'    {name} = {content["expr"]}'
            closure_line = '}'
            block.append('\n'.join([def_line, return_line, closure_line]))
        return '\n\n'.join(block)

    def initialBlock(self):
        ''' Create the INITIAL block of the MOD file. '''
        block = ['{} = {}'.format(k, v) for k, v in self.sstates_dict.items()]
        return 'INITIAL {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def breakpointBlock(self):
        ''' Create the BREAKPOINT block of the MOD file. '''
        block = [
            'SOLVE states METHOD cnexp',
            'Vm = V(Adrive * stimon, v)'
        ]
        for k, v in self.currents_dict.items():
            block.append('{} = {}'.format(k, v))
        return 'BREAKPOINT {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def derivativeBlock(self):
        ''' Create the DERIVATIVE block of the MOD file. '''
        block = ['{}\' = {}'.format(k, v) for k, v in self.dstates_dict.items()]
        return 'DERIVATIVE states {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def allBlocks(self):
        ''' Create all the blocks of the MOD file in the appropriate order. '''
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
        ''' Print the created MOD file on the console. '''
        print(self.allBlocks())

    def dump(self, outfile):
        ''' Dump the created MOD file in an output file. '''
        with open(outfile, "w") as fh:
            fh.write(self.allBlocks())

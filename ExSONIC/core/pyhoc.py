# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-19 15:04:35
# @Author: Theo Lemaire
# @Date:   2018-08-21 19:48:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-01-11 10:20:22

''' Utilities to manipulate HOC objects. '''

import os
import abc
import numpy as np
from neuron import h, hclass

from PySONIC.utils import logger


class Probe(hclass(h.Vector)):
    ''' Interface to a Hoc vector recording a particular variable. '''

    def __new__(cls, variable):
        ''' Instanciation. '''
        return super(Probe, cls).__new__(cls)

    def __init__(self, variable):
        ''' Initialization. '''
        super().__init__()
        self.record(variable)

    def to_array(self):
        ''' Return itself as a numpy array. '''
        return np.array(self.to_python())


class Matrix(hclass(h.Matrix)):
    ''' Interface to Hoc Matrix with facilitated initialization from 2D numpy array. '''

    def __new__(cls, arr):
        ''' Instanciation. '''
        return super(Matrix, cls).__new__(cls, *arr.shape)

    def __init__(self, arr):
        ''' Initialization. '''
        super().__init__(arr)
        for i, row in enumerate(arr):
            self.setrow(i, h.Vector(row))

    def to_array(self):
        ''' Return itself as a numpy array. '''
        return np.array([self.getrow(i).to_python() for i in range(self.nrow())])


class Section(hclass(h.Section)):
    ''' Interface to a Hoc Section with associated point-neuron mechanism. '''

    # Aliases for NMODL-protected variable names
    NEURON_aliases = {'O': 'O1', 'C': 'C1'}
    stimon_var = 'stimon'

    def __init__(self, mechname, mechstates, name=None, cell=None):
        ''' Initialization.

            :param mechname: mechanism name
            :param mechstates: list of mechanism states
        '''
        if cell is not None:
            super().__init__(name=name, cell=cell)
        else:
            super().__init__(name=name)
        self.mechname = mechname
        self.mechstates = mechstates
        self.assign()

    def assign(self):
        self.insert(self.mechname)

    def alias(self, state):
        ''' Return NEURON state alias. '''
        return self.NEURON_aliases.get(state, state)

    def setStimON(self, value):
        ''' Set stimulation ON or OFF.

            :param value: new stimulation state (0 = OFF, 1 = ON)
        '''
        setattr(self, f'{self.stimon_var}_{self.mechname}', value)

    def setProbe(self, var, loc=0.5):
        ''' Set recording vector for a range variable in a specific section location.

            :param var: range variable to record
            :return: recording probe object
        '''
        return Probe(getattr(self(loc), f'_ref_{var}'))

    def setMechProbe(self, var, loc=0.5):
        ''' Set recording vector for a mechanism specific variable. '''
        return self.setProbe(f'{var}_{self.mechname}', loc=loc)

    def setStimProbe(self):
        ''' Set recording vector for stimulation state. '''
        return self.setMechProbe(self.stimon_var)

    def setProbesDict(self):
        ''' Set recording vectors for all variables of interest in the section. '''
        return {
            'Qm': self.setProbe('v'),
            'Vm': self.setMechProbe('Vm'),
            **{k: self.setMechProbe(self.alias(k)) for k in self.mechstates}
        }

    def setMechValue(self, key, value):
        setattr(self, f'{key}_{self.mechname}', value)

    def insertVext(self, xr=1e20, xg=1e10, xc=0.):
        ''' Insert extracellular mechanism and set appropriate parameters.

            :param sec: section object
            :param xr: axial resistance of extracellular layer (Mohms/cm)
            :param xg: transverse conductance of extracellular layer (S/cm^2)
            :param xc: transverse capacitance of extracellular layer (uF/cm^2)
        '''
        self.insert('extracellular')
        self.xraxial[0] = xr
        self.xg[0] = xg
        self.xc[0] = xc


class IClamp(hclass(h.IClamp)):
    ''' IClamp object that allows setting parameters on creation. '''

    def __init__(self, segment, amplitude):
        super().__init__(segment)
        self.delay = 0  # we want to exert control over amp starting at 0 ms
        self.dur = 1e9  # dur must be long enough to span all our changes
        self.amp = 0.  # initially, we set the amplitude to zero
        self.xamp = amplitude

    def toggle(self, value):
        self.amp = value * self.xamp


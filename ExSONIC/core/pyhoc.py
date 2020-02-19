# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-19 22:08:10
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
        self.nseg = 1
        self.assign()

    def setGeometry(self, diam, L):
        ''' Set section diameter and length (provided in m). '''
        self.diam = diam * 1e6  # um
        self.L = L * 1e6        # um

    def setResistivity(self, value):
        self.Ra = value

    def membraneArea(self):
        ''' Compute section membrane surface area (in cm2) '''
        return np.pi * (self.diam * 1e-4) * (self.L * 1e-4)

    def axialArea(self):
        ''' Compute section axial area (in cm2) '''
        return np.pi * (self.diam * 1e-4)**2 / 4

    def axialResistance(self):
        ''' Compute section axial resistance (in Ohm) '''
        return self.Ra * (self.L * 1e-4) / self.axialArea()

    def setValue(self, key, value, x=None):
        setattr(self.target(x), key, value)

    def getValue(self, key, x=None):
        return getattr(self.target(x), key)

    def setMechValue(self, key, *args, **kwargs):
        self.setValue(f'{key}_{self.mechname}', *args, **kwargs)

    def getMechValue(self, key, *args, **kwargs):
        return self.getValue(f'{key}_{self.mechname}', *args, **kwargs)

    def assign(self):
        self.insert(self.mechname)

    def alias(self, state):
        ''' Return NEURON state alias. '''
        return self.NEURON_aliases.get(state, state)

    def setStimON(self, value):
        ''' Set stimulation ON or OFF.

            :param value: new stimulation state (0 = OFF, 1 = ON)
        '''
        self.setMechValue(self.stimon_var, value)

    def setProbe(self, var, loc=0.5):
        ''' Set recording vector for a range variable in a specific section location.

            :param var: range variable to record
            :return: recording probe object
        '''
        return Probe(self.getValue(f'_ref_{var}', x=loc))

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

    def target(self, x):
        return self(x) if x is not None else self

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

    def connect(self, parent):
        ''' Connect proximal end to the distal end of a parent section. '''
        super().connect(parent, 1, 0)



class IaxSection(Section):

    def __init__(self, *args, **kwargs):
        ''' Initialization.

            :param rmin: lower bound for axial resistance density (Ohm.cm2)
        '''
        cs, *args = args
        self.vref = cs.vref
        self.rmin = cs.rmin
        self.iax_name = 'Iax'
        self.has_iax = False
        super().__init__(*args, **kwargs)

    def setIaxValue(self, key, *args, **kwargs):
        self.setValue(f'{key}_{self.iax_name}', *args, **kwargs)

    def getIaxValue(self, key, *args, **kwargs):
        return self.getValue(f'{key}_{self.iax_name}', *args, **kwargs)

    def getVrefValue(self, **kwargs):
        return self.getValue(f'_ref_{self.vref}', **kwargs)

    def getIaxRef(self, **kwargs):
        return self.getValue(self.iax_name, **kwargs)

    def boundedAxialResistance(self):
        ''' Compute bounded value for axial resistance to ensure (resistance * membrane area)
            is always above a specific threshold, in order to limit the magnitude of axial currents.

            :return: bounded resistance value (Ohm)
        '''
        Am = self.membraneArea()    # cm2
        R = self.axialResistance()  # Ohm
        s = f'{self}: R*Am = {R * Am:.1e} Ohm.cm2'
        if R < self.rmin / Am:
            s += f' -> bounded to {self.rmin:.1e} Ohm.cm2'
            R = self.rmin / Am
            logger.warning(s)
        else:
            s += ' -> not bounded'
            logger.debug(s)
        return R

    @staticmethod
    def link(mechname, varname, ref_hocvar):
        ''' Set a specific mechanism variable to point dynamically towards another Hoc variable.

            :param mechname: mechanism name containing the pointing variable
            :param varname: name of the pointing variable
            :param ref_hocvar: reference hoc variable to point towards.
        '''
        h.setpointer(ref_hocvar, varname, mechname)

    def insertIax(self):
        ''' Insert axial density mechanism with appropriate parameters. '''

        # Insert axial current density mechanism
        self.insert(self.iax_name)

        # Compute section resistance
        R = self.axialResistance() if self.rmin is None else self.boundedAxialResistance()  # Ohm

        # Set axial mechanism parameters
        self.setIaxValue('R', R)
        self.setIaxValue('Am', self.membraneArea())
        self.link(self.getIaxRef(x=0.5), 'V', self.getVrefValue(x=0.5))

        # While section not connected: set neighboring sections' properties (resistance and
        # membrane potential) as those of current section
        for suffix in ['prev', 'next']:
            self.setValue(f'R{suffix}_{self.iax_name}', R)  # Ohm
            IaxSection.link(self.getIaxRef(x=0.5), f'V{suffix}', self.getVrefValue(x=0.5))

        # Return True to mark successfull mechanism insertion
        return True

    def connect(self, parent):
        ''' Connect to a parent section in series to enable trans-sectional axial current. '''
        for sec in [parent, self]:
            if not sec.has_iax:
                sec.has_iax = sec.insertIax()

        # Inform sections about each other's axial resistance (in Ohm)
        parent.setIaxValue('Rnext', self.getIaxValue('R'))
        self.setIaxValue('Rprev', parent.getIaxValue('R'))

        # Set bi-directional pointers to sections about each other's membrane potential
        IaxSection.link(self.getValue(self.iax_name, x=0.5), 'Vprev', parent.getVrefValue(x=0.5))
        IaxSection.link(parent.getValue(self.iax_name, x=0.5), 'Vnext', self.getVrefValue(x=0.5))

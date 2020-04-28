# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-27 12:36:36

''' Utilities to manipulate HOC objects. '''

import numpy as np
from neuron import h, hclass

from PySONIC.utils import logger
from ..utils import load_mechanisms, getNmodlDir
from ..constants import *

# Declare ground hoc variable
h('{ground = 0}')

load_mechanisms(getNmodlDir())


class Probe(hclass(h.Vector)):
    ''' Interface to a Hoc vector recording a particular variable. '''

    def __new__(cls, variable, factor=1.):
        ''' Instanciation. '''
        return super(Probe, cls).__new__(cls)

    def __init__(self, variable, factor=1.):
        ''' Initialization. '''
        super().__init__()
        self.factor = factor
        self.record(variable)

    def to_array(self):
        ''' Return itself as a numpy array. '''
        return np.array(self.to_python()) * self.factor

    def clear(self):
        super().__init__()


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
    ''' IClamp object that takes section with default relative position as input
        and allows setting parameters on creation.
    '''

    def __new__(cls, section, amplitude, x=0.5):
        ''' Instanciation. '''
        return super(IClamp, cls).__new__(cls, section(x))

    def __init__(self, section, amplitude, x=0.5):
        super().__init__(section)
        self.delay = 0  # we want to exert control over amp starting at 0 ms
        self.dur = 1e9  # dur must be long enough to span all our changes
        self.amp = 0.  # initially, we set the amplitude to zero
        self.xamp = amplitude

    def set(self, value):
        self.amp = value * self.xamp


class ExtField():
    ''' Extracellular field object that allows setting parameters on creation. '''

    def __init__(self, section, amplitude, **kwargs):
        self.section = section
        if not self.section.has_ext_mech:
            self.section.insertVext(**kwargs)
        self.xamp = amplitude
        self.factor = self.section.Cm0 if isinstance(self.section, QSection) else 1.

    def __repr__(self):
        return f'{self.__class__.__name__}({self.section.shortname()}, {self.xamp:.2f} mV)'

    def set(self, value):
        self.section.setVext(self.xamp * self.factor * value)


class Section(hclass(h.Section)):
    ''' Interface to a Hoc Section with nseg=1. '''

    def __init__(self, name=None, cell=None):
        ''' Initialization.

            :param name: section name
            :param cell: section cell
        '''
        if cell is not None:
            super().__init__(name=name, cell=cell)
        else:
            super().__init__(name=name)
        self.nseg = 1
        self.has_ext_mech = False

    def shortname(self):
        s = self.name()
        if '.' in s:
            s = s.split('.')[-1]
        return s

    def setGeometry(self, diam, L):
        ''' Set section diameter and length.

            :param diam: section diameter (m)
            :param L: section length (m)
        '''
        self.diam = diam * M_TO_UM  # um
        self.L = L * M_TO_UM        # um

    def setResistivity(self, value):
        ''' Set section resistivity.

            :param value: longitudinal resistivity (Ohm.cm)
        '''
        self.Ra = value

    def membraneArea(self):
        ''' Compute section membrane surface area.

            :return: membrane area (cm2)
         '''
        return np.pi * self.diam * self.L / CM_TO_UM**2

    def axialArea(self):
        ''' Compute section axial area.

            :return: axial area (cm2)
        '''
        return 1 / 4 * np.pi * self.diam**2 / CM_TO_UM**2

    def axialResistance(self):
        ''' Compute section axial resistance.

            :return: axial resistance (Ohm)
        '''
        return self.Ra * (self.L / CM_TO_UM) / self.axialArea()

    def target(self, x):
        ''' Return x-dependent resolved object (section's self of section's only segment).

            :param x: optional relative position over the section's length (0 <= x <= 1)
            :return: section's self (if x not provided) or of section's only segment (if x provided)
        '''
        return self(x) if x is not None else self

    def setValue(self, key, value, x=None):
        ''' Set the value of a section's (or segment's) attribute.

            :param key: attribute name
            :param value: attribute value
            :param x: relative position over the section's length
        '''
        setattr(self.target(x), key, value)

    def getValue(self, key, x=None):
        ''' Get the value of a section's (or segment's) attribute.

            :param key: attribute name
            :param x: relative position over the section's length
            :return: attribute value
        '''
        return getattr(self.target(x), key)

    def setProbe(self, var, loc=0.5, **kwargs):
        ''' Set recording vector for a range variable in a specific section location.

            :param var: range variable to record
            :return: recording probe object
        '''
        return Probe(self.getValue(f'_ref_{var}', x=loc), **kwargs)

    def insertVext(self, xr=1e20, xg=1e10, xc=0.):
        ''' Insert extracellular mechanism with specific parameters.

            :param xr: axial resistance per unit length of first extracellular layer (Ohm/cm)
            :param xg: transverse conductance of first extracellular layer (S/cm2)
            :param xc: transverse capacitance of first extracellular layer (uF/cm2)
        '''
        self.insert('extracellular')
        self.xraxial[0] = xr * OHM_TO_MOHM  # MOhm/cm
        self.xg[0] = xg              # S/cm2
        self.xc[0] = xc              # S/cm2
        self.has_ext_mech = True

    def setVext(self, Ve):
        ''' Set the extracellular potential just outside of a section. '''
        self.e_extracellular = Ve

    def connect(self, parent):
        ''' Connect proximal end to the distal end of a parent section.

            :param parent: parent sectio object
        '''
        super().connect(parent, 1, 0)

    def insertPassiveMech(self, g, Erev):
        ''' Insert a passive (leakage) mechanism with specifc parameters.

            :param g: conductance (S/cm2)
            :param Erev: reversal potential (mV)
        '''
        self.insert('pas')
        self.g_pas = g
        self.e_pas = Erev

    def getDetails(self):
        ''' Get details of section parameters. '''
        d = {
            'nseg': self.nseg,
            'diam (um)': self.diam,
            'L (um)': self.L,
            'cm (uF/cm2)': self.cm,
            'Ra (Ohm.cm)': self.Ra
        }
        if self.has_ext_mech:
            d.update({
                'xraxial (MOhms)': self.xraxial[0],
                'xg (S/cm2)': self.xg[0],
                'xc (uF/cm2)': self.xc[0]
            })
        return d


class VSection(Section):
    ''' Interface to a Hoc Section with voltage based transmembrane dynamics. '''

    def __init__(self, name=None, cell=None, Cm0=1.):
        ''' Initialization.

            :param mechname: mechanism name
            :param states: list of mechanism time-varying states
            :param Cm0: resting membrane capacitance (uF/cm2)
        '''
        super().__init__(name=name, cell=cell)
        self.Cm0 = Cm0
        self.cm = self.Cm0


class QSection(Section):
    ''' Interface to a Hoc Section with charge-density based transmembrane dynamics.

        In this section type, the differential variable "v" stands for charge density (in nC/cm2).
    '''

    def __init__(self, name=None, cell=None, Cm0=1.):
        ''' Initialization.

            :param mechname: mechanism name
            :param states: list of mechanism time-varying states
            :param Cm0: resting membrane capacitance (uF/cm2)
        '''
        super().__init__(name=name, cell=cell)
        self.Cm0 = Cm0

    def setResistivity(self, value):
        ''' Set corrected section resistivity to account for Q-based differential scheme.

            Since v represents the charge density, resistivity must be multiplied by
            membrane capacitance to ensure consistency of axial currents:
            Iax = dV / r = dQ / (r * cm)

            :param value: longitudinal resistivity (Ohm.cm)
        '''
        super().setResistivity(value * self.Cm0)  # uF.Ohm/cm

    def insertPassiveMech(self, g, Erev):
        ''' Insert a passive (leakage) mechanism with specific parameters.

            The reversal potential is multiplyied by the section's membrane capacitance
            to enable a Q-based synchronization across sections.

            :param g: conductance (S/cm2)
            :param Erev: reversal potential (mV)
        '''
        super().insertPassiveMech(g, Erev * self.Cm0)

    def insertVext(self, **kwargs):
        ''' Insert extracellular mechanism with specific parameters, corrected to account
            for Q-based differential scheme.
        '''
        super().insertVext(**kwargs)
        for i in range(2):
            self.xraxial[i] *= self.Cm0
            self.xc[i] /= self.Cm0
            self.xg[i] /= self.Cm0

    def connect(self, parent):
        ''' Connect two sections together, provided they have the same membrane capacitance. '''
        if self.Cm0 != parent.Cm0:
            raise ValueError('Cannot connect Q-based sections with differing capacitance')
        super().connect(parent)


class MechSection(Section):
    ''' Interface to a section with associated point-neuron mechanism. '''

    # Aliases for NMODL-protected variable names
    NEURON_aliases = {'O': 'O1', 'C': 'C1'}
    stimon_var = 'stimon'

    def __init__(self, mechname=None, states=None, **kwargs):
        ''' Initialization.

            :param mechname: mechanism name
            :param states: list of mechanism time-varying states
        '''
        if kwargs:
            super().__init__(**kwargs)
        self.mechname = mechname
        self.states = states

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, value):
        if value is None:
            value = []
        self._states = value

    def setMechValue(self, key, value, **kwargs):
        ''' Set the value of the section's mechanism attribute.

            :param key: attribute name
            :param value: attribute value
        '''
        self.setValue(f'{key}_{self.mechname}', value, **kwargs)

    def getMechValue(self, key, **kwargs):
        ''' Get the value of an attribute related to the section's mechanism.

            :param key: attribute name
            :return: attribute value
        '''
        return self.getValue(f'{key}_{self.mechname}', **kwargs)

    def alias(self, state):
        ''' Return NEURON state alias.

            :param state: state name
        '''
        return self.NEURON_aliases.get(state, state)

    def setStimON(self, value):
        ''' Set stimulation ON or OFF.

            :param value: new stimulation state (0 = OFF, 1 = ON)
        '''
        self.setMechValue(self.stimon_var, value)

    def setMechProbe(self, var, loc=0.5, **kwargs):
        ''' Set recording vector for a mechanism specific variable. '''
        return self.setProbe(f'{var}_{self.mechname}', loc=loc, **kwargs)

    def setStimProbe(self):
        ''' Set recording vector for stimulation state. '''
        return self.setMechProbe(self.stimon_var)

    def setProbesDict(self):
        ''' Set recording vectors for all variables of interest in the section.

            :return: probes object dictionary
        '''
        return {k: self.setMechProbe(self.alias(k)) for k in self.states}


class MechVSection(MechSection, VSection):
    ''' Interface to a V-based Section with associated point-neuron mechanism. '''

    def __init__(self, **kwargs):
        name = kwargs.pop('name', None)
        cell = kwargs.pop('cell', None)
        Cm0 = kwargs.pop('Cm0', 1.)
        VSection.__init__(self, name=name, cell=cell, Cm0=Cm0)
        MechSection.__init__(self, **kwargs)

    def setProbesDict(self):
        return {
            'Vm': self.setProbe('v'),
            'Qm': self.setMechProbe('Qm', factor=2.),
            **super().setProbesDict()
        }


class MechQSection(MechSection, QSection):
    ''' Interface to a Q-based Section with associated point-neuron mechanism. '''

    def __init__(self, **kwargs):
        name = kwargs.pop('name', None)
        cell = kwargs.pop('cell', None)
        Cm0 = kwargs.pop('Cm0', 1.)
        QSection.__init__(self, name=name, cell=cell, Cm0=Cm0)
        MechSection.__init__(self, **kwargs)

    def setProbesDict(self):
        return {
            'Qm': self.setProbe('v'),
            'Vm': self.setMechProbe('Vm'),
            **super().setProbesDict()
        }


class CustomConnectMechQSection(MechQSection):
    ''' Interface to a Hoc Section with associated point-neuron mechanism. '''

    def __init__(self, cs, *args, **kwargs):
        ''' Initialization.

            :param cs: connector scheme object
        '''
        super().__init__(*args, **kwargs)
        if cs.vref == 'v':
            self.vref = cs.vref
        else:
            self.vref = f'{cs.vref}_{self.mechname}'
        self.rmin = cs.rmin
        self.iax = None

    def setResistivity(self, value):
        ''' Set appropriate section's resistivity based on connection scheme.

            :param value: longitudinal resistivity (Ohm.cm)
        '''
        if self.vref == 'v':
            value *= self.Cm0
        self.Ra = value

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

    def getVrefValue(self, **kwargs):
        ''' Get the value of the section's reference voltage variable used to compute
            intracellular currents.

            :param return: reference variable value
        '''
        return self.getValue(f'_ref_{self.vref}', **kwargs)

    def attachIax(self):
        ''' Attach a custom axial current point process to the section. '''
        # Compute section resistance
        R = self.axialResistance() if self.rmin is None else self.boundedAxialResistance()  # Ohm

        # Return axial current point-process set to current section
        return Iax(self, R)

    def attachExtracellular(self, *args):
        return Extracellular(self, *args)

    def connect(self, parent):
        ''' Connect to a parent section in series to enable trans-sectional axial current. '''
        for sec in [parent, self]:
            if sec.iax is None:
                sec.iax = sec.attachIax()

        # Inform sections about each other's axial resistance (in Ohm)
        parent.iax.Rother[1] = self.iax.R
        self.iax.Rother[0] = parent.iax.R

        # Set bi-directional pointers to sections about each other's membrane potential
        parent.iax.set_Vother(1, self.getVrefValue(x=0.5))
        self.iax.set_Vother(0, parent.getVrefValue(x=0.5))

        # If both sections have extracellular mechanisms
        if self.has_ext_mech and parent.has_ext_mech:
            # Inform sections about each other's extracellular axial resistances (in Ohm)
            parent.ext.xrother0[1] = self.ext.xr[0]
            parent.ext.xrother1[1] = self.ext.xr[1]
            self.ext.xrother0[0] = parent.ext.xr[0]
            self.ext.xrother1[0] = parent.ext.xr[1]

            # Set bi-directional pointers to sections about each other's extracellular potential
            parent.iax.set_Vextother(1, self.ext._ref_V0)
            self.iax.set_Vextother(0, parent.ext._ref_V0)

        # Register sections to one another
        self.parent = parent
        parent.child = self

    def insertPassiveMech(self, g, Erev):
        raise NotImplementedError

    def insertVext(self, xr=1e20, xg=1e10, xc=0.):
        ''' Insert extracellular mechanism with specific parameters.

            :param xr: axial resistance per unit length of first extracellular layer (Ohm/cm)
            :param xg: transverse conductance of first extracellular layer (S/cm2)
            :param xc: transverse capacitance of first extracellular layer (uF/cm2)
        '''
        self.ext = self.attachExtracellular(xc, xg, xr)
        self.has_ext_mech = True

    def setVext(self, Ve):
        ''' Set the extracellular potential just outside of a section. '''
        self.ext.e_extracellular = Ve / self.Cm0
        self.iax.setVext(self.ext._ref_V0)


class Iax(hclass(h.Iax)):
    ''' Axial current point process object that allows setting parameters on creation. '''

    def __new__(cls, sec, _):
        ''' Instanciation. '''
        return super(Iax, cls).__new__(cls, sec(0.5))

    def __init__(self, sec, R):
        ''' Initialization.

            :param sec: section object
            :param R: section resistance (Ohm)
        '''
        # Assign attributes
        super().__init__(sec)
        self.R = R

        # Declare Vother pointer arrays
        self.declare_Vother()
        self.declare_Vextother()

        # Set pointer to section's membrane potential
        h.setpointer(sec.getVrefValue(x=0.5), 'V', self)

        # While section not connected: assign infinite resistance and local membrane potential
        # to (nonexistent) neighboring sections
        for i in range(MAX_CUSTOM_CON):
            self.Rother[i] = 1e20    # Ohm
            self.set_Vother(i, sec.getVrefValue(x=0.5))

        # Assign local and neighboring extracellular potentials to ground
        self.setVext(h._ref_ground)
        for i in range(MAX_CUSTOM_CON):
            self.set_Vextother(i, h._ref_ground)

    def setVext(self, ref_hocvar):
        h.setpointer(ref_hocvar, 'Vext', self)


class Extracellular(hclass(h.custom_extracellular)):

    def __new__(cls, sec, *_):
        return super(Extracellular, cls).__new__(cls, sec(0.5))

    def __init__(self, sec, xr, xg, xc):
        super().__init__(sec)

        # Assign provided parameters
        self.xc[0] = xc                       # uF/cm2
        self.xg[0] = xg                       # S/cm2
        self.xr[0] = xr * (sec.L / CM_TO_UM)  # Ohm

        # Set remaining parameters
        self.xg[1] = 1e10                     # S/cm2
        self.xc[1] = 0                        # uF/cm2
        self.xr[1] = 1e20 * sec.L / CM_TO_UM  # Ohm
        self.Am = sec.membraneArea()          # cm2
        self.e_extracellular = 0.             # mV

        # Determine number of extracellular levels
        self.NLEVELS = 1

        # Declare Vother pointer arrays
        self.declare_Vother()
        self.declare_Vextother()

        # Assign infinite extracellular resistance to (nonexistent) neighboring sections
        for i in range(MAX_CUSTOM_CON):
            self.xrother0[i] = 1e20  # Ohm
            self.xrother1[i] = 1e20  # Ohm

        # Assign 2 levels neighboring extracellular potentials to local values
        for i in range(MAX_CUSTOM_CON):
            self.set_Vother(i, self._ref_V0)     # mV
            self.set_Vextother(i, self._ref_V1)  # mV

        # If section has an axial point-process
        if sec.iax is not None:
            # Assign pointers bidirectionally between axial point process and this one.
            sec.iax.setVext(self._ref_V0)
            self.setIax(sec.iax._ref_iax)

            # For each connected section
            for i, k in enumerate(['child', 'parent']):
                if hasattr(sec, k):
                    connected_sec = getattr(sec, k)

                    # If the section also has an extracellular mechanism
                    if connected_sec.has_ext_mech:
                        # Set pointers to neighboring ext. potentials in axial point-processes
                        connected_sec.iax.set_Vextother(i, self._ref_V0)
                        sec.iax.set_Vextother(1 - i, connected_sec.ext._ref_V0)

                        # Inform both sections about each other's extracellular axial reistances
                        connected_sec.ext.xrother0[i] = self.xr[0]      # Ohm
                        connected_sec.ext.xrother1[i] = self.xr[1]      # Ohm
                        self.xrother0[1 - i] = connected_sec.ext.xr[0]  # Ohm
                        self.xrother1[1 - i] = connected_sec.ext.xr[1]  # Ohm

                        # Set pointers to neighbor extracellular potentials in this point-process
                        connected_sec.ext.set_Vother(i, self._ref_V0)         # mV
                        connected_sec.ext.set_Vextother(i, self._ref_V1)      # mV
                        self.set_Vother(1 - i, connected_sec.ext._ref_V0)     # mV
                        self.set_Vextother(1 - i, connected_sec.ext._ref_V1)  # mV

        # Otherwise
        else:
            # Set axial current of this point-process to point towards zero value
            self.setIax(h._ref_ground)

    def setIax(self, ref_hocvar):
        h.setpointer(ref_hocvar, 'iax', self)

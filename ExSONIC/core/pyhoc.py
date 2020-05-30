# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-05-30 14:00:37

''' Utilities to manipulate HOC objects. '''

import numpy as np
from neuron import h, hclass

from PySONIC.utils import logger, isWithin
from ..utils import load_mechanisms, getNmodlDir, seriesGeq
from ..constants import *

# # Declare ground hoc variable
# h('{ground = 0}')

# load_mechanisms(getNmodlDir())


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

    def to_array(self):
        ''' Return itself as a numpy array. '''
        return np.array([self.getrow(i).to_python() for i in range(self.nrow())])

    @classmethod
    def from_array(cls, arr):
        ''' Create matrix object from numpy array. '''
        m = cls(*arr.shape)
        for i, row in enumerate(arr):
            m.setrow(i, h.Vector(row))
        return m

    def shape(self):
        ''' Return shape of matrix. '''
        return (self.nrow(), self.ncol())

    def addval(self, irow, jcol, val):
        ''' Add value to an element. '''
        self.setval(irow, jcol, self.getval(irow, jcol) + val)

    def expand(self, n):
        ''' Expand by a factor n. '''
        self.resize(self.nrow() * n, self.ncol() * n)

    def to_sparse(self):
        ''' Return a sparse version of the matrix (only diagonal elements). '''
        msparse = Matrix(self.nrow(), self.ncol(), 2)
        msparse.setdiag(0, self.getdiag(0))
        return msparse


class Vector(hclass(h.Vector)):
    ''' Interface to Hoc Vector with facilitated initialization from 2D numpy array. '''

    def expand(self, n):
        ''' Expand by a factor n. '''
        self.resize(self.size() * n)


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
        # section.iclamp = self

    def set(self, value):
        self.amp = value * self.xamp


class ExtField():
    ''' Extracellular field object that allows setting parameters on creation. '''

    def __init__(self, section, amplitude, **kwargs):
        self.section = section
        if not self.section.has_ext_mech:
            self.section.insertVext(**kwargs)
        self.xamp = amplitude

    def __repr__(self):
        return f'{self.__class__.__name__}({self.section.shortname()}, {self.xamp:.2f} mV)'

    def set(self, value):
        self.section.setVext(self.xamp * value)


class Section(hclass(h.Section)):
    ''' Interface to a Hoc Section with nseg=1. '''

    passive_mechname = 'custom_pas'
    # passive_mechname = 'pas'

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
        self.iclamp = None

    @property
    def cfac(self):
        raise NotImplementedError

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

            Applies cfac-correction to Q-based sections, such that
            Iax = dV / r = dQ / (r * cm)

            :param value: longitudinal resistivity (Ohm.cm)
        '''
        self.Ra = value * self.cfac

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

            Applies cfac-correction to Q-based sections.

            :param xr: axial resistance per unit length of first extracellular layer (Ohm/cm)
            :param xg: transverse conductance of first extracellular layer (S/cm2)
            :param xc: transverse capacitance of first extracellular layer (uF/cm2)
        '''
        self.insert('extracellular')
        self.xraxial[0] = xr * OHM_TO_MOHM  # MOhm/cm
        self.xg[0] = xg                     # S/cm2
        self.xc[0] = xc                     # S/cm2
        for i in range(2):
            self.xraxial[i] *= self.cfac
            self.xc[i] /= self.cfac
            self.xg[i] /= self.cfac
        self.has_ext_mech = True

    def setVext(self, Ve):
        ''' Set the extracellular potential just outside of a section. '''
        self.e_extracellular = Ve * self.cfac

    def getVextRef(self):
        ''' Get reference to section's extracellular voltage. '''
        return self(0.5)._ref_vext[0]

    def setVextProbe(self):
        ''' Set a probe for the section's extracellular voltage. '''
        return Probe(self.getVextRef(), factor=1 / self.cfac)

    def connect(self, parent):
        ''' Connect proximal end to the distal end of a parent section.

            :param parent: parent sectio object
        '''
        super().connect(parent, 1, 0)

    def hasPassive(self):
        ''' Determine if section contains a passive membrane mechanism. '''
        return h.ismembrane(self.passive_mechname, sec=self)

    def insertPassive(self):
        ''' Insert passive (leakage) mechanism. '''
        self.insert(self.passive_mechname)

    def setPassive(self, key, value):
        ''' Set an attribute of the passive (leakage) mechanism. '''
        setattr(self, f'{key}_{self.passive_mechname}', value)

    def setPassiveG(self, value):
        ''' Set the passive conductance (S/cm2). '''
        self.setPassive('g', value)

    def setPassiveE(self, value):
        ''' Set the passive reversal potential (mV). '''
        self.setPassive('e', value * self.cfac)

    # def iStimDensity(self):
    #     ''' Return the section's stimulating current divided by its membrane area. '''
    #     if self.iclamp is None:
    #         return 0
    #     return self.iclamp.i / self.membraneArea() * 1e-6 * self.cfac  # mA/cm2

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

    def setVmProbe(self):
        raise NotImplementedError

    def setQmProbe(self):
        raise NotImplementedError

    def setProbes(self):
        ''' Set recording vectors for all variables of interest in the section.

            :return: probes object dictionary
        '''
        d = {
            'Qm': self.setQmProbe(),
            'Vm': self.setVmProbe()}
        if self.has_ext_mech:
            d['Vext'] = self.setVextProbe()
        return d

    def setStimON(self, value):
        pass


class VSection(Section):
    ''' Interface to a Hoc Section with voltage based transmembrane dynamics. '''

    cfac = 1

    def __init__(self, name=None, cell=None, Cm0=1.):
        ''' Initialization.

            :param mechname: mechanism name
            :param states: list of mechanism time-varying states
            :param Cm0: resting membrane capacitance (uF/cm2)
        '''
        super().__init__(name=name, cell=cell)
        self.Cm0 = Cm0
        self.cm = self.Cm0

    def setVmProbe(self):
        return self.setProbe('v')

    def setQmProbe(self):
        return self.setProbe('v', factor=self.Cm0 / C_M2_TO_NC_CM2)


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

    @property
    def cfac(self):
        return self.Cm0

    def connect(self, parent):
        ''' Connect two sections together, provided they have the same membrane capacitance. '''
        if self.Cm0 != parent.Cm0:
            raise ValueError('Cannot connect Q-based sections with differing capacitance')
        super().connect(parent)

    def setVmProbe(self):
        return self.setProbe('v', factor=1 / self.Cm0)

    def setQmProbe(self):
        return self.setProbe('v', factor=1 / C_M2_TO_NC_CM2)


class MechSection(Section):
    ''' Interface to a section with associated point-neuron mechanism. '''

    # Aliases for NMODL-protected variable names
    NEURON_aliases = {'O': 'O1', 'C': 'C1'}
    stimon_var = 'stimon'

    def __init__(self, mechname, states=None, **kwargs):
        ''' Initialization.

            :param mechname: mechanism name
            :param states: list of mechanism time-varying states
        '''
        if kwargs:
            super().__init__(**kwargs)
        self.mechname = mechname
        self.states = states

    @property
    def mechname(self):
        return self._mechname

    @mechname.setter
    def mechname(self, value):
        if value == 'pas':
            value = self.passive_mechname
        self.insert(value)
        self._mechname = value

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

    def setProbes(self):
        return {
            **super().setProbes(),
            **{k: self.setMechProbe(self.alias(k)) for k in self.states}
        }


class MechVSection(MechSection, VSection):
    ''' Interface to a V-based Section with associated point-neuron mechanism. '''

    def __init__(self, **kwargs):
        name = kwargs.pop('name', None)
        cell = kwargs.pop('cell', None)
        Cm0 = kwargs.pop('Cm0', 1.)
        VSection.__init__(self, name=name, cell=cell, Cm0=Cm0)
        MechSection.__init__(self, **kwargs)

    def setQmProbe(self):
        return self.setMechProbe('Qm', factor=1 / C_M2_TO_NC_CM2)


class MechQSection(MechSection, QSection):
    ''' Interface to a Q-based Section with associated point-neuron mechanism. '''

    def __init__(self, **kwargs):
        name = kwargs.pop('name', None)
        cell = kwargs.pop('cell', None)
        Cm0 = kwargs.pop('Cm0', 1.)
        QSection.__init__(self, name=name, cell=cell, Cm0=Cm0)
        MechSection.__init__(self, **kwargs)

    def setVmProbe(self):
        return self.setMechProbe('Vm')


def getCustomConnectSection(section_class):

    class CustomConnectSection(section_class):

        passive_mechname = 'custom_pas'

        # Bounds of extracellular parameters (from extcelln.c)
        ext_bounds = {
            'xr': (1e-3, 1e21),  # Ohm/cm
            'xg': (0., 1e15),    # S/cm2
            'xc': (0., 1e15)     # uF/cm2
        }

        def __init__(self, cs, *args, **kwargs):
            ''' Initialization.

                :param cs: connector scheme object
            '''
            super().__init__(*args, **kwargs)
            if cs.vref == 'v':
                self.vref = cs.vref
            else:
                try:
                    key = self.mechname
                    self.imref = f'iNet_{key}'
                except AttributeError:
                    key = self.passive_mechname
                    self.imref = f'iPas_{key}'
                self.vref = f'{cs.vref}_{key}'
            self.rmin = cs.rmin
            # self.iax = None
            # self.parent_ref = None
            # self.child_ref = None
            self.ex = 0.       # mV
            self.ex_last = 0.  # mV

        @property
        def cfac(self):
            ''' Coupling-variable depdendent multiplying factor. '''
            return self.Cm0 if self.vref == 'v' else 1.

        # def has_parent(self):
        #     ''' Determine if section has parent. '''
        #     return self.parent_ref is not None

        # def has_child(self):
        #     ''' Determine if section has child. '''
        #     return self.child_ref is not None

        # def parent(self):
        #     ''' Return parent section. '''
        #     return self.parent_ref.sec

        # def child(self):
        #     ''' Return child section. '''
        #     return self.child_ref.sec

        def getIm(self, **kwargs):
            return self.getValue(self.imref, **kwargs)

        def getVm(self, **kwargs):
            ''' Get the value of the section's reference voltage variable used to compute
                intracellular currents.

                :param return: reference variable value
            '''
            return self.getValue(self.vref, **kwargs)

        def getVmRef(self, **kwargs):
            ''' Get the reference to the section's reference voltage variable used to compute
                intracellular currents.

                :param return: reference variable value
            '''
            return self.getValue(f'_ref_{self.vref}', **kwargs)

        @property
        def Ga_half(self):
            ''' Half-section intracellular axial conductance (S). '''
            return 2 / self.axialResistance()

        # def boundedAxialResistance(self):
        #     ''' Compute bounded value for axial resistance to ensure (resistance * membrane area)
        #         is always above a specific threshold, in order to limit the magnitude of axial currents.

        #         :return: bounded resistance value (Ohm)
        #     '''
        #     Am = self.membraneArea()    # cm2
        #     R = self.axialResistance()  # Ohm
        #     if self.rmin is None:
        #         return R
        #     s = f'{self}: R*Am = {R * Am:.1e} Ohm.cm2'
        #     if R < self.rmin / Am:
        #         s = f'{s} -> bounded to {self.rmin:.1e} Ohm.cm2'
        #         R = self.rmin / Am
        #         logger.warning(s)
        #     else:
        #         s = f'{s} -> not bounded'
        #         logger.debug(s)
        #     return R

        def getVextRef(self):
            ''' Get reference to section's extracellular voltage. '''
            return self.cell().getVextRef(self)

        # def getIaxRef(self):
        #     ''' Get reference to section's intracellular current. '''
        #     return self.iax._ref_iax

        # def iaxDensity(self):
        #     ''' Return the section's intracellular axial current divided by its membrane area. '''
        #     return self.iax.iax / self.membraneArea() * 1e-6 * self.cfac  # mA/cm2

        # def connectIntracellular(self, parent):
        #     ''' Connect the intracellular layers of a section and its parent. '''
        #     if self.cell().use_iax_pp:
        #         # Define axial current point-process to both sections, if not already done.
        #         for sec in [parent, self]:
        #             if sec.iax is None:
        #                 sec.iax = Iax(sec)

        #         # Add half axial conductances to appropriate indexes of Gax vectors
        #         parent.iax.updateAxialConductance(1, self.iax.Ghalf)  # S
        #         self.iax.updateAxialConductance(0, parent.iax.Ghalf)  # S

        #         # Set bi-directional pointers to sections about each other's membrane potential
        #         parent.iax.set_Vother(1, self.getVmRef(x=0.5))  # mV
        #         self.iax.set_Vother(0, parent.getVmRef(x=0.5))  # mV

        # def connectExtracellular(self, parent):
        #     ''' Connect the extracellular layers of a section and its parent. '''
        #     # Update extracellular conductance matrix
        #     self.cell().connectExtracellularNodes(self, parent)

        #     # # Set bi-directional pointers to sections about each other's extracellular potential
        #     # if self.cell().use_iax_pp:
        #     #     parent.iax.set_Vextother(1, self.getVextRef())  # mV
        #     #     self.iax.set_Vextother(0, parent.getVextRef())  # mV

        def connect(self, parent):
            ''' Connect to a parent section in series to enable trans-sectional axial currents. '''
            self.cell().registerConnection(parent, self)
            # Connect intracellular layer between the two sections
            # self.connectIntracellular(parent)

            # # Register parent and child section
            # self.parent_ref = h.SectionRef(sec=parent)
            # parent.child_ref = h.SectionRef(sec=self)

        @property
        def Gp_half(self):
            ''' Compute half-section extracellular axial conductance (S). '''
            return 2 / (self._xr * self.L / CM_TO_UM)

        def insertVext(self, xr=1e20, xg=1e10, xc=0.):
            ''' Insert extracellular mechanism with specific parameters.

                :param xr: axial resistance per unit length of first extracellular layer (Ohm/cm)
                :param xg: transverse conductance of first extracellular layer (S/cm2)
                :param xc: transverse capacitance of first extracellular layer (uF/cm2)
            '''
            # Check that provided parameters are within reasonable bounds, scale by cfac and assign
            self._xr = isWithin('xr', xr, self.ext_bounds['xr']) * self.cfac
            self._xg = isWithin('xg', xg, self.ext_bounds['xg']) / self.cfac
            self._xc = isWithin('xc', xc, self.ext_bounds['xc']) / self.cfac
            # if self._xc == 0:
            #     self._xc = 1e-20

            # Add values to the correct indexes of the extracellular arrays
            self.cell().setExtracellularNode(self)

            # # Compute half-section extracellular axial conductance
            # self.gp_half = 2 / (xr * self.L / CM_TO_UM * self.membraneArea())  # S/cm2

            # If section has Iax point-process, assign all Vext pointers to section's vext
            # if self.cell().use_iax_pp:
            #     if self.iax is not None:
            #         self.iax.setVextPointer(self.getVextRef())
            #         for i in range(MAX_CUSTOM_CON):
            #             self.iax.set_Vextother(i, self.getVextRef())

            # # If section has parent/child with extracellular mechanism, connect the extracellular parts
            # if self.has_parent() and self.parent().has_ext_mech:
            #     self.connectExtracellular(self.parent())
            # if self.has_child() and self.child().has_ext_mech:
            #     self.child().connectExtracellular(self)

            # Inform section that it has an extracellular mechanism
            self.has_ext_mech = True

        def setVext(self, Ve):
            ''' Set the extracellular potential just outside of a section. '''
            self.ex = Ve * self.cfac  # mV

    # Correct class name for consistency with input class
    CustomConnectSection.__name__ = f'CustomConnect{section_class.__name__}'

    return CustomConnectSection


# class Iax(hclass(h.Iax)):
#     ''' Axial current point process object that allows setting parameters on creation. '''

#     def __new__(cls, sec):
#         ''' Instanciation. '''
#         return super(Iax, cls).__new__(cls, sec(0.5))

#     def __init__(self, sec):
#         ''' Initialization.

#             :param sec: section object
#         '''
#         super().__init__(sec)

#         # Initialize axial conductance vector with section's half conductance
#         self.Ghalf = 2 / (sec.boundedAxialResistance() * OHM_TO_MOHM)  # S
#         for i in range(MAX_CUSTOM_CON):
#             self.Gax[i] = self.Ghalf  # uS

#         # Set Vm pointer to section's membrane potential and Vext pointer to ground
#         self.setVmPointer(sec)
#         self.setVextPointer(h._ref_ground)

#         # Declare Vother and Vextother pointer arrays, and set them to local membrane potential
#         # and ground, respectively
#         self.declare_Vother()
#         self.declare_Vextother()
#         for i in range(MAX_CUSTOM_CON):
#             self.set_Vother(i, sec.getVmRef(x=0.5))
#             self.set_Vextother(i, h._ref_ground)

#     def updateAxialConductance(self, i, G):
#         ''' Add a conductance in series to the ith element of the axial conductance vector. '''
#         self.Gax[i] = seriesGeq(self.Gax[i], G)

#     def setVmPointer(self, sec):
#         ''' Set pointer to section's membrane potential. '''
#         h.setpointer(sec.getVmRef(x=0.5), 'V', self)

#     def setVextPointer(self, ref_hocvar):
#         ''' Set pointer to section's extracellular potential. '''
#         h.setpointer(ref_hocvar, 'Vext', self)

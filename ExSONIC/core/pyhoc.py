# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-06-06 01:05:35

''' Utilities to manipulate HOC objects. '''

import numpy as np
from neuron import h, hclass

from PySONIC.utils import isWithin
from ..constants import *
from ..utils import seriesGeq


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


class Vector(hclass(h.Vector)):
    ''' Interface to Hoc Vector extra functionalities. '''

    def expand(self, n=2):
        ''' Expand by a factor n. '''
        self.resize(self.size() * n)


class Matrix(hclass(h.Matrix)):
    ''' Interface to Hoc Matrix with extra functionalities. '''

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

    def adddiag(self, i, vin):
        ''' Add values to a matrix diagonal. '''
        self.setdiag(i, self.getdiag(i) + vin)

    def expand(self, n=2):
        ''' Expand by a factor n. '''
        self.resize(self.nrow() * n, self.ncol() * n)

    def emptyClone(self):
        ''' Return empty matrix of identical shape. '''
        return Matrix(*self.shape())

    def mulByRow(self, x):
        ''' Return new matrix with rows multiplied by vector values. '''
        assert x.size == self.nrow(), f'Input vector must be of size {self.nrow()}'
        mout = self.emptyClone()
        for i in range(self.nrow()):
            mout.setrow(i, self.getrow(i) * x[i])
        return mout

    def copyTo(self, mout, i, j):
        ''' Copy the current matrix to a destination matrix, starting at a specific
            row and column index.
        '''
        self.bcopy(0, 0, self.nrow(), self.ncol(), i, j, mout)

    def addTo(self, mout, i, j, fac=1):
        ''' Add the current matrix to a destination matrix, starting at a specific
            row and column index.
        '''
        for k in range(self.nrow()):
            for l in range(self.ncol()):
                mout.addval(k + i, l + j, self.getval(k, l) * fac)

    def checkAgainst(self, other):
        assert isinstance(other, self.__class__), 'differing classes'

    def checkNullRows(self):
        ''' Check that all rows sum up to zero (or close). '''
        for i in range(self.nrow()):
            rsum = self.getrow(i).sum()
            assert np.isclose(rsum, .0, atol=1e-15), f'non-zero sum on line {i}: {rsum}'

    def __add__(self, other):
        ''' Addition operator. '''
        self.checkAgainst(other)
        mout = self.emptyClone()
        self.add(other, mout)
        return mout

    def __sub__(self, other):
        ''' Subtraction operator. '''
        return self.__add__(-other)

    def __mul__(self, other):
        ''' Multiplication operator. '''
        if isinstance(other, (float, int)):
            mout = self.c()
            mout.muls(other)
            return mout
        elif isinstance(other, Vector):
            return self.mulv(other)
        else:
            self.checkAgainst(other)
            mout = Matrix(self.nrow(), other.ncol())
            self.mulm(other, mout)
            return mout


class SquareMatrix(Matrix):

    def __new__(cls, n):
        ''' Instanciation. '''
        return super(SquareMatrix, cls).__new__(cls, n, n)

    def nside(self):
        ''' Return side of matrix side. '''
        return self.nrow()

    def emptyClone(self):
        return SquareMatrix(self.nside())

    def addLink(self, i, j, w):
        ''' Add a bi-directional link between two nodes with a specific weight.

            :param i: first node index
            :param j: second node index
            :param w: link weight
        '''
        self.addval(i, i, w)
        self.addval(i, j, -w)
        self.addval(j, j, w)
        self.addval(j, i, -w)


class DiagonalMatrix(SquareMatrix):

    def __new__(cls, x):
        ''' Instanciation. '''
        return super(DiagonalMatrix, cls).__new__(cls, x.size)

    def __init__(self, x):
        self.setdiag(0, Vector(x))


class ConductanceMatrix(SquareMatrix):

    def __new__(cls, Gvec, **_):
        ''' Instanciation. '''
        return super(ConductanceMatrix, cls).__new__(cls, Gvec.size)

    def __init__(self, Gvec, links=None):
        self.Gvec = Gvec
        self.links = links
        if self.links is not None:
            self.addLinks()

    def addLink(self, i, j):
        super().addLink(i, j, seriesGeq(self.Gvec[i], self.Gvec[j]))

    def addLinks(self):
        for i, j in self.links:
            self.addLink(i, j)
        self.checkNullRows()

    def reset(self, gvec):
        assert gvec.size == self.nrow(), f'Input vector must be of size {self.nrow()}'
        self.gvec = gvec
        self.zero()
        self.addLinks()


class SubMatrix:

    def __init__(self, m, row_offset, col_offset):
        self.mref = h.Pointer(m)
        self.row_offset = row_offset
        self.col_offset = col_offset

    @property
    def m(self):
        return self.mref.val

    @property
    def diag_offset(self):
        return self.col_offset - self.row_offset

    def getval(self, i, j):
        return self.m.getval(i + self.row_offset, j + self.col_offset)

    def setval(self, i, j, x):
        self.m.setval(i + self.row_offset, j + self.col_offset, x)

    def getdiag(i):
        return self.m.getdiag(i + self.diag_offset)

    def setdiag(i, x):
        self.m.setdiag(i + self.diag_offset, x)


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
        self.secref = h.SectionRef(sec=section)

    def set(self, value):
        self.amp = value * self.xamp
        self.secref.sec.istim = self.amp


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

    def __init__(self, name=None, cell=None, Cm0=1.):
        ''' Initialization.

            :param name: section name
            :param cell: section cell
            :param Cm0: resting membrane capacitance (uF/cm2)
        '''
        if name is None:
            raise ValueError('section must have a name')
        if cell is not None:
            super().__init__(name=name, cell=cell)
        else:
            super().__init__(name=name)
        self.Cm0 = Cm0
        self.nseg = 1
        self.has_ext_mech = False
        self.istim = 0.

    @property
    def cfac(self):
        raise NotImplementedError

    @property
    def istim(self):
        return self._istim

    @istim.setter
    def istim(self, value):
        self._istim = value / MA_TO_NA / self.membraneArea()

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

    @property
    def Ga_half(self):
        ''' Half-section intracellular axial conductance (S). '''
        return 2 / self.axialResistance()

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

    def insertPassive(self):
        ''' Insert passive (leakage) mechanism. '''
        self.insert(self.passive_mechname)

    def hasPassive(self):
        ''' Determine if section contains a passive membrane mechanism. '''
        return h.ismembrane(self.passive_mechname, sec=self)

    def setPassive(self, key, value):
        ''' Set an attribute of the passive (leakage) mechanism. '''
        setattr(self, f'{key}_{self.passive_mechname}', value)

    def setPassiveG(self, value):
        ''' Set the passive conductance (S/cm2). '''
        self.setPassive('g', value)

    def setPassiveE(self, value):
        ''' Set the passive reversal potential (mV). '''
        self.setPassive('e', value * self.cfac)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cm = self.Cm0

    @property
    def cfac(self):
        return 1.

    def setVmProbe(self):
        return self.setProbe('v')

    def setQmProbe(self):
        return self.setProbe('v', factor=self.Cm0 / C_M2_TO_NC_CM2)


class QSection(Section):
    ''' Interface to a Hoc Section with charge-density based transmembrane dynamics.

        In this section type, the differential variable "v" stands for charge density (in nC/cm2).
    '''

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
            'rx': (1e-3, 1e21),  # Ohm/cm
            'gx': (0., 1e15),    # S/cm2
            'cx': (0., 1e15)     # uF/cm2
        }

        # Default transverse conductance of second extracellular layer (from online doc)
        default_xg = 1e9  # S/cm2

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
            self.ex = 0.       # mV
            self.ex_last = 0.  # mV

        @property
        def cfac(self):
            ''' Coupling-variable depdendent multiplying factor. '''
            return self.Cm0 if self.vref == 'v' else 1.

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

        def getCm(self, **kwargs):
            return self.getValue('v', **kwargs) / self.getVm(**kwargs)

        def getVextRef(self):
            ''' Get reference to section's extracellular voltage. '''
            return self.cell().getVextRef(self)

        def connect(self, parent):
            self.cell().registerConnection(parent, self)

        def checkXBounds(self, key, val):
            return isWithin(key, val, self.ext_bounds[key])

        @property
        def rx(self):
            return self._rx

        @rx.setter
        def rx(self, value):
            self._rx = self.checkXBounds('rx', value) * self.cfac

        @property
        def gx(self):
            return self._gx

        @gx.setter
        def gx(self, value):
            ''' Add NEURON's default xg in series to mimick 2nd extracellular layer. '''
            self._gx = seriesGeq(self.checkXBounds('gx', value), self.default_xg) / self.cfac

        @property
        def cx(self):
            return self._cx

        @cx.setter
        def cx(self, value):
            self._cx = self.checkXBounds('cx', value) / self.cfac

        @property
        def Gp_half(self):
            ''' Half-section extracellular axial conductance (S). '''
            return 2 / (self.rx * self.L / CM_TO_UM)

        def insertVext(self, xr=1e20, xg=1e10, xc=0.):
            ''' Insert extracellular mechanism with specific parameters.

                :param xr: axial resistance per unit length of first extracellular layer (Ohm/cm)
                :param xg: transverse conductance of first extracellular layer (S/cm2)
                :param xc: transverse capacitance of first extracellular layer (uF/cm2)
            '''
            # Check input parameters before assignment, and set extracellular node
            self.rx = xr
            self.gx = xg
            self.cx = xc
            self.cell().setExtracellularNode(self)
            self.has_ext_mech = True

        def setVext(self, Ve):
            ''' Set the extracellular potential just outside of a section. '''
            self.ex = Ve * self.cfac  # mV

    # Correct class name for consistency with input class
    CustomConnectSection.__name__ = f'CustomConnect{section_class.__name__}'

    return CustomConnectSection

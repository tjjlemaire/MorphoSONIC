# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-02-27 23:08:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-07-21 18:38:41

import numpy as np

from PySONIC.core import Lookup
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger

from ..constants import *
from .nmodel import FiberNeuronModel
from .sonic import addSonicFeatures

# MRG lookup table from McIntyre 2002
mrg_lkp = Lookup(
    refs={
        'fiberD': np.array([5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]) / M_TO_UM},
    tables={
        'nodeD': np.array([1.9, 2.4, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5]) / M_TO_UM,
        'interD': np.array([3.4, 4.6, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7]) / M_TO_UM,
        'interL': np.array([500., 750., 1000., 1150., 1250., 1350., 1400., 1450., 1500.]) / M_TO_UM,
        'flutL': np.array([35., 38., 40., 46., 50., 54., 56., 58., 60.]) / M_TO_UM,
        'nlayers': np.array([80, 100, 110, 120, 130, 135, 140, 145, 150])},
    interp_method='linear', extrapolate=True)


@addSonicFeatures
class MRGFiber(FiberNeuronModel):
    ''' Generic double-cable, myelinated fiber model based on McIntyre 2002, extended
        to allow use with any fiber diameter.

        Refereence:
        *McIntyre, C.C., Richardson, A.G., and Grill, W.M. (2002). Modeling the
        excitability of mammalian nerve fibers: influence of afterpotentials on
        the recovery cycle. J. Neurophysiol. 87, 995–1006.*
    '''
    simkey = 'mrg'
    is_myelinated = True
    _pneuron = getPointNeuron('MRGnode')
    _rs = 70.0                      # axoplasm resistivity (Ohm.cm)
    _nodeL = 1e-6                   # node length (m)
    _mysaL = 3e-6                   # MYSA length (m)
    _mysa_space = 2e-9              # MYSA periaxonal space width (m)
    _flut_space = 4e-9              # FLUT periaxonal space width (m)
    _stin_space = 4e-9              # STIN periaxonal space width (m)
    _C_inter = 2.                   # internodal axolammelar capacitance (uF/cm2)
    _g_mysa = 0.001                 # MYSA axolammelar conductance (S/cm2)
    _g_flut = 0.0001                # FLUT axolammelar conductance (S/cm2)
    _g_stin = 0.0001                # STIN axolammelar conductance (S/cm2)
    _C_myelin_per_membrane = 0.1    # myelin capacitance per lamella (uF/cm2)
    _g_myelin_per_membrane = 0.001  # myelin transverse conductance per lamella (S/cm2)
    _nstin_per_inter = 6            # number of stin sections per internode
    correction_choices = ('myelin', 'axoplasm')  # possible choices of correction level

    def __init__(self, fiberD, nnodes, correction_level=None, **kwargs):
        ''' Constructor.

            :param fiberD: fiber diameter (m)
            :param nnodes: number of nodes
            :param correction_level: level at which model properties are adjusted to ensure correct
                myelin representation with the extracellular mechanism ('myelin' or 'axoplasm')
        '''
        self.fiberD = fiberD
        self.nnodes = nnodes
        self.correction_level = correction_level
        for k, v in mrg_lkp.project('fiberD', self.fiberD).items():
            setattr(self, k, v)
        self.setCelsius()
        super().__init__(**kwargs)

    @property
    def meta(self):
        return {**super().meta, 'correction_level': self.correction_level}

    @staticmethod
    def getMetaArgs(meta):
        args, kwargs = FiberNeuronModel.getMetaArgs(meta)
        kwargs.update({'correction_level': meta['correction_level']})
        return args, kwargs

    @property
    def mysaL(self):
        return self._mysaL

    @mysaL.setter
    def mysaL(self, value):
        if value <= 0:
            raise ValueError('mysa length must be positive')
        self.set('mysaL', value)

    @property
    def flutL(self):
        return self._flutL

    @flutL.setter
    def flutL(self, value):
        if value <= 0:
            raise ValueError('paranode length must be positive')
        self.set('flutL', value)

    @property
    def nlayers(self):
        return self._nlayers

    @nlayers.setter
    def nlayers(self, value):
        if value < 1:
            raise ValueError('number of myelin layers must be at least one')
        self.set('nlayers', int(np.round(value)))

    @property
    def rhop(self):
        ''' Periaxonal resistivity (Ohm.cm) '''
        return self.rs

    @property
    def nMYSA(self):
        ''' Number of paranodal myelin attachment (MYSA) sections. '''
        return 2 * self.ninters

    @property
    def nFLUT(self):
        ''' Number of paranodal main (FLUT) sections. '''
        return 2 * self.ninters

    @property
    def nSTIN(self):
        ''' Number of internodal (STIN) sections. '''
        return self._nstin_per_inter * self.ninters

    @property
    def mysaD(self):
        ''' Diameter of paranodal myelin attachment (MYSA) sections (m). '''
        return self.nodeD

    @property
    def flutD(self):
        ''' Diameter of paranodal main (FLUT) sections (m). '''
        return self.interD

    @property
    def stinD(self):
        ''' Diameter of internodal (STIN) sections (m). '''
        return self.interD

    @property
    def stinL(self):
        ''' Length of internodal sections (m). '''
        return (self.interL - (self.nodeL + 2 * (self.mysaL + self.flutL))) / self._nstin_per_inter

    @property
    def R_mysa(self):
        ''' MYSA intracellular axial resistance (Ohm). '''
        return self.axialResistance(self.rhoa, self.mysaL, self.mysaD)

    @property
    def R_flut(self):
        ''' FLUT intracellular axial resistance (Ohm). '''
        return self.axialResistance(self.rhoa, self.flutL, self.flutD)

    @property
    def R_stin(self):
        ''' STIN intracellular axial resistance (Ohm). '''
        return self.axialResistance(self.rhoa, self.stinL, self.stinD)

    @property
    def R_node_to_node(self):
        ''' Node-to-node intracellular axial resistance (Ohm). '''
        return self.R_node + 2 * (self.R_mysa + self.R_flut) + self._nstin_per_inter * self.R_stin

    @property
    def Rp_node(self):
        ''' Node periaxonal axial resistance per unit length (Ohm/cm). '''
        return self.periaxonalResistancePerUnitLength(self.nodeD, self._mysa_space)

    @property
    def Rp_mysa(self):
        ''' MYSA periaxonal axial resistance per unit length (Ohm/cm). '''
        return self.periaxonalResistancePerUnitLength(self.mysaD, self._mysa_space)

    @property
    def Rp_flut(self):
        ''' FLUT periaxonal axial resistance per unit length (Ohm/cm). '''
        return self.periaxonalResistancePerUnitLength(self.flutD, self._flut_space)

    @property
    def Rp_stin(self):
        ''' STIN periaxonal axial resistance per unit length (Ohm/cm). '''
        return self.periaxonalResistancePerUnitLength(self.stinD, self._stin_space)

    @property
    def C_myelin(self):
        ''' Compute myelin capacitance per unit area from its nominal value per lamella membrane.

            ..note: This formula assumes that each lamella membrane (2 per myelin layer) is
            represented by an individual RC circuit with a capacitor and passive resitor, and
            that these components are connected independently in series to form a global RC circuit.
        '''
        return self._C_myelin_per_membrane / (2 * self.nlayers)  # uF/cm2

    @property
    def g_myelin(self):
        ''' Compute myelin passive conductance per unit area from its nominal value per
            lamella membrane.

            ..note: This formula assumes that each lamella membrane (2 per myelin layer) is
            represented by an individual RC circuit with a capacitor and passive resitor, and
            that these components are connected independently in series to form a global RC circuit.
        '''
        return self._g_myelin_per_membrane / (2 * self.nlayers)  # S/cm2

    @property
    def correction_level(self):
        return self._correction_level

    @correction_level.setter
    def correction_level(self, value):
        if value is None:
            value = self.correction_choices[0]
        if not isinstance(value, str) or value not in self.correction_choices:
            raise ValueError(f'correction_level choices are : ({self.correction_choices})')
        self._correction_level = value

    @property
    def mysaIDs(self):
        return [f'MYSA{i}' for i in range(self.nMYSA)]

    @property
    def flutIDs(self):
        return [f'FLUT{i}' for i in range(self.nFLUT)]

    @property
    def stinIDs(self):
        return [f'STIN{i}' for i in range(self.nSTIN)]

    @property
    def refsection(self):
        return self.nodelist[0]

    @property
    def inters(self):
        return {
            'MYSA': self.mysas,
            'FLUT': self.fluts,
            'STIN': self.stins,
        }

    @property
    def sections(self):
        return {'node': self.nodes, **self.inters}

    @property
    def interlist(self):
        d = []
        for secdict in self.inters.values():
            d += list(secdict.values())
        return d

    @property
    def seclist(self):
        return self.nodelist + self.interlist

    @property
    def length(self):
        return self.interL * (self.nnodes - 1) + self.nodeL

    def getNodeCoords(self):
        ''' Return vector of node coordinates along axial dimension, centered at zero (um). '''
        xnodes = self.interL * np.arange(self.nnodes)
        return xnodes - xnodes[int((self.nnodes - 1) / 2)]

    def getMysaCoords(self):
        xref = self.getNodeCoords()
        delta = 0.5 * (self.nodeL + self.mysaL)
        return np.ravel(np.column_stack((xref[:-1] + delta, xref[1:] - delta)))

    def getFlutCoords(self):
        xref = self.getMysaCoords()
        delta = 0.5 * (self.mysaL + self.flutL)
        return np.ravel(np.column_stack((xref[::2] + delta, xref[1::2] - delta)))

    def getStinCoords(self):
        xref = self.getFlutCoords()[::2] + 0.5 * (self.flutL + self.stinL)
        return np.ravel([xref + i * self.stinL for i in range(self._nstin_per_inter)], order='F')

    def isInternodalDistance(self, d):
        return np.isclose(d, self.interL)

    def periaxonalResistancePerUnitLength(self, d, w):
        ''' Compute the periaxonal axial resistance per unit length of a cylindrical section.

            :param d: section inner diameter (m)
            :param w: periaxonal space width (m)
            :return: resistance per unit length (Ohm/cm)
        '''
        return self.axialResistancePerUnitLength(self.rhop, d + 2 * w, d_in=d)

    def clear(self):
        ''' delete all model sections. '''
        del self.nodes
        del self.mysas
        del self.stins
        del self.fluts

    def createSections(self):
        ''' Create morphological sections with specific membrane mechanisms. '''
        self.nodes = {k: self.createSection(
            k, mech=self.mechname, states=self.pneuron.statesNames(),
            Cm0=self.pneuron.Cm0 * F_M2_TO_UF_CM2) for k in self.nodeIDs}
        self.mysas = {k: self.createSection(k, Cm0=self._C_inter) for k in self.mysaIDs}
        self.fluts = {k: self.createSection(k, Cm0=self._C_inter) for k in self.flutIDs}
        self.stins = {k: self.createSection(k, Cm0=self._C_inter) for k in self.stinIDs}

    def sectionsdict(self):
        return {
            'node': self.nodes,
            'MYSA': self.mysas,
            'FLUT': self.fluts,
            'STIN': self.stins
        }

    def diameterRatio(self, sectype):
        ''' Return the ratio between a section's type diameter and the fiber outer diameter. '''
        return getattr(self, f'{sectype.lower()}D') / self.fiberD

    def setGeometry(self):
        ''' Set sections geometry.

            .. note:: if the axoplasm correction level is selected, all internodal sections
            are created with a diameter corresponding to the fiber's outer diameter (which
            differs from their actual diameter).
        '''
        logger.debug('setting sections geometry')
        for sec in self.nodes.values():
            sec.setGeometry(self.nodeD, self.nodeL)
        for sectype, secdict in self.inters.items():
            d, L = [getattr(self, f'{sectype.lower()}{x}') for x in ['D', 'L']]
            if self.correction_level == 'axoplasm':
                d = self.fiberD
            for sec in secdict.values():
                sec.setGeometry(d, L)

    def setResistivity(self):
        ''' Assign sections axial resistivity.

            .. note:: if the axoplasm correction level is selected, internodal sections
            resistivities are divided by the squared ratio of the section's real diameter
            divided by its assigned diameter (i.e. the fiber's outer diameter), to ensure
            correct sections resistances per unit length.
        '''
        logger.debug('setting sections resistivity')
        for sec in self.nodes.values():
            sec.setResistivity(self.rhoa)
        for sectype, secdict in self.inters.items():
            rhoa = self.rhoa
            if self.correction_level == 'axoplasm':
                rhoa /= self.diameterRatio(sectype)**2
            for sec in secdict.values():
                sec.setResistivity(rhoa)

    def setBiophysics(self):
        ''' Assign biophysics to model sections.

            .. note:: if the axoplasm correction level is selected, internodal sections
            leakage conductance and capacitance per unit area are multiplied by the ratio
            of the section's real diameter divided by its assigned diameter (i.e. the
            fiber's outer diameter), to ensure correct sections extensive membrane properties.
        '''
        logger.debug('setting sections biophysics')
        super().setBiophysics()
        for sectype, secdict in self.inters.items():
            g = getattr(self, f'_g_{sectype.lower()}')
            if self.correction_level == 'axoplasm':
                for sec in secdict.values():
                    sec.cm *= self.diameterRatio(sectype)
                g *= self.diameterRatio(sectype)
            for sec in secdict.values():
                sec.insertPassive()
                sec.setPassiveG(g)
                sec.setPassiveE(self.pneuron.Vm0)

    def setExtracellular(self):
        ''' Set the sections' extracellular mechanisms.

            .. note:: if the myelin correction level is selected, internodal sections
            myelin conductance and capacitance per unit area are divided by the ratio
            of the section's diameter divided by the fiber's outer diameter, to ensure
            correct myelin extensive membrane properties.
        '''
        logger.debug('setting extracellular mechanisms')
        Rp_node = self.Rp_node
        for sec in self.nodes.values():
            sec.insertVext(xr=Rp_node * OHM_TO_MOHM)
        for sectype, secdict in self.inters.items():
            xr = getattr(self, f'Rp_{sectype.lower()}')
            xc, xg = self.C_myelin, self.g_myelin
            if self.correction_level == 'myelin':
                xc /= self.diameterRatio(sectype)
                xg /= self.diameterRatio(sectype)
            for sec in secdict.values():
                sec.insertVext(xr=xr * OHM_TO_MOHM, xc=xc, xg=xg)

    def setTopology(self):
        ''' Connect the sections together '''
        logger.debug('connecting sections together')
        for i in range(self.nnodes - 1):
            self.connect('node', i, 'MYSA', 2 * i)
            self.connect('MYSA', 2 * i, 'FLUT', 2 * i)
            self.connect('FLUT', 2 * i, 'STIN', self._nstin_per_inter * i)
            for j in range(self._nstin_per_inter - 1):
                self.connect('STIN', self._nstin_per_inter * i + j,
                             'STIN', self._nstin_per_inter * i + j + 1)
            self.connect('STIN', self._nstin_per_inter * (i + 1) - 1, 'FLUT', 2 * i + 1)
            self.connect('FLUT', 2 * i + 1, 'MYSA', 2 * i + 1)
            self.connect('MYSA', 2 * i + 1, 'node', i + 1)

    def toInjectedCurrents(self, Ve):
        # Extract and reshape STIN Ve to (6 x ninters) array to facilitate downstream computations
        Ve_stin = Ve['STIN'].reshape((self._nstin_per_inter, self.ninters), order='F')

        # Compute equivalent currents flowing across sections (in nA)
        I_node_mysa = np.vstack((  # currents flowing from nodes to MYSA compartments
            Ve['node'][:-1] - Ve['MYSA'][::2],  # node -> right-side MYSA
            Ve['node'][1:] - Ve['MYSA'][1::2]   # left-side MYSA <- node
        )) * 2 / (self.R_node + self.R_mysa) * MA_TO_NA
        I_mysa_flut = np.vstack((  # currents flowing from MYSA to FLUT compartments
            Ve['MYSA'][::2] - Ve['FLUT'][::2],    # right-side MYSA -> right-side FLUT
            Ve['MYSA'][1::2] - Ve['FLUT'][1::2],  # left-side FLUT <- left-side MYSA
        )) * 2 / (self.R_mysa + self.R_flut) * MA_TO_NA
        I_flut_stin = np.vstack((  # currents flowing from FLUT to boundary STIN compartments
            Ve['FLUT'][::2] - Ve_stin[0],   # right-side FLUT -> right-side boundary STIN
            Ve['FLUT'][1::2] - Ve_stin[-1]  # left-side boundary STIN <- left-side FLUT
        )) * 2 / (self.R_flut + self.R_stin) * MA_TO_NA
        I_stin_stin = np.diff(Ve_stin, axis=0) / self.R_stin * MA_TO_NA  # STIN -> STIN

        # Create an array to currents flowing to nodes with appropriate zero-padding at extremities
        I_mysa_node = np.vstack((
            np.hstack((-I_node_mysa[0], [0.])),  # node <- right-side MYSA: 0 at the end
            np.hstack(([0.], -I_node_mysa[1]))   # left-side MYSA -> node: 0 at the beginning
        ))

        # Return dictionary with sum of currents flowing to each section
        return {
            'node': I_mysa_node[0] + I_mysa_node[1],  # -> node
            'MYSA': np.ravel(I_node_mysa - I_mysa_flut, order='F'),  # -> MYSA
            'FLUT': np.ravel(I_mysa_flut - I_flut_stin, order='F'),  # -> FLUT
            'STIN': np.ravel(np.vstack((
                I_stin_stin[0] + I_flut_stin[0],   # right-side boundary STIN
                np.diff(I_stin_stin, axis=0),      # central STIN
                -I_stin_stin[-1] + I_flut_stin[1]  # left-side boundary STIN
            )), order='F')
        }

    @property
    def CV_estimate(self):
        ''' Estimated diameter-dependent conduction veclocity
            (from linear fit across 5-20 um range)
        '''
        return 6.8 * self.fiberD * M_TO_UM - 18.5  # m/s


class MRGFiberVbased(MRGFiber.__original__):
    ''' Small class allowing to simulate the original MRG model with a V-based scheme. '''

    @property
    def refvar(self):
        return 'Vm'

    @property
    def modfile(self):
        return f'{self.pneuron.name}_clean.mod'

    @property
    def mechname(self):
        return self.pneuron.name


MRGFiber.__originalVbased__ = MRGFiberVbased

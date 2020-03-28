# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-02-27 23:08:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-03-28 14:07:27

import numpy as np
from neuron import h

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import si_format, logger, plural
from PySONIC.threshold import threshold, getLogStartPoint

from ..utils import getNmodlDir, load_mechanisms, array_print_options
from .nmodel import FiberNeuronModel
from .pyhoc import IClamp, MechVSection, ExtField
from .node import IintraNode
from .connectors import SerialConnectionScheme


class MRGFiber(FiberNeuronModel):
    ''' Generic double-cable, myelinated fiber model based on McIntyre 2002, extended
        to allow use with any fiber diameter.

        Refereence:
        *McIntyre, C.C., Richardson, A.G., and Grill, W.M. (2002). Modeling the
        excitability of mammalian nerve fibers: influence of afterpotentials on
        the recovery cycle. J. Neurophysiol. 87, 995–1006.*
    '''
    # Diameters of nodal and paranodal sections (m)
    nodeL = 1e-6  # node
    mysaL = 3e-6  # MYSA

    nstin_per_inter = 6
    inter_mechname = 'custom_pas'

    # Periaxonal space widths in paranodal and internodal sections (m)
    mysa_space = 2e-9  # MYSA
    flut_space = 4e-9  # FLUT
    stin_space = 4e-9  # STIN

    # Axolammelar and myelin conductances in paranodal and internodal sections (S/cm2)
    g_mysa = 0.001                         # MYSA axolammelar conductance
    g_flut = 0.0001                        # FLUT axolammelar conductance
    g_stin = 0.0001                        # STIN axolammelar conductance
    g_myelin_per_lamella_membrane = 0.001  # myelin transverse conductance per lamella

    # Specific membrane and myelin capacitances across sections (uF/cm2)
    C_inter = 2.                         # internodal axolammelar capacitance
    C_myelin_per_lamella_membrane = 0.1  # myelin capacitance per lamella

    # Possible choices of correction level for correct extracellular coupling
    correction_choices = ('myelin', 'axoplasm')

    def __init__(self, pneuron, nnodes, rs, fiberD, nodeD, interD, interL, flutL, nlayers,
                 correction_level=None):
        ''' Constructor.

            :param pneuron: point-neuron model object
            :param nnodes: number of nodes
            :param rs: longitudinal resistivity (Ohm.cm)
            :param fiberD: fiber diameter (m)
            :param nodeD: diameter of nodal and MYSA paranodal sections (m)
            :param interD: diameter of FLUT and STIN internodal sections (m)
            :param interL: internodal distance (m)
            :param flutL: main paranodal segment length (m)
            :param nlayers: number of myelin lamellae
            :param correction_level: level at which model properties are adjusted to ensure correct
                myelin representation with the extracellular mechanism ('myelin' or 'axoplasm')
        '''
        # Assign attributes
        self.pneuron = pneuron
        self.nnodes = nnodes
        self.rs = rs          # Ohm.cm
        self.fiberD = fiberD  # m
        self.nodeD = nodeD    # m
        self.interD = interD  # m
        self.interL = interL  # m
        self.flutL = flutL    # m
        self.nlayers = nlayers
        self.correction_level = correction_level

        # Set temperature
        self.setCelsius()

        # Compute sections intracellular resistances (in Ohm)
        self.R_node = self.axialResistance(self.rhoa, self.nodeL, self.nodeD)
        self.R_mysa = self.axialResistance(self.rhoa, self.mysaL, self.mysaD)
        self.R_flut = self.axialResistance(self.rhoa, self.flutL, self.flutD)
        self.R_stin = self.axialResistance(self.rhoa, self.stinL, self.stinD)

        # Compute periaxonal axial resistances per unit length (in Ohm/cm)
        self.Rp_node = self.periaxonalResistancePerUnitLength(self.nodeD, self.mysa_space)
        self.Rp_mysa = self.periaxonalResistancePerUnitLength(self.mysaD, self.mysa_space)
        self.Rp_flut = self.periaxonalResistancePerUnitLength(self.flutD, self.flut_space)
        self.Rp_stin = self.periaxonalResistancePerUnitLength(self.stinD, self.stin_space)

        # Compute myelin global capacitance and conductance per unit area from their equivalent
        # nominal values per lamella membrane, assuming that each lamella membrane (2 per myelin
        # layer) is represented by an individual RC circuit with a capacitor and passive resitor,
        # and these components are connected independently in series to form a global RC circuit.
        # Note that these global values are still intensive proerties (per unit area), that NEURON
        # integrates a over a given surface area before solving the PDE.
        nmembranes = 2 * self.nlayers
        self.C_myelin = self.C_myelin_per_lamella_membrane / nmembranes  # uF/cm2
        self.g_myelin = self.g_myelin_per_lamella_membrane / nmembranes  # S/cm2

        # Load mechanisms and set appropriate membrane mechanism
        load_mechanisms(getNmodlDir(), self.modfile)

        # Construct model
        self.construct()

    @property
    def nlayers(self):
        return self._nlayers

    @nlayers.setter
    def nlayers(self, value):
        if value < 1:
            raise ValueError('number of myelin layers must be at least one')
        self._nlayers = int(np.round(value))

    @property
    def fiberD(self):
        return self._fiberD

    @fiberD.setter
    def fiberD(self, value):
        if value <= 0:
            raise ValueError('fiber diameter must be positive')
        self._fiberD = value

    @property
    def flutL(self):
        return self._flutL

    @flutL.setter
    def flutL(self, value):
        if value <= 0:
            raise ValueError('paranode length must be positive')
        self._flutL = value

    @property
    def interL(self):
        return self._interL

    @interL.setter
    def interL(self, value):
        if value <= 0:
            raise ValueError('internode length must be positive')
        self._interL = value

    @property
    def rhoa(self):
        ''' Axoplasmic resistivity (Ohm.cm) '''
        return self.rs

    @property
    def rhop(self):
        ''' Periaxonal resistivity (Ohm.cm) '''
        return self.rs

    @property
    def ninters(self):
        ''' Number of (abstract) internodal sections. '''
        return self.nnodes - 1

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
        return self.nstin_per_inter * self.ninters

    @property
    def ntot(self):
        ''' Total number of sections in the model. '''
        return self.nnodes + self.nMYSA + self.nFLUT + self.nSTIN

    @property
    def stinL(self):
        ''' Length of internodal sections (m). '''
        return (self.interL - (self.nodeL + 2 * (self.mysaL + self.flutL))) / self.nstin_per_inter

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
    def meta(self):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'nnodes': self.nnodes,
            'rs': self.rs,
            'fiberD': self.fiberD,
            'nodeD': self.nodeD,
            'interD': self.interD,
            'interL': self.interL,
            'flutL': self.flutL,
            'nlayers': self.nlayers
        }

    @staticmethod
    def getMRGArgs(meta):
        return [meta[x] for x in ['nnodes', 'rs', 'fiberD', 'nodeD', 'interD',
                                  'interL', 'flutL', 'nlayers']]

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
        return np.ravel([xref + i * self.stinL for i in range(self.nstin_per_inter)], order='F')

    def isNormalDistance(self, d):
        return np.isclose(d, self.interL)

    @classmethod
    def initFromMeta(cls, meta):
        return cls(getPointNeuron(meta['neuron']), *cls.getMRGArgs(meta))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.fiberD * 1e6:.1f} um, {self.nnodes} nodes)'

    @property
    def modelcodes(self):
        return {
            **self.corecodes,
            'nnodes': f'{self.nnodes}node{plural(self.nnodes)}',
            'rs': f'rs{self.rs:.0f}ohm.cm',
            'fiberD': f'{self.fiberD * 1e6:.1f}um',
            'nodeD': f'nodeD{self.nodeD * 1e6:.1f}um',
            'interD': f'{self.interD * 1e6:.1f}um',
            'interL': f'{self.interL * 1e6:.1f}um',
            'flutL': f'{self.flutL * 1e6:.1f}um',
            'nlayers': f'{self.nlayers}layer{plural(self.nlayers)}'
        }

    def str_geometry(self):
        ''' Format model geometrical parameters into string. '''
        return f'fiberD = {si_format(self.fiberD, 1)}m'

    def periaxonalResistancePerUnitLength(self, d, w):
        ''' Compute the periaxonal axial resistance per unit length of a cylindrical section.

            :param d: section inner diameter (m)
            :param w: periaxonal space width (m)
            :return: resistance per unit length (Ohm/cm)
        '''
        return self.axialResistancePerUnitLength(self.rhop, d + 2 * w, d_in=d)

    def construct(self):
        ''' Create and connect node sections with assigned membrane dynamics. '''
        self.createSections()
        self.setGeometry()
        self.setResistivity()
        self.setInterBiophysics()
        self.setExtracellular()
        self.setTopology()

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
            Cm0=self.pneuron.Cm0 * 1e2) for k in self.nodeIDs}
        self.mysas = {k: self.createSection(k, Cm0=self.C_inter) for k in self.mysaIDs}
        self.fluts = {k: self.createSection(k, Cm0=self.C_inter) for k in self.flutIDs}
        self.stins = {k: self.createSection(k, Cm0=self.C_inter) for k in self.stinIDs}

    def sectionsdict(self):
        return {
            'node': self.nodes,
            'MYSA': self.mysas,
            'FLUT': self.fluts,
            'STIN': self.stins
        }

    def get(self, sectype, pname):
        ''' Retrieve the value of a parameter for a specific section type.

            :param sectype: section type ('node', 'MYSA', 'FLUT' or 'STIN')
            :param pname: name of the parameter (e.g. 'L', 'D')
            :return: parameter value
        '''
        return getattr(self, f'{sectype.lower()}{pname}')

    def diameterRatio(self, sectype):
        ''' Return the ratio between a section's type diameter and the fiber outer diameter. '''
        return self.get(sectype, 'D') / self.fiberD

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
            d, L = self.get(sectype, 'D'), self.get(sectype, 'L')
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

    def setInterBiophysics(self):
        ''' Assign biophysics to internodal sections.

            .. note:: if the axoplasm correction level is selected, internodal sections
            leakage conductance and capacitance per unit area are multiplied by the ratio
            of the section's real diameter divided by its assigned diameter (i.e. the
            fiber's outer diameter), to ensure correct sections extensive membrane properties.
        '''
        logger.debug('setting internodal biophysics')
        for sectype, secdict in self.inters.items():
            g = getattr(self, f'g_{sectype.lower()}')
            if self.correction_level == 'axoplasm':
                for sec in secdict.values():
                    sec.cm *= self.diameterRatio(sectype)
                g *= self.diameterRatio(sectype)
            for sec in secdict.values():
                sec.insertPassiveMech(g, self.pneuron.Vm0)

    def setExtracellular(self):
        ''' Set the sections' extracellular mechanisms.

            .. note:: if the myelin correction level is selected, internodal sections
            myelin conductance and capacitance per unit area are divided by the ratio
            of the section's diameter divided by the fiber's outer diameter, to ensure
            correct myelin extensive membrane properties.
        '''
        logger.debug('setting extracellular mechanisms')
        for sec in self.nodes.values():
            sec.insertVext(xr=self.Rp_node)
        for sectype, secdict in self.inters.items():
            xr = getattr(self, f'Rp_{sectype.lower()}')
            xc, xg = self.C_myelin, self.g_myelin
            if self.correction_level == 'myelin':
                xc /= self.diameterRatio(sectype)
                xg /= self.diameterRatio(sectype)
            for sec in secdict.values():
                sec.insertVext(xr=xr, xc=xc, xg=xg)

    def connect(self, k1, i1, k2, i2):
        self.sections[k2][f'{k2}{i2:d}'].connect(self.sections[k1][f'{k1}{i1:d}'])

    def setTopology(self):
        ''' Connect the sections together '''
        logger.debug('connecting sections together')
        for i in range(self.nnodes - 1):
            self.connect('node', i, 'MYSA', 2 * i)
            self.connect('MYSA', 2 * i, 'FLUT', 2 * i)
            self.connect('FLUT', 2 * i, 'STIN', self.nstin_per_inter * i)
            for j in range(self.nstin_per_inter - 1):
                self.connect('STIN', self.nstin_per_inter * i + j,
                             'STIN', self.nstin_per_inter * i + j + 1)
            self.connect('STIN', self.nstin_per_inter * (i + 1) - 1, 'FLUT', 2 * i + 1)
            self.connect('FLUT', 2 * i + 1, 'MYSA', 2 * i + 1)
            self.connect('MYSA', 2 * i + 1, 'node', i + 1)

    def setFuncTables(self, *args, **kwargs):
        super().setFuncTables(*args, **kwargs)
        logger.debug(f'setting V functable in inter mechanism')
        self.setFuncTable(self.inter_mechname, 'V', self.lkp['V'], self.Aref, self.Qref)

    def setOtherProbes(self):
        return {sectype: {k: {'Vm': v.setProbe('v')} for k, v in secdict.items()}
                for sectype, secdict in self.inters.items()}


class EStimMRGFiber(MRGFiber):

    def __init__(self, *args, **kwargs):
        # Initialize parent class
        super().__init__(*args, **kwargs)

        # Set invariant function tables
        # self.setFuncTables()

    @property
    def mechname(self):
        return self.pneuron.name

    def initToSteadyState(self):
        ''' Initialize model variables to pre-stimulus resting state values. '''
        h.finitialize(self.pneuron.Vm0)  # nC/cm2

    def createSection(self, id, mech=None, states=None, Cm0=None):
        ''' Create a model section with a given id. '''
        if Cm0 is None:
            Cm0 = self.pneuron.Cm0 * 1e2  # uF/cm2
        return MechVSection(mechname=mech, states=states, name=id, cell=self, Cm0=Cm0)

    def setPyLookup(self, *args, **kwargs):
        return IintraNode.setPyLookup(self, *args, **kwargs)

    def titrate(self, source, pp):
        Arange = self.getArange(source)
        x0 = getLogStartPoint(Arange, x=0.2)
        Ithr = threshold(
            lambda x: self.titrationFunc(source.updatedX(-x if source.is_cathodal else x), pp),
            Arange, x0=x0, rel_eps_thr=1e-2, precheck=False)
        if source.is_cathodal:
            Ithr = -Ithr
        return Ithr


class IintraMRGFiber(EStimMRGFiber):

    simkey = 'mrg_Iintra'
    A_range = (1e-12, 1e-7)  # A

    def setDrives(self, source):
        ''' Set distributed stimulation drives. '''
        Iinj = source.computeDistributedAmps(self)['node']
        with np.printoptions(**array_print_options):
            logger.debug(f'Intracellular currents: Iinj = {Iinj} nA')
        self.drives = [IClamp(sec(0.5), I) for sec, I in zip(self.nodelist, Iinj)]


class IextraMRGFiber(EStimMRGFiber):

    simkey = 'mrg_Iextra'
    A_range = (1e0, 1e5)  # mV
    use_equivalent_currents = True

    def toInjectedCurrents(self, Ve):
        ''' Convert extracellular potential array into equivalent injected currents.

            :param Ve: model-sized vector of extracellular potentials (mV)
            :return: model-sized vector of intracellular injected currents (nA)
        '''
        # Extract and reshape STIN Ve to (6 x ninters) array to facilitate downstream computations
        Ve_stin = Ve['STIN'].reshape((self.nstin_per_inter, self.ninters), order='F')

        # Compute equivalent currents flowing across sections (in nA)
        I_node_mysa = np.vstack((  # currents flowing from nodes to MYSA compartments
            Ve['node'][:-1] - Ve['MYSA'][::2],  # node -> right-side MYSA
            Ve['node'][1:] - Ve['MYSA'][1::2]   # left-side MYSA <- node
        )) * 2 / (self.R_node + self.R_mysa) * self.mA_to_nA
        I_mysa_flut = np.vstack((  # currents flowing from MYSA to FLUT compartments
            Ve['MYSA'][::2] - Ve['FLUT'][::2],    # right-side MYSA -> right-side FLUT
            Ve['MYSA'][1::2] - Ve['FLUT'][1::2],  # left-side FLUT <- left-side MYSA
        )) * 2 / (self.R_mysa + self.R_flut) * self.mA_to_nA
        I_flut_stin = np.vstack((  # currents flowing from FLUT to boundary STIN compartments
            Ve['FLUT'][::2] - Ve_stin[0],   # right-side FLUT -> right-side boundary STIN
            Ve['FLUT'][1::2] - Ve_stin[-1]  # left-side boundary STIN <- left-side FLUT
        )) * 2 / (self.R_flut + self.R_stin) * self.mA_to_nA
        I_stin_stin = np.diff(Ve_stin, axis=0) / self.R_stin * self.mA_to_nA  # STIN -> STIN

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

    def setDrives(self, source):
        ''' Set distributed stimulation drives. '''
        Ve_dict = source.computeDistributedAmps(self)
        logger.debug(f'Extracellular potentials:')
        with np.printoptions(**array_print_options):
            for k, Ve in Ve_dict.items():
                logger.debug(f'{k}: Ve = {Ve} mV')
        self.drives = []
        if self.use_equivalent_currents:
            # Variant 1: inject equivalent intracellular currents
            Iinj = self.toInjectedCurrents(Ve_dict)
            for k, secdict in self.sections.items():
                self.drives += [IClamp(sec(0.5), I) for sec, I in zip(
                    secdict.values(), Iinj[k])]
        else:
            # Variant 2: insert extracellular mechanisms for a more realistic depiction
            # of the extracellular field
            for k, secdict in self.sections.items():
                self.drives += [ExtField(sec, Ve) for sec, Ve in zip(secdict.values(), Ve_dict[k])]

    def simulate(self, source, pp):
        return super().simulate(source, pp, dt=1e-5)

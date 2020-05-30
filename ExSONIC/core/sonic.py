# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-30 21:40:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-05-30 14:05:19

from neuron import h
import numpy as np

from PySONIC.core import PointNeuron, NeuronalBilayerSonophore, AcousticDrive, ElectricDrive
from PySONIC.utils import logger, isWithin

from ..constants import *
from ..utils import array_print_options, seriesGeq
from .sources import AcousticSource
from .connectors import SerialConnectionScheme
from .pyhoc import getCustomConnectSection, Vector, Matrix

from .nmodel import NeuronModel, SpatiallyExtendedNeuronModel


def addSonicFeatures(Base):

    # Check that the base class inherits from NeuronModel class
    assert issubclass(Base, NeuronModel), 'Base class must inherit from "NeuronModel" class'

    class SonicBase(Base):
        ''' Generic class inheriting from a NeuronModel class and adding gneric SONIC features. '''

        def __init__(self, *args, a=None, fs=1., d=0., **kwargs):
            ''' Initialization.

                :param a: sonophore diameter (m)
                :param fs: sonophore membrane coverage fraction (-)
                :param d: embedding depth (m)
            '''
            # Set point neuron attribute
            try:
                # Retrieve class default pneuron if existing
                self.pneuron = Base._pneuron
            except AttributeError:
                # Otherwise, take first initialization argument
                self.pneuron = args[0]
            self.fs = fs
            self.d = d
            self.a = a
            self.fref = None
            self.pylkp = None
            super().__init__(*args, **kwargs)

        @property
        def nbls(self):
            if hasattr(self, '_nbls'):
                return self._nbls
            else:
                return None

        @nbls.setter
        def nbls(self, value):
            if not isinstance(value, NeuronalBilayerSonophore):
                raise TypeError(f'{value} is not a valid NeuronalBilayerSonophore instance')
            self.set('nbls', value)

        @property
        def pneuron(self):
            if self.nbls is not None:
                return self.nbls.pneuron
            else:
                return self._pneuron

        @pneuron.setter
        def pneuron(self, value):
            if not isinstance(value, PointNeuron):
                raise TypeError(f'{value} is not a valid PointNeuron instance')
            if self.nbls is not None:
                self.nbls = NeuronalBilayerSonophore(self.a, value, self.d)
            else:
                self._pneuron = value

        @property
        def a(self):
            if self.nbls is not None:
                return self.nbls.a
            else:
                return None

        @a.setter
        def a(self, value):
            if value is not None:
                self.nbls = NeuronalBilayerSonophore(value, self.pneuron, self.d)

        def checkForSonophoreRadius(self):
            if self.a is None:
                raise ValueError('Cannot apply acoustic stimulus: sonophore radius not specified')

        @property
        def fs(self):
            if self.nbls is not None:
                return self._fs
            else:
                return None

        @fs.setter
        def fs(self, value):
            if value is None:
                value = 1.
            value = isWithin('fs', value, (0., 1.))
            self.set('fs', value)

        @property
        def d(self):
            if self.nbls is not None:
                return self.nbls.d
            else:
                return self._d

        @d.setter
        def d(self, value):
            if self.a is not None:
                self.nbls = NeuronalBilayerSonophore(self.a, self.pneuron, value)
            else:
                self._d = value

        def setPyLookup(self, f=None):
            if f is not None:
                if self.pylkp is None or f != self.fref:
                    self.pylkp = self.nbls.getLookup2D(f, self.fs)
                    self.fref = f
            else:
                super().setPyLookup()

        @property
        def a_str(self):
            return self.nbls.a_str

        @property
        def fs_str(self):
            return f'{self.fs * 1e2:.0f}%'

        def __repr__(self):
            s = super().__repr__()
            if self.nbls is not None:
                s = f'{s[:-1]}, a={self.a_str}, fs={self.fs_str})'
            return s

        @property
        def meta(self):
            return {
                **super().meta,
                'a': self.a,
                'fs': self.fs,
                'd': self.d
            }

        @property
        def modelcodes(self):
            d = super().modelcodes
            if self.nbls is not None:
                d.update({
                    'a': f'{self.a * M_TO_NM:.0f}nm',
                    'fs': f'fs{self.fs_str}' if self.fs <= 1 else None
                })
            return d

        def titrate(self, obj, pp, **kwargs):
            if isinstance(obj, AcousticDrive) or isinstance(obj, AcousticSource):
                self.setFuncTables(obj.f)  # pre-loading lookups to have a defined Arange
            return super().titrate(obj, pp, **kwargs)

    class SonicNode(SonicBase):

        @property
        def simkey(self):
            if self.nbls is not None:
                return self.nbls.simkey
            else:
                return super().simkey

        @property
        def drives(self):
            if self.drive is None:
                return []
            else:
                return [self.drive]

        def setUSDrive(self, drive):
            ''' Set US drive. '''
            self.checkForSonophoreRadius()
            self.setFuncTables(drive.f)
            self.section.setMechValue('Adrive', drive.A * PA_TO_KPA)
            return None

        @property
        def drive_funcs(self):
            d = super().drive_funcs
            if self.nbls is not None:
                d.update({AcousticDrive: self.setUSDrive})
            return d

        @property
        def Arange_funcs(self):
            d = super().Arange_funcs
            if self.nbls is not None:
                d.update({AcousticDrive: self.nbls.getArange})
            return d

        @property
        def meta(self):
            return {**super().meta, 'method': 'NEURON'}

        def filecodes(self, *args):
            if isinstance(args[0], ElectricDrive):
                return super().filecodes(*args)
            else:
                args = list(args) + [self.fs, 'NEURON', None]
                return self.nbls.filecodes(*args)

    class SonicMorpho(SonicBase):

        # use_iax_pp = False

        def __init__(self, *args, **kwargs):
            self.connection_scheme = SerialConnectionScheme(vref=f'Vm', rmin=self.rmin)
            self.network = None
            super().__init__(*args, **kwargs)

        def copy(self):
            other = super().copy()
            other.connection_scheme = self.connection_scheme
            return other

        @property
        def connection_scheme(self):
            return self._connection_scheme

        @connection_scheme.setter
        def connection_scheme(self, value):
            if value is not None and not isinstance(value, SerialConnectionScheme):
                raise ValueError(f'{value} is not a SerialConnectionScheme object')
            self.set('connection_scheme', value)

        def getSectionClass(self, *args, **kwargs):
            sec_class = super().getSectionClass(*args, **kwargs)
            if self.connection_scheme is None:
                return sec_class
            return getCustomConnectSection(sec_class)

        def createSection(self, *args, **kwargs):
            return super().createSection(args[0], self.connection_scheme, *args[1:], **kwargs)

        @staticmethod
        def getMetaArgs(meta):
            args, kwargs = Base.getMetaArgs(meta)
            kwargs.update({k: meta[k] for k in ['a', 'fs']})
            return args, kwargs

        def createSections(self):
            super().createSections()
            self.initNetworkArrays(self.nsections)

        def initNetworkArrays(self, n):
            ''' Initialize arrays that will be used to set the voltage network.

                Considering the following terms:
                - vi: intracelular voltage
                - vm: transmembrane voltage
                - vx: extracellular voltage
                - ex: imposed voltage outside of the surrounding extracellular membrane
                - is: stimulating current
                - cm: membrane capacitance
                - i(vm): transmembrane ionic current
                - ga: intracellular axial conductance between nodes
                - cx: capacitance of surrounding extracellular membrane (i.e. myelin)
                - gx: passive transverse conductance of surrounding extracellular membrane (i.e. myelin)
                - gp: extracellular axial conductance between nodes (e.g. periaxonal space)
                - j: index indicating the connection to a neighboring node

                Governing equations for internal and external nodes are, respectively:
                (1) cm * dvm/dt + i(vm) = is + ga_j * (vi_j - vi)
                (2) cx * dvx/dt + gx * (vx - ex) = cm * dvm/dt + i(vm) + gp_j * (vx_j - vx)

                Putting all voltage dependencies on the left-hand sides, and developing, we find:
                (1) cm * dvm/dt + ga_j * vi - ga_j * vi_j = is - i(vm)
                (2) cx * dvx/dt - cm * dvm/dt + (gx + gp_j) * vx - gp_j * vx_j = i(vm) + gx * ex

                Re-expressing vi as (vm + vx), we find the matrix equation rows for the two nodes:
                (1) cm * dvm/dt + ga_j * (vm + vx) - ga_j * (vm_j + vx_j) = is - i(vm)
                (2) cx * dvx/dt - cm * dvm/dt + (gx + gp_j) * vx - gp_j * vx_j = i(vm) + gx * ex

                ----------- VARIANT: removing membrane derivatives in second term -----------

                Inserting the equivalence of (1) into (2), we obtain:
                (1) cm * dvm/dt + i(vm) = is + ga_j * (vi_j - vi)
                (2) cx * dvx/dt + gx * (vx - ex) = is + ga_j*(vi_j - vi) + gp_j * (vx_j - vx)

                Putting all voltage dependencies on the left-hand sides, and developing, we find:
                (1) cm * dvm/dt + ga_j * vi - ga_j * vi_j = is - i(vm)
                (2) cx * dvx/dt + (gx + gp_j) * vx - gp_j * vx_j + ga_j * vi - ga_j * vi_j = gx * ex + is

                Re-expressing vi as (vm + vx), we find:
                (1) cm * dvm/dt + ga_j * (vm + vx) - ga_j * (vm_j + vx_j) = is - i(vm)
                (2) cx * dvx/dt + (gx + gp_j) * vx - gp_j * vx_j + ga_j * (vm + vx) - ga_j * (vm_j + vx_j) = gx * ex + is

                Developing each voltage separately, we find the matrix equation rows for the two nodes:
                (1) cm * dvm/dt + ga_j * vm - ga_j * vm_j + ga_j * vx - ga_j * vx_j = is - i(vm)
                (2) cx * dvx/dt + (gx + gp_j + ga_j) * vx - (gp_j + ga_j) * vx_j + ga_j * vm - ga_j * vm_j = gx * ex + is
            '''
            self.c_mat = Matrix(n, n)  # capacitance matrix (mF/cm2)
            self.g_mat = Matrix(n, n)  # conductance matrix (S/cm2)
            self.v_vec = Vector(n)     # voltage vector (mV)
            self.i_vec = Vector(n)     # current vector (mA/cm2)

        @property
        def network_size(self):
            return self.v_vec.size()

        def expandNetworkArrays(self):
            self.gx_vec = np.zeros(self.network_size)  # ext. transverse conductance vector (S/cm2)
            self.Gp_vec = np.zeros(self.network_size)  # longitudinal half-conductance vector (S)
            self.c_mat.expand(2)
            self.g_mat.expand(2)
            self.v_vec.expand(2)
            self.i_vec.expand(2)

        def getSecIndex(self, sec):
            ''' Get index of section in the model's section list. '''
            return self.seclist.index(sec)

        def setGeometry(self):
            super().setGeometry()
            self.Am_vec = np.array([sec.membraneArea() for sec in self.seclist])  # cm2

        def setResistivity(self):
            super().setResistivity()
            self.Ga_vec = np.array([sec.Ga_half for sec in self.seclist])

        def setBiophysics(self):
            super().setBiophysics()
            self.Cm_vec = np.array([sec.Cm0 for sec in self.seclist])

        def registerConnection(self, sec1, sec2):
            i, j = [self.getSecIndex(x) for x in [sec1, sec2]]
            self.connection_pairs.append((i, j))

        def addLongitudinalLink(self, i, j, G, offset1=0, offset2=0):
            ''' Add an longitudinal link between two nodes in the conductance matrix.

                :param i: first section index
                :param j: second section index
                :param g: connection conductance (S)
            '''
            self.g_mat.addval(i + offset1, i + offset2, G / self.Am_vec[i])
            self.g_mat.addval(i + offset1, j + offset2, -G / self.Am_vec[i])
            self.g_mat.addval(j + offset1, j + offset2, G / self.Am_vec[j])
            self.g_mat.addval(j + offset1, i + offset2, -G / self.Am_vec[j])

        def addIntracellularLink(self, i, j):
            self.addLongitudinalLink(  # LHS-1: ga_j * vm - ga_j * vm_j
                i, j, seriesGeq(self.Ga_vec[i] / self.Cm_vec[i], self.Ga_vec[j] / self.Cm_vec[j]),
                offset1=0, offset2=0)
            if self.has_ext_mech:
                self.addLongitudinalLink(  # LHS-1: ga_j * vx - ga_j * vx_j
                    i, j, seriesGeq(self.Ga_vec[i], self.Ga_vec[j]),
                    offset1=0, offset2=self.nsections)

        def addExtracellularLink(self, i, j):
            self.addLongitudinalLink(  # LHS-2: gp_j * vx - gp_j * vx_j
                i, j, seriesGeq(self.Gp_vec[i], self.Gp_vec[j]),
                offset1=self.nsections, offset2=self.nsections)
            # Ga_ij = seriesGeq(self.Ga_vec[i], self.Ga_vec[j])
            # self.addLongitudinalLink(  # LHS-2: ga_j * vx - ga_j * vx_j
            #     i, j, Ga_ij, offset1=self.nsections, offset2=self.nsections)
            # self.addLongitudinalLink(  # LHS-2: ga_j * vm - ga_j * vm_j
            #     i, j, Ga_ij, offset1=0, offset2=0)

        def addExtracellularTransverseWeight(self, i):
            self.g_mat.addval(  # LHS-2: gx * vx
                i + self.nsections, i + self.nsections, self.gx_vec[i])

        def setTopology(self):
            self.connection_pairs = []
            super().setTopology()

        def getOrderedSecIndexes(self):
            l = []
            for i, j in self.connection_pairs:
                if len(l) == 0:
                    l = [i, j]
                else:
                    if i in l and j in l:
                        raise ValueError('pair error: both indexes already in list')
                    elif i not in l and j not in l:
                        raise ValueError('pair error: no index in list')
                    elif i in l:
                        l.insert(l.index(i) + 1, j)
                    elif j in l:
                        l.insert(l.index(j), i)
            return l

        def printTopology(self):
            ''' Print the model's topology. '''
            logger.info('topology:')
            print('\n|\n'.join([self.seclist[i].shortname() for i in self.getOrderedSecIndexes()]))

        def setExtracellularNode(self, sec):
            ''' Register the extracellular voltage node of a specific sections by updating
                the appropriate elements of the extracellular network matrices and vectors.

                :param sec: section object
            '''
            if self.network_size == self.nsections:
                self.expandNetworkArrays()
            i = self.getSecIndex(sec)
            self.gx_vec[i] = sec._xg      # S/cm2
            self.Gp_vec[i] = sec.Gp_half  # S
            self.c_mat.addval(  # LHS-2: cx * dvx/dt
                i + self.nsections, i + self.nsections, sec._xc * UF_CM2_TO_MF_CM2)  # mF/cm2
            self.c_mat.addval(  # LHS-2: -cm * dvm/dt
                i + self.nsections, i, -sec.Cm0 * UF_CM2_TO_MF_CM2)  # mF/cm2

        def getVextRef(self, sec):
            return self.v_vec._ref_x[self.getSecIndex(sec) + self.nsections]

        def printNetwork(self):
            logger.info('c_mat:')
            self.c_mat.printf()
            logger.info('g_mat:')
            self.g_mat.printf()

        def initNetwork(self):
            ''' Initialize a linear mechanism to represent the voltage network. '''
            if self.network is None:
                self.update_network_lhs()
                self.v0_vec = h.Vector(self.v_vec.size())  # initial conditions vector
                self.network = h.LinearMechanism(
                    self.update, self.c_mat, self.g_mat, self.v_vec, self.v0_vec, self.i_vec,
                    h.SectionList(self.seclist), h.Vector([0.5] * self.nsections))

        def update_Cm_vec(self):
            for i, sec in enumerate(self.seclist):
                self.Cm_vec[i] = sec.getValue('v') / sec.getVm(x=0.5)

        def update_network_lhs(self):
            ''' Update the left-hand side of the network (conductance matrix). '''
            self.g_mat.zero()
            for i, j in self.connection_pairs:
                self.addIntracellularLink(i, j)
            if self.has_ext_mech:
                for i in range(self.nsections):
                    self.addExtracellularTransverseWeight(i)
                for i, j in self.connection_pairs:
                    self.addExtracellularLink(i, j)

        def update_network_rhs(self):
            ''' Update the right-hand side of the network (current vector). '''
            for i, sec in enumerate(self.seclist):
                # Propagate e_ext discontinuities into v_vec
                if sec.ex != sec.ex_last:
                    self.v_vec.x[i + self.nsections] += sec.ex - sec.ex_last
                    sec.ex_last = sec.ex
                # RHS-2: gx * ex + im
                self.i_vec.x[i + self.nsections] = self.gx_vec[i] * sec.ex + sec.getIm(x=0.5)

                # RHS-2: gx * ex + is
                # self.i_vec.x[i + self.nsections] = self.gx_vec[i] * sec.ex + sec.iStimDensity()

        def update(self):
            ''' LinearMechanism callback that updates the conductance network. '''
            self.update_Cm_vec()
            self.update_network_lhs()
            self.update_network_rhs()

        def setUSDrives(self, A_dict):
            logger.debug(f'Acoustic pressures:')
            with np.printoptions(**array_print_options):
                for k, amps in A_dict.items():
                    logger.debug(f'{k}: A = {amps * PA_TO_KPA} kPa')
            for k, amps in A_dict.items():
                for A, sec in zip(amps, self.sections[k].values()):
                    sec.setMechValue('Adrive', A * PA_TO_KPA)
            return []

        @property
        def drive_funcs(self):
            d = super().drive_funcs
            if self.nbls is not None:
                d.update({AcousticSource: self.setUSDrives})
            return d

        def setDrives(self, source):
            if isinstance(source, AcousticSource):
                self.checkForSonophoreRadius()
                self.setFuncTables(source.f)
            super().setDrives(source)
            self.initNetwork()

        def needsFixedTimeStep(self, _):
            return True

        @property
        def Aranges(self):
            d = super().Aranges
            if self.nbls is not None:
                d.update({AcousticSource: self.nbls.getArange(None)})
            return d

    # Choose subclass depending on input class
    if issubclass(Base, SpatiallyExtendedNeuronModel):
        SonicClass = SonicMorpho
    else:
        SonicClass = SonicNode

    # Correct class name for consistency with input class
    SonicClass.__name__ = f'{Base.__name__}'

    # Add original class as an attribute of the new decorated class (with modified simkey)
    class Original(Base):
        simkey = f'original_{Base.simkey}'
    Original.__name__ = f'Original{Base.__name__}'
    SonicClass.__original__ = Original

    # Return SONIC-enabled class
    return SonicClass

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-30 21:40:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-06-06 01:25:34

from neuron import h
import numpy as np

from PySONIC.core import PointNeuron, NeuronalBilayerSonophore, AcousticDrive, ElectricDrive
from PySONIC.utils import logger, isWithin

from ..constants import *
from ..utils import array_print_options
from .sources import AcousticSource
from .connectors import SerialConnectionScheme
from .pyhoc import *

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
            self.initNetworkArrays()

        def setGeometry(self):
            super().setGeometry()
            self.Am_vec = np.array([sec.membraneArea() for sec in self.seclist])  # cm2

        def setResistivity(self):
            super().setResistivity()
            self.Ga_vec = np.array([sec.Ga_half for sec in self.seclist])

        def setBiophysics(self):
            super().setBiophysics()
            self.cm_vec = np.array([sec.Cm0 for sec in self.seclist])

        def registerConnection(self, sec1, sec2):
            self.connection_pairs.append(tuple([self.seclist.index(x) for x in [sec1, sec2]]))

        def setTopology(self):
            self.connection_pairs = []
            super().setTopology()
            self.isec_ordered = self.getOrderedSecIndexes()
            self.ordered_seclist = [self.seclist[i] for i in self.isec_ordered]

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
            print('\n|\n'.join([sec.shortname() for sec in self.ordered_seclist]))

        def getSecIndex(self, sec):
            ''' Get index of section in a model's section list. '''
            return self.seclist.index(sec)

        def initNetworkArrays(self):
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
                - gx: transverse conductance of surrounding extracellular membrane (i.e. myelin)
                - gp: extracellular axial conductance between nodes (e.g. periaxonal space)
                - j: index indicating the connection to a neighboring node

                Governing equations for internal and external nodes are, respectively:

                (1) cm * dvm/dt + i(vm) = is + ga_j * (vi_j - vi)
                (2) cx * dvx/dt + gx * (vx - ex) = cm * dvm/dt + i(vm) + gp_j * (vx_j - vx)

                Putting all voltage dependencies on the left-hand sides, and developing, we find:

                (1) cm * dvm/dt + ga_j * (vi - vi_j) = is - i(vm)
                (2) cx * dvx/dt - cm * dvm/dt + gx * vx + gp_j * (vx - vx_j) = i(vm) + gx * ex

                Re-expressing vi as (vm + vx), we find the matrix equation rows for the two nodes:

                (1) cm * dvm/dt + ga_j * (vm - vm_j) + ga_j * (vx - vx_j) = is - i(vm)
                (2) cx * dvx/dt - cm * dvm/dt + gx * vx + gp_j * (vx - vx_j) = i(vm) + gx * ex

                or in developed form:

                (1) cm * dvm/dt + ga_j * vm - ga_j * vm_j + ga_j * vx - ga_j * vx_j = is - i(vm)
                (2) cx * dvx/dt - cm * dvm/dt + gx * vx + gp_j * vx - gp_j * vx_j = i(vm) + gx * ex

                Special attention must be brought on the fact that we use NEURONS's v variable as an
                alias for the section's membrane charge density Qm. Therefore, the system must be
                adapted in consequence. To this effect, we introduce a change of variable:

                Qm = cm * vm; dQm/dt = cm * dvm/dt  (cm considered time-invariant)

                Recasting the system according to this change of variable thus yields:

                (1) dQm/dt
                    + ga_j/cm * Qm - ga_j/cm_j * Qm_j
                    + ga_j * vx - ga_j * vx_j
                    = is - i(vm)
                (2) cx * dvx/dt - dQm/dt
                    + gx * vx
                    + gp_j * vx - gp_j * vx_j
                    = i(vm) + gx * ex

                Finally, we replace dQm/dt + i(vm) in (2) by equivalents intracellular axial and
                stimulation currents, in order to remove the need to access net membrane current.
                After re-arranging all linear voltage terms on the left-hand side, we have:

                (1) dQm/dt
                    + ga_j/cm * Qm - ga_j/cm_j * Qm_j
                    + ga_j * vx - ga_j * vx_j
                    = is - i(vm)
                (2) cx * dvx/dt
                    + ga_j/cm * Qm - ga_j/cm * Qm_j
                    + ga_j * vx - ga_j * vx_j
                    + gp_j * vx - gp_j * vx_j
                    + gx * vx
                    = is + gx * ex

                Among these equation rows, 2 terms are automatically handled by NEURON, namely:
                - LHS-1: dQm/dt
                - RHS-1: is - i(vm)

                Hence, 8 terms remain to be handled as part of the additional network:
                - LHS-1: ga_j/cm * Qm - ga_j/cm_j * Qm_j
                - LHS-1: ga_j * vx - ga_j * vx_j
                - LHS-2: cx * dvx/dt
                - LHS-2: ga_j/cm * Qm - ga_j/cm_j * Qm_j
                - LHS-2: ga_j * vx - ga_j * vx_j
                - LHS-2: gp_j * (vx - vx_j)
                - LHS-2: gx * vx
                - RHS-2: is + gx * ex

                Note that axial conductance terms (ga and gp) must be normalized by the
                appropriate section membrane area in order to obtain a consistent system.

                How do we do this?

                The above equations describe a partial differential system of the form:

                    C * dy/dt + G * y = I

                with C a capacitance matrix, G a conductance matrix, y a hybrid vector
                of transmembrane charge density and external voltage, and I a current vector.

                Thankfully, NEURON provides a so-called "LinearMechanism" class that allows
                to define such a system, where the first n elements of the y vector
                can be mapped to the "v" variable of a specific list of sections. This allows
                us to add a linear network atop of NEURON's native network, in which we define
                the additional terms.

                Note also that the gx conductance is connected in series with another transverse
                conductance of value 1e9, to mimick NEURON's default 2nd extracellular layer.
            '''
            self.c_mat = SquareMatrix(self.nsections)  # capacitance matrix (mF/cm2)
            self.g_mat = SquareMatrix(self.nsections)  # conductance matrix (S/cm2)
            self.v_vec = Vector(self.nsections)     # voltage vector (mV)
            self.i_vec = Vector(self.nsections)     # current vector (mA/cm2)

        @property
        def network_size(self):
            ''' Return size of linear mechanism network. '''
            return self.v_vec.size()

        def expandNetworkArrays(self):
            ''' Expand the network arrays to add a connected extracellular layer. '''
            self.cx_vec = np.zeros(self.network_size)  # ext. transverse capacitance vector (uF/cm2)
            self.gx_vec = np.zeros(self.network_size)  # ext. transverse conductance vector (S/cm2)
            self.Gp_vec = np.zeros(self.network_size)  # longitudinal half-conductance vector (S)
            self.c_mat.expand()
            self.g_mat.expand()
            self.v_vec.expand()
            self.i_vec.expand()

        def setExtracellularNode(self, sec):
            ''' Register the extracellular voltage node of a specific section.

                :param sec: section object
            '''
            if self.network_size == self.nsections:
                self.expandNetworkArrays()
            i = self.getSecIndex(sec)
            self.cx_vec[i] = sec.cx       # uF/cm2
            self.gx_vec[i] = sec.gx       # S/cm2
            self.Gp_vec[i] = sec.Gp_half  # S

        def getVextRef(self, sec):
            return self.v_vec._ref_x[self.getSecIndex(sec) + self.nsections]

        def updateCmat(self):
            ''' Set the network capacitance matrix. '''
            self.c_mat.zero()
            # Lower-right diagonal: cx
            self.c_mat.setdiag(0, Vector(np.hstack((
                np.zeros(self.nsections), self.cx_vec * UF_CM2_TO_MF_CM2))))

        def updateCm(self):
            for i, sec in enumerate(self.seclist):
                self.cm_vec[i] = sec.getCm(x=0.5)

        @property
        def ga_mat(self):
            ''' Ga matrix normalized by Am. '''
            return self.Ga_mat.mulByRow(1 / self.Am_vec)

        @property
        def gacm_mat(self):
            ''' Ga matrix normalized by cm * Am. '''
            return self.Ga_mat.mulByRow(1 / (self.cm_vec * self.Am_vec))

        @property
        def gp_mat(self):
            ''' Gp matrix normalized by Am. '''
            return self.Gp_mat.mulByRow(1 / self.Am_vec)

        @property
        def gx_mat(self):
            return DiagonalMatrix(self.gx_vec)

        def updateGmat(self):
            ''' Update the network conductance matrix. '''
            self.g_mat.zero()
            # LHS-1: ga_j/cm * Qm - ga_j/cm_j * Qm_j  (upper-left)
            self.gacm_mat.addTo(self.g_mat, 0, 0)
            if self.has_ext_mech:
                # # LHS-1: ga_j * (vx - vx_j)  (upper-right)
                self.ga_mat.addTo(self.g_mat, 0, self.nsections)
                # LHS-2: gp_j * (vx - vx_j)    (lower-right)
                self.gp_mat.addTo(self.g_mat, self.nsections, self.nsections)
                # LHS-2: gx * vx    (lower-right)
                self.gx_mat.addTo(self.g_mat, self.nsections, self.nsections)
                # LHS-2: ga_j/cm * Qm - ga_j/cm_j * Qm_j    (lower-left)
                self.gacm_mat.addTo(self.g_mat, self.nsections, 0)
                # LHS-2: ga_j * vx - ga_j * vx_j (lower-right)
                self.ga_mat.addTo(self.g_mat, self.nsections, self.nsections)

        def updateRHS(self):
            ''' Update the right-hand side of the network (current vector). '''
            for i, sec in enumerate(self.seclist):
                # Propagate e_ext discontinuities into v_vec
                if sec.ex != sec.ex_last:
                    self.v_vec.x[i + self.nsections] += sec.ex - sec.ex_last
                    sec.ex_last = sec.ex
                # RHS-2: gx * ex + is
                self.i_vec.x[i + self.nsections] = self.gx_vec[i] * sec.ex + sec.istim

        def updateNetwork(self):
            ''' Update the linear mechanism network components before the call to fadvance. '''
            self.updateCm()
            # self.updateGmat()
            if self.has_ext_mech:
                self.updateRHS()

        def printNetwork(self):
            if self.nsections > 23:
                logger.warning('exceeding max number of sections to print network correctly')
                return
            fmt = '%-8g' if self.nsections <= 11 else '%-4g'
            with np.printoptions(**array_print_options):
                logger.info(f'cm_vec: {self.cm_vec} uF/cm2')
                logger.info(f'cx_vec: {self.cx_vec} uF/cm2')
                logger.info('c_mat (mF/cm2):')
                self.c_mat.printf(fmt)
                logger.info(f'Ga_vec: {self.Ga_vec}')
                logger.info('Ga_mat:')
                self.Ga_mat.printf()
                logger.info('ga_mat:')
                self.ga_mat.printf()
                logger.info(f'Gp_vec: {self.Gp_vec}')
                logger.info('Gp_mat:')
                self.Gp_mat.printf()
                logger.info('gp_mat:')
                self.gp_mat.printf()
                logger.info(f'gx_vec: {self.gx_vec}')
                logger.info(f'g_mat: ({self.g_mat.nrow()} x {self.g_mat.ncol()})')
                self.g_mat.printf(fmt)

        def initNetwork(self):
            ''' Initialize a linear mechanism to represent the voltage network. '''
            self.initToSteadyState()
            self.Ga_mat = ConductanceMatrix(self.Ga_vec, links=self.connection_pairs)
            self.Gp_mat = ConductanceMatrix(self.Gp_vec, links=self.connection_pairs)
            self.updateGmat()
            if self.has_ext_mech:
                self.updateCmat()
            self.updateNetwork()
            self.printNetwork()

            if self.network is None:
                self.v0_vec = Vector(np.zeros(self.nsections * 2))  # initial conditions vector
                self.network = h.LinearMechanism(
                    self.updateNetwork, self.c_mat, self.g_mat, self.v_vec, self.v0_vec, self.i_vec,
                    h.SectionList(self.seclist), Vector([0.5] * self.nsections))

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

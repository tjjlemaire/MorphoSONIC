# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-30 21:40:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-05-19 12:36:33

from neuron import h
import numpy as np

from PySONIC.core import PointNeuron, NeuronalBilayerSonophore, AcousticDrive, ElectricDrive
from PySONIC.utils import logger, isWithin

from ..constants import *
from ..utils import array_print_options
from .sources import AcousticSource
from .connectors import SerialConnectionScheme
from .pyhoc import CustomConnectMechQSection, Matrix

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
            self.has_topology = False
            self.ext_lm = None
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

        @property
        def section_class(self):
            if self.connection_scheme is not None:
                return CustomConnectMechQSection
            else:
                return super().section_class

        def createSection(self, *args, **kwargs):
            return super().createSection(args[0], self.connection_scheme, *args[1:], **kwargs)

        @staticmethod
        def getMetaArgs(meta):
            args, kwargs = Base.getMetaArgs(meta)
            kwargs.update({k: meta[k] for k in ['a', 'fs']})
            return args, kwargs

        def setOrderedSecRefList(self):
            ''' Return a list of ordered section references based on the model's topology. '''
            ordered_secref_list = []
            for sec in self.seclist:
                if not sec.has_parent():
                    ordered_secref_list = [sec]
                    break
            while ordered_secref_list[-1].child_ref is not None:
                ordered_secref_list.append(ordered_secref_list[-1].child_ref.sec)
            if len(ordered_secref_list) < len(self.seclist):
                raise ValueError('There are some unconnected sections in the model...')
            self.ordered_secref_list = ordered_secref_list

        def getSecIndex(self, sec):
            ''' Get index of section in the model's ordered section list. '''
            return self.ordered_secref_list.index(sec)

        def initExtracellularNetworkArrays(self, n):
            ''' Initialize arrays that will be used to set the 1-layer extracellular network.

                Governing equations for an external node are:

                cx * dvx/dt + gx * (vx - ex) = cm * dvm/dt + i(vm) + ax_j * (vx_j - vx)

                which we simplify using the currents balance at the internal node:
                iax = cm * dvm/dt + i(vm)

                cx * dvx/dt + gx * (vx - ex) = iax + ax_j * (vx_j - vx)

                Putting all vx dependencies on the left-hand side, and developing, we find the
                matrix equation row for an external node:

                cx * dvx/dt + (gx + ax_j) * vx - ax_j * vx_j = iax + gx * ex
            '''
            self.cx_mat = Matrix(n, n, 2)     # capacitance matrix (mF/cm2)
            self.gx_mat = Matrix(n, n)        # conductance matrix (S/cm2)
            self.gx_vec = h.Vector(n)         # transverse conductance vector (S/cm2)
            self.ix_vec = h.Vector(n)         # currents (mA/cm2)
            self.vx = h.Vector(n)             # extracellular voltage vector
            self.vx0 = h.Vector(np.zeros(n))  # initial conditions vector

        def initExtracellularNetwork(self):
            ''' Initialize a linear mechanism to represent the extracellular voltage network. '''
            if self.ext_lm is not None:
                raise ValueError('Extracellular Linear Mechanism already exists')
            self.ext_lm = h.LinearMechanism(
                self.vx_callback, self.cx_mat, self.gx_mat, self.vx, self.vx0, self.ix_vec)

        def vx_callback(self):
            ''' LinearMechanism callback that updates the currents vector. '''
            for i, sec in enumerate(self.ordered_secref_list):
                # Propagate e_ext discontinuities into vx
                if sec.ex != sec.ex_last:
                    self.vx.x[i] += sec.ex - sec.ex_last
                    sec.ex_last = sec.ex
                self.ix_vec.x[i] = self.gx_vec.x[i] * sec.ex - sec.iaxDensity()

        def setTopology(self):
            super().setTopology()
            self.setOrderedSecRefList()
            self.initExtracellularNetworkArrays(self.nsections)
            self.has_topology = True

        def setExtracellular(self):
            if not self.has_topology:
                raise ValueError(
                    'Model topology must be defined prior to setting the extracellular network.')
            super().setExtracellular()

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

        def simulate(self, *args, **kwargs):
            self.initExtracellularNetwork()
            return super().simulate(*args, **kwargs)

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

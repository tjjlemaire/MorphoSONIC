# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-30 21:40:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-22 15:18:44

import numpy as np

from PySONIC.core import PointNeuron, NeuronalBilayerSonophore, AcousticDrive, ElectricDrive
from PySONIC.utils import logger, isWithin

from ..constants import *
from ..utils import array_print_options
from ..sources import AcousticSource
from .pyhoc import *
from .cgi_network import HybridNetwork

from .nmodel import NeuronModel, SpatiallyExtendedNeuronModel


def addSonicFeatures(Base):

    # Check that the base class inherits from NeuronModel class
    assert issubclass(Base, NeuronModel), 'Base class must inherit from "NeuronModel" class'

    class SonicBase(Base):
        ''' Generic class inheriting from a NeuronModel class and adding gneric SONIC features. '''

        passive_mechname = CUSTOM_PASSIVE_MECHNAME

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
            self.initargs = (args, kwargs)
            super().__init__(*args, **kwargs)

        def original(self):
            ''' Return an equivalent instance from the original model (not SONIC-adpated). '''
            return self.mirrored(self.__original__)

        def compdict(self, original_key='original', sonic_key='sonic'):
            ''' Return dictionary with the model instance and its "original" equivalent. '''
            return {original_key: self.original(), sonic_key: self}

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

        def setFuncTables(self, *args, **kwargs):
            ''' Add V func table to passive sections, if any. '''
            super().setFuncTables(*args, **kwargs)
            if hasattr(self, 'has_passive_sections') and self.has_passive_sections:
                logger.debug(f'setting {CUSTOM_PASSIVE_MECHNAME} function tables')
                self.setFuncTable(
                    CUSTOM_PASSIVE_MECHNAME, 'V', self.lkp['V'], self.Aref, self.Qref)

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

        def __init__(self, *args, inter_fs=1., **kwargs):
            self.network = None
            if not hasattr(self, 'use_explicit_iax'):
                self.use_explicit_iax = False
            if not hasattr(self, 'gmax'):
                self.gmax = None
            self.inter_fs = inter_fs
            self.inter_pylkp = None
            super().__init__(*args, **kwargs)

        @property
        def inter_fs(self):
            if self.nbls is not None:
                return self._inter_fs
            else:
                return None

        @inter_fs.setter
        def inter_fs(self, value):
            if value is None:
                value = 1.
            value = isWithin('inter_fs', value, (0., 1.))
            self.set('inter_fs', value)

        @property
        def inter_fs_str(self):
            return f'{self.inter_fs * 1e2:.0f}%'

        @property
        def coverages(self):
            ''' Sonophore coverage factor per section type. '''
            return {k: self.fs if k == 'node' else self.inter_fs for k in self.sectypes}

        def __repr__(self):
            s = super().__repr__()
            if self.has_passive_sections and self.nbls is not None:
                s = f'{s[:-1]}, inter_fs={self.inter_fs_str})'
            return s

        @property
        def meta(self):
            d = super().meta
            if self.has_passive_sections:
                d.update({'inter_fs': self.inter_fs})
            return d

        @property
        def modelcodes(self):
            d = super().modelcodes
            if self.nbls is not None and self.has_passive_sections:
                d.update({
                    'inter_fs': f'interfs{self.inter_fs_str}' if self.inter_fs <= 1 else None
                })
            return d

        def setPyLookup(self, f=None):
            ''' Add inter_pylkp if needed. '''
            fref = self.fref  # store fref value before call to parent setPyLookup
            super().setPyLookup(f=f)
            if self.has_passive_sections:
                if f is not None:  # acoustic case: load separate lookup with inter fs
                    if self.inter_pylkp is None or f != fref:
                        self.inter_pylkp = self.nbls.getLookup2D(f, self.inter_fs)
                elif self.inter_pylkp is None:  # Electrical case: copy nodal lookup
                    self.inter_pylkp = self.pylkp.copy()

        def setModLookup(self, *args, **kwargs):
            ''' Add inter lookup for passive sections, if any. '''
            super().setModLookup(*args, **kwargs)
            if self.has_passive_sections:
                _, _, self.inter_lkp = self.Py2ModLookup(self.inter_pylkp)

        def setFuncTables(self, *args, **kwargs):
            ''' Add V func table to passive sections, if any. '''
            super().setFuncTables(*args, **kwargs)
            if self.has_passive_sections:
                logger.debug(f'setting {CUSTOM_PASSIVE_MECHNAME} function tables')
                self.setFuncTable(
                    CUSTOM_PASSIVE_MECHNAME, 'V', self.inter_lkp['V'], self.Aref, self.Qref)

        def clearLookups(self):
            self.inter_lkp = None
            super().clearLookups()

        def copy(self):
            other = super().copy()
            other.network = self.network
            return other

        @property
        def network(self):
            return self._network

        def clear(self):
            if self._network is not None:
                self._network.clear()
            super().clear()

        @network.setter
        def network(self, value):
            if hasattr(self, '_network') and isinstance(self._network, HybridNetwork):
                self._network.clear()
            if value is not None:
                if isinstance(value, HybridNetwork):
                    logger.debug(f'initialized {value}')
                else:
                    raise ValueError(f'network must be a {HybridNetwork.__name__} instance')
            self._network = value

        @property
        def has_network(self):
            return self._network is not None

        def getSectionClass(self, *args, **kwargs):
            return getCustomConnectSection(super().getSectionClass(*args, **kwargs))

        def createSection(self, *args, **kwargs):
            sec = super().createSection(args[0], *args[1:], **kwargs)
            if self.use_explicit_iax and self.gmax is not None:
                sec.gmax = self.gmax
            return sec

        @classmethod
        def getMetaArgs(cls, meta):
            args, kwargs = Base.getMetaArgs(meta)
            additional_kwargs = ['a', 'fs']
            if cls.has_passive_sections:
                additional_kwargs.append('inter_fs')
            kwargs.update({k: meta[k] for k in additional_kwargs})
            return args, kwargs

        def setTopology(self):
            self.connections = []
            super().setTopology()

        def registerConnection(self, sec1, sec2):
            self.connections.append(tuple([self.seclist.index(x) for x in [sec1, sec2]]))

        def getOrderedSecIndexes(self):
            l = []
            for i, j in self.connections:
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

        def getVextRef(self, sec):
            return self.network.getVextRef(sec)

        def setEx(self, *args, **kwargs):
            if self.has_network:
                self.network.setEx(*args, **kwargs)

        def setIstim(self, *args, **kwargs):
            if self.has_network and self.network.has_ext_layer:
                self.network.setIstim(*args, **kwargs)

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

        def initToSteadyState(self):
            self.network.startLM()
            super().initToSteadyState()

        def isDynamicCmSource(self, source):
            return isinstance(source, AcousticSource)

        def setDrives(self, source):
            is_dynamic_cm = False
            if self.isDynamicCmSource(source):
                self.checkForSonophoreRadius()
                self.setFuncTables(source.f)
                is_dynamic_cm = True
            self.network = HybridNetwork(
                self.seclist,
                self.connections,
                self.has_ext_mech,
                is_dynamic_cm=is_dynamic_cm,
                use_explicit_iax=self.use_explicit_iax)
            super().setDrives(source)

        # def needsFixedTimeStep(self, source):
        #     if isinstance(source, AcousticSource):
        #         return True
        #     return super().needsFixedTimeStep(source)

        @property
        def Aranges(self):
            d = super().Aranges
            if self.nbls is not None:
                d.update({AcousticSource: self.nbls.getArange(None)})
            return d

        def computeIax(self, Vi_dict):
            ''' Compute axial currents based on dictionary of intracellular potential vectors.

                :param Vi_dict: dictionary of intracellular potential vectors.
                :return: dictionary of axial currents (mA/m2)
            '''
            Vi_array = np.array([v for v in Vi_dict.values()])
            iax_array = np.array([
                self.network.computeIax(Vi_vec) * M_TO_CM**2 for Vi_vec in Vi_array.T]).T  # mA/m2
            return dict(zip(Vi_dict.keys(), iax_array))

        def addIaxToSolution(self, sp_ext_sol):
            ''' Add axial currents to solution. '''
            Vi_dict = {k: sol['Vin' if 'Vin' in sol else 'Vm'] for k, sol in sp_ext_sol.items()}
            for k, v in self.computeIax(Vi_dict).items():
                sp_ext_sol[k]['iax'] = v  # mA/m2

        def simulate(self, *args, **kwargs):
            ''' Add axial currents to solution. '''
            sp_ext_sol, meta = super().simulate(*args, **kwargs)
            self.addIaxToSolution(sp_ext_sol)
            return sp_ext_sol, meta

    # Choose subclass depending on input class
    if issubclass(Base, SpatiallyExtendedNeuronModel):
        SonicClass = SonicMorpho
    else:
        SonicClass = SonicNode

    # Correct class name for consistency with input class
    SonicClass.__name__ = Base.__name__

    # Add original class as an attribute of the new decorated class (with modified simkey)
    class Original(Base):
        simkey = f'original_{Base.simkey}'
    Original.__name__ = f'Original{Base.__name__}'
    SonicClass.__original__ = Original

    # Return SONIC-enabled class
    return SonicClass

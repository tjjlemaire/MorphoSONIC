# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-27 09:23:32
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-03 17:59:45

from PySONIC.core import PointNeuron, ElectricDrive
from PySONIC.utils import logger

from ..utils import getNmodlDir, load_mechanisms
from .pyhoc import IClamp
from .nmodel import NeuronModel
from .sonic import addSonicFeatures


@addSonicFeatures
class Node(NeuronModel):
    ''' Node model. '''

    def __init__(self, pneuron, construct=True):
        ''' Initialization.

            :param pneuron: point-neuron model
        '''
        self.pneuron = pneuron
        logger.debug('Creating {} model'.format(self))
        load_mechanisms(getNmodlDir(), self.modfile)
        self.clear()
        if construct:
            self.construct()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pneuron})'

    def construct(self):
        ''' Create model sections and set invariant function tables. '''
        self.section = self.createSection(
            'node', mech=self.mechname, states=self.pneuron.statesNames())
        self.setFuncTables()
        self.is_constructed = True

    def clear(self):
        self.pylkp = None
        self.section = None
        self.drive = None

    @property
    def simkey(self):
        return self.pneuron.simkey

    @property
    def seclist(self):
        return [self.section]

    @property
    def drives(self):
        return [self.drive]

    def getAreaNormalizationFactor(self):
        ''' Return area normalization factor '''
        A0 = self.section(0.5).area() * 1e-12  # section area (m2)
        A = self.pneuron.area                  # neuron membrane area (m2)
        return A0 / A

    @staticmethod
    def getNSpikes(data):
        return PointNeuron.getNSpikes(data)

    @property
    def meta(self):
        return self.pneuron.meta

    def desc(self, meta):
        return f'{self}: simulation @ {meta["drive"].desc}, {meta["pp"].desc}'

    def setIClamp(self, drive):
        ''' Set intracellular electrical stimulation drive

            :param drive: electric drive object.
        '''
        Iinj = drive.I * self.section(0.5).area() * 1e-6  # nA
        logger.debug(f'Equivalent injected current: {Iinj:.1f} nA')
        return IClamp(self.section, Iinj)

    @property
    def drive_funcs(self):
        return {ElectricDrive: self.setIClamp}

    def setDrive(self, drive):
        ''' Set stimulation drive.

            :param drive: drive object.
        '''
        logger.debug(f'Stimulus: {drive}')
        self.drive = None
        match = False
        for drive_class, drive_func in self.drive_funcs.items():
            if isinstance(drive, drive_class):
                self.drive = drive_func(drive)
                match = True
        if not match:
            raise ValueError(f'Unknown drive type: {drive}')

    @property
    def Arange_funcs(self):
        return {ElectricDrive: self.pneuron.getArange}

    def getArange(self, drive):
        ''' Get the stimulus amplitude range allowed. '''
        for drive_class, Arange_func in self.Arange_funcs.items():
            if isinstance(drive, drive_class):
                return Arange_func(drive)
        raise ValueError(f'Unknown drive type: {drive}')

    def filecodes(self, *args):
        codes = self.pneuron.filecodes(*args)
        codes['method'] = 'NEURON'
        return codes


class DrivenNode(Node):

    def __init__(self, pneuron, Idrive, *args, **kwargs):
        self.Idrive = Idrive
        Node.__init__(self, pneuron, *args, **kwargs)
        logger.debug(f'setting {self.Idrive:.2f} mA/m2 driving current')
        Iinj = self.Idrive * self.section.membraneArea() * 1e-2  # nA
        logger.debug(f'Equivalent injected current: {Iinj:.1f} nA')
        self.iclamp = IClamp(self.section, Iinj)
        self.iclamp.toggle(1)

    def __repr__(self):
        return super().__repr__()[:-1] + f', Idrive = {self.Idrive:.2f} mA/m2)'

    def filecodes(self, *args):
        codes = Node.filecodes(self, *args)
        codes['Idrive'] = f'Idrive{self.Idrive:.1f}mAm2'
        return codes

    @property
    def meta(self):
        meta = super().meta
        meta['Idrive'] = self.Idrive
        return meta

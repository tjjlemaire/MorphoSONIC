# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-18 21:14:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-06 16:18:16

import matplotlib.pyplot as plt
from PySONIC.parsers import *
from PySONIC.utils import si_format

from .plt import SectionGroupedTimeSeries, SectionCompTimeSeries


class SpatiallyExtendedParser(Parser):

    def __init__(self):
        super().__init__()
        self.defaults.update({'rs': 110.0})
        self.factors.update({'rs': 1e0})
        self.addResistivity()
        self.addSection()

    def addResistivity(self):
        self.add_argument(
            '--rs', nargs='+', type=float, help='Intracellular resistivity (Ohm.cm)')

    def addSection(self):
        self.add_argument(
            '--section', nargs='+', type=str, help='Section of interest for plot')

    def parse(self, args=None):
        if args is None:
            args = super().parse()
        for key in ['rs']:
            args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args

    @staticmethod
    def parseSimInputs(args):
        return [args[k] for k in ['rs']]

    @staticmethod
    def parsePlot(args, output):
        render_args = {}
        if 'spikes' in args:
            render_args['spikes'] = args['spikes']
        if args['section'] == ['all']:
            raise ValueError('sections names must be explicitly specified')
        if args['compare']:
            if args['plot'] == ['all']:
                logger.error('Specific variables must be specified for comparative plots')
                return
            for key in ['cmap', 'cscale']:
                render_args[key] = args[key]
            for pltvar in args['plot']:
                comp_plot = SectionCompTimeSeries(output, pltvar, args['section'])
                comp_plot.render(**render_args)
        else:
            for key in args['section']:
                scheme_plot = SectionGroupedTimeSeries(key, output, pltscheme=args['pltscheme'])
                scheme_plot.render(**render_args)
        plt.show()


class SennParser(SpatiallyExtendedParser):

    def __init__(self):
        super().__init__()
        self.defaults.update({'nnodes': 11, 'fiberD': 20., 'neuron': 'FH'})
        self.factors.update({'fiberD': 1e-6})
        self.addNnodes()
        self.addFiberDiameter()

    def addNnodes(self):
        self.add_argument(
            '--nnodes', nargs='+', type=int, help='Number of nodes of Ranvier')

    def addFiberDiameter(self):
        self.add_argument(
            '-d', '--fiberD', nargs='+', type=float, help='Fiber diameter (um)')

    def parse(self, args=None):
        args = super().parse(args=args)
        for key in ['fiberD']:
            if len(args[key]) > 1 or args[key][0] is not None:
                args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args

    @staticmethod
    def parseSimInputs(args):
        return SpatiallyExtendedParser.parseSimInputs(args)

    def parsePlot(self, args, output):
        if args['section'] == ['all']:
            args['section'] = [f'node{i}' for i in range(args['nnodes'][0])]
        return SpatiallyExtendedParser.parsePlot(args, output)


class EStimSennParser(SennParser, PWSimParser):

    def __init__(self):
        PWSimParser.__init__(self)
        SennParser.__init__(self)
        self.defaults.update({'tstim': 0.1, 'toffset': 3.})
        self.allowed.update({'mode': ['cathode', 'anode']})
        self.addElectrodeMode()
        self.addAstim()

    def addElectrodeMode(self):
        self.add_argument(
            '--mode', type=str, help='Electrode polarity mode ("cathode" or "anode")')

    def addAstim(self):
        self.add_argument(
            '-A', '--amp', nargs='+', type=float,
            help=f'Point-source current amplitude ({self.amp_unit})')
        self.add_argument(
            '--Arange', type=str, nargs='+',
            help='Point-source current amplitude range {} ({})'.format(self.dist_str, self.amp_unit))
        self.to_parse['amp'] = self.parseAmp

    def parseAmp(self, args):
        return EStimParser.parseAmp(self, args)

    def parse(self):
        return SennParser.parse(self, args=PWSimParser.parse(self))

    @staticmethod
    def parseSimInputs(args):
        return PWSimParser.parseSimInputs(args) + SpatiallyExtendedParser.parseSimInputs(args)

    def parsePlot(self, *args):
        return SennParser.parsePlot(self, *args)


class IextSennParser(EStimSennParser):

    amp_unit = 'mA'

    def __init__(self):
        super().__init__()
        self.defaults.update({'xps': 0., 'zps': None, 'mode': 'cathode', 'amp': -0.7})
        self.factors.update({'amp': 1e-3, 'xps': 1e-3, 'zps': 1e-3})
        self.addPointSourcePosition()

    def addPointSourcePosition(self):
        self.add_argument(
            '--xps', nargs='+', type=float, help='Point source x-position (mm)')
        self.add_argument(
            '--zps', nargs='+', type=float, help='Point source z-position (mm)')

    def parse(self):
        args = super().parse()
        for key in ['xps', 'zps']:
            if len(args[key]) > 1 or args[key][0] is not None:
                args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args


class IinjSennParser(EStimSennParser):

    amp_unit = 'nA'

    def __init__(self):
        super().__init__()
        self.defaults.update({'inode': None, 'mode': 'anode', 'amp': 2.0})
        self.factors.update({'amp': 1e-9})
        self.addNodeIClamp()

    def addNodeIClamp(self):
        self.add_argument(
            '--inode', nargs='+', type=int, help='Node index for current clamp')


class SpatiallyExtendedAStimParser(SpatiallyExtendedParser, AStimParser):

    def __init__(self):
        AStimParser.__init__(self)
        SpatiallyExtendedParser.__init__(self)
        self.defaults.pop('method')
        self.allowed.pop('method')
        self.to_parse.pop('method')

    def parse(self):
        args = SpatiallyExtendedParser.parse(self, args=AStimParser.parse(self))
        del args['method']
        return args

    @staticmethod
    def parseSimInputs(args):
        return [args['freq']] + PWSimParser.parseSimInputs(args) + SpatiallyExtendedParser.parseSimInputs(args)

    @staticmethod
    def parsePlot(*args):
        return SpatiallyExtendedParser.parsePlot(*args)


class ExtSonicNodeAStimParser(SpatiallyExtendedAStimParser):

    def __init__(self):
        super().__init__()
        self.defaults.update({'deff': 100., 'fs': 50., 'section': 'sonophore'})
        self.factors.update({'deff': 1e-9})
        self.addDeff()

    def addDeff(self):
        self.add_argument(
            '--deff', nargs='+', type=float, help='Effective intracellular depth (nm)')

    def parse(self):
        args = super().parse()
        args['deff'] = self.parse2array(args, 'deff', factor=self.factors['deff'])
        return args

    def parseSimInputs(self, args):
        return super().parseSimInputs(args) + [args[k] for k in ['fs', 'deff']]

    def parsePlot(self, args, output):
        if args['section'] == ['all']:
            args['section'] = ['sonophore', 'surroundings']
        super().parsePlot(args, output)


class SpatiallyExtendedTimeSeriesParser(TimeSeriesParser):

    def __init__(self):
        super().__init__()
        self.addSection()

    def addSection(self):
        SpatiallyExtendedParser.addSection(self)


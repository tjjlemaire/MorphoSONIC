# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-18 21:14:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-04 15:08:07

import matplotlib.pyplot as plt
from PySONIC.parsers import *

from .plt import SectionGroupedTimeSeries, SectionCompTimeSeries
from .core import models_dict


class SpatiallyExtendedParser(Parser):

    def __init__(self):
        super().__init__()
        self.addSection()

    def addResistivity(self):
        self.add_argument(
            '--rs', nargs='+', type=float, help='Intracellular resistivity (Ohm.cm)')

    def addSection(self):
        self.add_argument(
            '--section', nargs='+', type=str, help='Section of interest for plot')

    def addSectionID(self):
        self.add_argument(
            '--secid', nargs='+', type=str, help='Section ID')

    def parse(self, args=None):
        if args is None:
            args = super().parse()
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


class FiberParser(SpatiallyExtendedParser):

    def __init__(self):
        super().__init__()
        self.defaults.update({'type': 'senn', 'fiberD': 20., 'nnodes': 21})
        self.factors.update({'fiberD': 1e-6})
        self.addType()
        self.addFiberDiameter()
        self.addNnodes()

    def addResistivity(self):
        pass

    def addType(self):
        self.add_argument(
            '--type', nargs='+', type=str, help='Fiber model type')

    def addFiberDiameter(self):
        self.add_argument(
            '-d', '--fiberD', nargs='+', type=float, help='Fiber diameter (um)')

    def addNnodes(self):
        self.add_argument(
            '--nnodes', nargs='+', type=int, help='Number of nodes of Ranvier')

    def parsePlot(self, args, output):
        if args['section'] == ['all']:
            args['section'] = [f'node{i}' for i in range(args['nnodes'][0])]
        return SpatiallyExtendedParser.parsePlot(args, output)

    @staticmethod
    def parseSimInputs(args):
        return SpatiallyExtendedParser.parseSimInputs(args)

    def parse(self, args=None):
        args = super().parse(args=args)
        args['type'] = [models_dict[model_key] for model_key in args['type']]
        for key in ['fiberD']:
            if len(args[key]) > 1 or args[key][0] is not None:
                args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args


class EStimFiberParser(FiberParser, PWSimParser):

    def __init__(self):
        PWSimParser.__init__(self)
        FiberParser.__init__(self)
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
            help=f'Point-source current amplitude range {self.dist_str} ({self.amp_unit})')
        self.to_parse['amp'] = self.parseAmplitude

    def parseAmplitude(self, args):
        return EStimParser.parseAmplitude(self, args)

    def parse(self):
        args = FiberParser.parse(self, args=PWSimParser.parse(self))
        if isIterable(args['mode']):
            args['mode'] = args['mode'][0]
        return args

    @staticmethod
    def parseSimInputs(args):
        return PWSimParser.parseSimInputs(args) + SpatiallyExtendedParser.parseSimInputs(args)

    def parsePlot(self, *args):
        return FiberParser.parsePlot(self, *args)


class IextraFiberParser(EStimFiberParser):

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


class IintraFiberParser(EStimFiberParser):

    amp_unit = 'nA'

    def __init__(self):
        super().__init__()
        self.defaults.update({'secid': None, 'mode': 'anode', 'amp': 2.0})
        self.factors.update({'amp': 1e-9})
        self.addSectionID()


class AStimFiberParser(FiberParser, AStimParser):

    def __init__(self):
        AStimParser.__init__(self)
        FiberParser.__init__(self)
        for x in [self.defaults, self.allowed, self.to_parse]:
            x.pop('method')
        self.defaults.update({'tstim': 0.1, 'toffset': 3.})

    @staticmethod
    def parseSimInputs(args):
        return AStimParser.parseSimInputs(args) + SpatiallyExtendedParser.parseSimInputs(args)

    def parsePlot(self, *args):
        return FiberParser.parsePlot(self, *args)


class SectionAStimFiberParser(AStimFiberParser):

    amp_unit = 'kPa'

    def __init__(self):
        super().__init__()
        self.defaults.update({'sec_id': None})
        self.addSectionID()

    def parseAmplitude(self, args):
        return AStimParser.parseAmplitude(self, args)

    def parse(self):
        args = super().parse()
        args['secid'] = [args['secid']]
        return args


class SpatiallyExtendedTimeSeriesParser(TimeSeriesParser):

    def __init__(self):
        super().__init__()
        self.addSection()

    def addSection(self):
        SpatiallyExtendedParser.addSection(self)


class TestNodeNetworkParser(TestParser):

    def __init__(self, valid_subsets):
        super().__init__(valid_subsets)
        self.addConnect()

    def addConnect(self):
        self.add_argument(
            '--connect', default=False, action='store_true', help='Connect nodes')

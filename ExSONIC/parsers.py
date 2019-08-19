# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-18 21:14:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-19 07:07:36

import matplotlib.pyplot as plt
from PySONIC.parsers import *

from .plt import SectionGroupedTimeSeries, SectionCompTimeSeries


class ExtendedSonicAstimParser(AStimParser):

    def __init__(self):
        super().__init__()
        self.defaults.update({'rs': 1e2})
        self.factors.update({'rs': 1e0})
        self.addResistivity()
        self.addSection()

    def addResistivity(self):
        self.add_argument(
            '--rs', nargs='+', type=float, help='Intracellular resistivity (Ohm.cm)')

    def addSection(self):
        self.add_argument(
            '--section', nargs='+', type=str, help='Section of interest for plot')

    def parse(self):
        args = super().parse()
        del args['method']
        args['rs'] = self.parse2array(args, 'rs', factor=self.factors['rs'])
        return args

    @staticmethod
    def parseSimInputs(args):
        return [args['freq']] + PWSimParser.parseSimInputs(args) + [args['rs']]

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
                if key in args:
                    render_args[key] = args[key]
            for pltvar in args['plot']:
                comp_plot = SectionCompTimeSeries(output, pltvar, args['section'])
                comp_plot.render(**render_args)
        else:
            for key in args['section']:
                scheme_plot = SectionGroupedTimeSeries(key, output, pltscheme=args['pltscheme'])
                scheme_plot.render(**render_args)
        plt.show()


class ExtSonicNodeAStimParser(ExtendedSonicAstimParser):

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
        self.add_argument(
            '--section', nargs='+', type=str, help='Section of interest for plot')

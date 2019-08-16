# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-15 20:33:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-16 20:05:33

''' Run A-STIM simulations of an extended SONIC node with a specific point-neuron mechanism. '''

import matplotlib.pyplot as plt

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.parsers import EStimParser, PWSimParser, AStimParser

from ExSONIC.core import ExtendedSonicNode
from ExSONIC.plt import SectionGroupedTimeSeries


class ExtSonicNodeAStimParser(AStimParser):

    def __init__(self):
        super().__init__()
        self.defaults.update({'deff': 100., 'rs': 1e2, 'fs': 50., 'section': 'sonophore'})
        self.factors.update({'deff': 1e-9, 'rs': 1e0})
        self.addDeff()
        self.addResistivity()
        self.addSection()

    def addDeff(self):
        self.add_argument(
            '--deff', nargs='+', type=float, help='Effective intracellular depth (nm)')

    def addResistivity(self):
        self.add_argument(
            '--rs', nargs='+', type=float, help='Intracellular resistivity (Ohm.cm)')

    def addSection(self):
        self.add_argument(
            '--section', nargs='+', type=str, help='Section of interest for plot')

    def parse(self):
        args = super().parse()
        del args['method']
        for key in ['deff', 'rs']:
            args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args

    @staticmethod
    def parseSimInputs(args):
        return [args['freq']] + PWSimParser.parseSimInputs(args) + [args[k] for k in ['fs', 'deff', 'rs']]

    @staticmethod
    def parsePlot(args, output):
        render_args = {}
        if 'spikes' in args:
            render_args['spikes'] = args['spikes']
        if args['compare']:
            if args['plot'] == ['all']:
                logger.error('Specific variables must be specified for comparative plots')
                return
            for key in ['cmap', 'cscale']:
                if key in args:
                    render_args[key] = args[key]
            for pltvar in args['plot']:
                comp_plot = CompTimeSeries(output, pltvar)
                comp_plot.render(**render_args)
        else:
            if args['section'] == ['all']:
                args['section'] = list(output[0][0].keys())
            for key in args['section']:
                scheme_plot = SectionGroupedTimeSeries(key, output, pltscheme=args['pltscheme'])
                scheme_plot.render(**render_args)
        plt.show()


def main():
    # Parse command line arguments
    parser = ExtSonicNodeAStimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    if args['mpi']:
        logger.warning('NEURON multiprocessing disabled')

    # Run A-STIM batch
    logger.info("Starting Extended Sonic Node A-STIM simulation batch")
    queue = PointNeuron.simQueue(*EStimParser.parseSimInputs(args), outputdir=args['outputdir'])
    output = []
    for a in args['radius']:
        for pneuron in args['neuron']:
            for Fdrive in args['freq']:
                for fs in args['fs']:
                    for deff in args['deff']:
                        for rs in args['rs']:
                            extnode = ExtendedSonicNode(pneuron, rs, a=a, Fdrive=Fdrive, fs=fs, deff=deff)
                            batch = Batch(extnode.simAndSave if args['save'] else extnode.simulate, queue)
                            output += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)


if __name__ == '__main__':
    main()

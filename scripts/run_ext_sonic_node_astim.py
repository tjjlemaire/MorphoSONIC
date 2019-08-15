# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-15 20:33:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-15 21:27:40

''' Run A-STIM simulations of an extended SONIC node with a specific point-neuron mechanism. '''

import matplotlib.pyplot as plt

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.parsers import EStimParser, PWSimParser, AStimParser

from ExSONIC.core import ExtendedSonicNode
from ExSONIC.plt import getData, plotSignals


class ExtSonicNodeAStimParser(AStimParser):

    def __init__(self):
        super().__init__()
        self.defaults.update({'deff': 100., 'rs': 1e2, 'fs': 50.})
        self.factors.update({'deff': 1e-9, 'rs': 1e0})
        self.addDeff()
        self.addResistivity()

    def addDeff(self):
        self.add_argument(
            '--deff', nargs='+', type=float, help='Effective intracellular depth (nm)')

    def addResistivity(self):
        self.add_argument(
            '--rs', nargs='+', type=float, help='Intracellular resistivity (Ohm.cm)')

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
        for item in output:
            data, meta = getData(item)
            lbls = list(data.keys())
            t = data[lbls[0]]['t']
            for pltvar in args['plot']:
                signals = [df[pltvar].values for df in data.values()]
                plotSignals(t, signals, lbls=lbls)
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

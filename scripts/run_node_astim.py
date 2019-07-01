# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-02-13 18:16:09
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-01 17:22:43

''' Run A-STIM simulations of a specific point-neuron. '''

import matplotlib.pyplot as plt

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.plt import GroupedTimeSeries
from PySONIC.parsers import AStimParser, EStimParser
from ExSONIC.core import SonicNode


def main():
    # Parse command line arguments
    parser = AStimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    if args['mpi']:
        logger.warning('NEURON multiprocessing disabled')

    # Run A-STIM batch
    logger.info("Starting A-STIM simulation batch")
    queue = PointNeuron.simQueue(*EStimParser.parseSimInputs(args), outputdir=args['outputdir'])
    output = []
    for a in args['radius']:
        for pneuron in args['neuron']:
            for Fdrive in args['freq']:
                for fs in args['fs']:
                    node = SonicNode(pneuron, a=a, Fdrive=Fdrive, fs=fs)
                    batch = Batch(node.simAndSave if args['save'] else node.simulate, queue)
                    output += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)


if __name__ == '__main__':
    main()

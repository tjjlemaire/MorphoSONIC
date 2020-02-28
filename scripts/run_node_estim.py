# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-20 18:02:21

''' Run E-STIM simulations of a specific point-neuron. '''

import matplotlib.pyplot as plt

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.plt import GroupedTimeSeries
from PySONIC.parsers import EStimParser
from ExSONIC.core import IintraNode


def main():
    # Parse command line arguments
    parser = EStimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    if args['mpi']:
        logger.warning('NEURON multiprocessing disabled')

    # Run E-STIM batch
    logger.info("Starting E-STIM simulation batch")
    queue = PointNeuron.simQueue(*parser.parseSimInputs(args), outputdir=args['outputdir'])
    output = []
    for pneuron in args['neuron']:
        node = IintraNode(pneuron)
        batch = Batch(node.simAndSave if args['save'] else node.simulate, queue)
        output += batch(loglevel=args['loglevel'], mpi=args['mpi'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)

if __name__ == '__main__':
    main()

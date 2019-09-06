# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-15 20:33:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-06 16:12:42

''' Run simulations of an SENN fiber model with a specific point-neuron mechanism
    upon extracellular electrical stimulation. '''

import matplotlib.pyplot as plt

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.parsers import EStimParser
from ExSONIC.core import VextSennFiber, ExtracellularCurrent
from ExSONIC.parsers import IextSennParser


def main():
    # Parse command line arguments
    parser = IextSennParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    if args['mpi']:
        logger.warning('NEURON multiprocessing disabled')

    # Run batch
    logger.info('Starting SENN fiber Iext-STIM simulation batch')
    queue = PointNeuron.simQueue(*EStimParser.parseSimInputs(args), outputdir=args['outputdir'])
    output = []
    for pneuron in args['neuron']:
        for fiberD in args['fiberD']:
                for nnodes in args['nnodes']:
                    for rs in args['rs']:
                        fiber = VextSennFiber(pneuron, fiberD, nnodes, rs=rs)
                        for xps in args['xps']:
                            for zps in args['zps']:
                                if zps is None:
                                    zps = fiber.interL
                                psource = ExtracellularCurrent(xps, zps, args['mode'])
                                if args['save']:
                                    simqueue = [[item[0], psource, *item[1:]] for item in queue]
                                else:
                                    simqueue = [[psource, *item] for item in queue]
                                batch = Batch(fiber.simAndSave if args['save'] else fiber.simulate, simqueue)
                                output += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)


if __name__ == '__main__':
    main()

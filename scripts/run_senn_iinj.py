# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-09-06 16:12:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-27 13:38:19

''' Run simulations of an SENN fiber model with a specific point-neuron mechanism
    upon intracellular electrical stimulation. '''

import matplotlib.pyplot as plt

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.parsers import EStimParser
from ExSONIC.core import IinjSennFiber, IntracellularCurrent
from ExSONIC.parsers import IinjSennParser


def main():
    # Parse command line arguments
    parser = IinjSennParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    if args['mpi']:
        logger.warning('NEURON multiprocessing disabled')

    # Run batch
    logger.info('Starting SENN fiber Iinj-STIM simulation batch')
    queue = PointNeuron.simQueue(*EStimParser.parseSimInputs(args), outputdir=args['outputdir'])
    output = []
    for pneuron in args['neuron']:
        for fiberD in args['fiberD']:
                for nnodes in args['nnodes']:
                    for rs in args['rs']:
                        for nodeL in args['nodeL']:
                            for d_ratio in args['d_ratio']:
                                fiber = IinjSennFiber(
                                    pneuron, fiberD, nnodes, rs=rs, nodeL=nodeL, d_ratio=d_ratio)
                                for inode in args['inode']:
                                    if inode is None:
                                        inode = nnodes // 2
                                    psource = IntracellularCurrent(inode, args['mode'])
                                    if args['save']:
                                        simqueue = [[item[0], psource, *item[1:]] for item in queue]
                                    else:
                                        simqueue = [[psource, *item] for item in queue]
                                    batch = Batch(
                                        fiber.simAndSave if args['save'] else fiber.simulate, simqueue)
                                    output += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)


if __name__ == '__main__':
    main()

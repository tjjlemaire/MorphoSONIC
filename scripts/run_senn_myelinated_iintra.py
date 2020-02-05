# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-09-06 16:12:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-05 19:01:09

''' Run simulations of an SENN fiber model with a specific point-neuron mechanism
    upon intracellular electrical stimulation. '''

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.parsers import EStimParser
from ExSONIC.core import IintraFiber, myelinatedFiber, IntracellularCurrent
from ExSONIC.parsers import IintraMyelinatedFiberParser


def main():
    # Parse command line arguments
    parser = IintraMyelinatedFiberParser()
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
                                fiber = myelinatedFiber(IintraFiber, pneuron, fiberD, nnodes,
                                    rs=rs, nodeL=nodeL, d_ratio=d_ratio)
                                for inode in args['inode']:
                                    if inode is None:
                                        inode = nnodes // 2
                                    psource = IntracellularCurrent(inode, mode=args['mode'])
                                    if args['save']:
                                        simqueue = [(
                                            [psource.updatedX(item[0][0].I), *item[0][1:]],
                                            item[1]
                                        ) for item in queue]
                                        func = fiber.simAndSave
                                    else:
                                        simqueue = [
                                            [psource.updatedX(item[0].I), *item[1:]]
                                            for item in queue]
                                        func = fiber.simulate
                                    batch = Batch(func, simqueue)
                                    output += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)


if __name__ == '__main__':
    main()

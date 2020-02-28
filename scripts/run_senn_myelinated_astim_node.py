# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-09-27 14:28:52
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-19 21:36:43

''' Run simulations of an SENN SONIC fiber model with a specific point-neuron mechanism
    upon ultrasound stimulation at one onde. '''

from PySONIC.core import Batch, NeuronalBilayerSonophore
from PySONIC.utils import logger
from PySONIC.parsers import AStimParser
from ExSONIC.core import SonicFiber, myelinatedFiber, NodeAcousticSource
from ExSONIC.parsers import NodeAStimMyelinatedFiberParser


def main():
    # Parse command line arguments
    parser = NodeAStimMyelinatedFiberParser()
    args = parser.parse()
    args['method'] = [None]
    logger.setLevel(args['loglevel'])
    if args['mpi']:
        logger.warning('NEURON multiprocessing disabled')

    # Run batch
    logger.info('Starting SENN fiber Iext-STIM simulation batch')
    queue = [item[:2] for item in NeuronalBilayerSonophore.simQueue(
        *AStimParser.parseSimInputs(args), outputdir=args['outputdir'])]
    if args['save']:
        queue = [(item[0][:2], item[1]) for item in queue]
    output = []
    for pneuron in args['neuron']:
        for fiberD in args['fiberD']:
                for nnodes in args['nnodes']:
                    for rs in args['rs']:
                        for nodeL in args['nodeL']:
                            for d_ratio in args['d_ratio']:
                                for a in args['radius']:
                                    for fs in args['fs']:
                                        fiber = myelinatedFiber(SonicFiber, pneuron, fiberD, nnodes,
                                            rs=rs, nodeL=nodeL, d_ratio=d_ratio, a=a, fs=fs)
                                        for inode in args['inode']:
                                            if inode is None:
                                                inode = nnodes // 2
                                            # psource = NodeAcousticSource(inode, Fdrive)
                                            if args['save']:
                                                simqueue = [(
                                                    [NodeAcousticSource(inode, item[0][0].f, item[0][0].A), *item[0][1:]],
                                                    item[1]
                                                ) for item in queue]
                                                func = fiber.simAndSave
                                            else:
                                                simqueue = [
                                                    [NodeAcousticSource(inode, item[0].f, item[0].A), *item[1:]]
                                                    for item in queue]
                                                # simqueue = [[psource, *item] for item in queue]
                                                func = fiber.simulate
                                            batch = Batch(func, simqueue)
                                            output += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)


if __name__ == '__main__':
    main()

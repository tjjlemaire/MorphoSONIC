# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-09-27 14:28:52
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-27 15:12:28

''' Run simulations of an SENN SONIC fiber model with a specific point-neuron mechanism
    upon ultrasound stimulation at one onde. '''

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.parsers import EStimParser
from ExSONIC.core import SonicSennFiber, NodeAcousticSource
from ExSONIC.parsers import NodeAstimSennParser


def main():
    # Parse command line arguments
    parser = NodeAstimSennParser()
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
                        for nodeL in args['nodeL']:
                            for d_ratio in args['d_ratio']:
                                for a in args['radius']:
                                    for Fdrive in args['freq']:
                                        for fs in args['fs']:
                                            fiber = SonicSennFiber(
                                                pneuron, fiberD, nnodes, a=a, Fdrive=Fdrive, rs=rs,
                                                nodeL=nodeL, d_ratio=d_ratio)
                                            for inode in args['inode']:
                                                if inode is None:
                                                    inode = nnodes // 2
                                                psource = NodeAcousticSource(inode, Fdrive)
                                                if args['save']:
                                                    simqueue = [[item[0], psource, *item[1:]] for item in queue]
                                                    method = fiber.simAndSave
                                                else:
                                                    simqueue = [[psource, *item] for item in queue]
                                                    method = fiber.simulate
                                                batch = Batch(method, simqueue)
                                                output += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)


if __name__ == '__main__':
    main()

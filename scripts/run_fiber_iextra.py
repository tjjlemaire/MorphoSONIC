# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-15 20:33:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-02 17:54:59

''' Run simulations of an SENN fiber model with a specific point-neuron mechanism
    upon extracellular electrical stimulation. '''

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.parsers import EStimParser
from ExSONIC.core import ExtracellularCurrent
from ExSONIC.parsers import IextraFiberParser


def main():
    # Parse command line arguments
    parser = IextraFiberParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    if args['mpi']:
        logger.warning('NEURON multiprocessing disabled')

    # Run batch
    logger.info('Starting SENN fiber Iext-STIM simulation batch')
    queue = PointNeuron.simQueue(*EStimParser.parseSimInputs(args), outputdir=args['outputdir'])
    output = []
    for fiber_class in args['type']:
        for fiberD in args['fiberD']:
            for nnodes in args['nnodes']:
                fiber = fiber_class(fiberD, nnodes)
                for xps in args['xps']:
                    for zps in args['zps']:
                        if zps is None:
                            zps = fiber.interL
                        source = ExtracellularCurrent((xps, zps), mode=args['mode'])
                        if args['save']:
                            simqueue = [(
                                [source.updatedX(item[0][0].I), *item[0][1:]],
                                item[1]
                            ) for item in queue]
                            func = fiber.simAndSave
                        else:
                            simqueue = [
                                [source.updatedX(item[0].I), *item[1:]]
                                for item in queue]
                            func = fiber.simulate
                        batch = Batch(func, simqueue)
                        output += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)


if __name__ == '__main__':
    main()

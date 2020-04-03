# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-09-06 16:12:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-03 19:44:25

''' Run simulations of an SENN fiber model with a specific point-neuron mechanism
    upon intracellular electrical stimulation. '''

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.parsers import EStimParser
from ExSONIC.core import IntracellularCurrent
from ExSONIC.parsers import IintraFiberParser


def main():
    # Parse command line arguments
    parser = IintraFiberParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    if args['mpi']:
        logger.warning('NEURON multiprocessing disabled')

    # Run batch
    logger.info('Starting fiber Iinj-STIM simulation batch')
    queue = PointNeuron.simQueue(*EStimParser.parseSimInputs(args), outputdir=args['outputdir'])
    output = []
    for fiber_class in args['type']:
        for fiberD in args['fiberD']:
            for nnodes in args['nnodes']:
                fiber = fiber_class(fiberD, nnodes)
                for sec_id in args['secid']:
                    if sec_id is None:
                        sec_id = fiber.central_ID
                    psource = IntracellularCurrent(sec_id, mode=args['mode'])
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

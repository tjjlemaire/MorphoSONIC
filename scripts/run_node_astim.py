# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-02-13 18:16:09
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:46:49

''' Run A-STIM simulations of a specific point-neuron. '''

from PySONIC.core import Batch, NeuronalBilayerSonophore
from PySONIC.utils import logger
from PySONIC.parsers import AStimParser
from MorphoSONIC.models import Node


def main():
    # Parse command line arguments
    parser = AStimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    if args['mpi']:
        logger.warning('NEURON multiprocessing disabled')
    sim_inputs = parser.parseSimInputs(args)
    simQueue_func = {9: 'simQueue', 10: 'simQueueBurst'}[len(sim_inputs)]

    # Run A-STIM batch
    logger.info("Starting node A-STIM simulation batch")
    queue = getattr(NeuronalBilayerSonophore, simQueue_func)(
        *sim_inputs, outputdir=args['outputdir'], overwrite=args['overwrite'])
    queue = [item[:2] for item in queue]
    if args['save']:
        queue = [(x[0][:-3], x[1]) for x in queue]  # adapt for NEURON case
    output = []
    for a in args['radius']:
        for pneuron in args['neuron']:
            for fs in args['fs']:
                node = Node(pneuron, a=a, fs=fs)
                batch = Batch(node.simAndSave if args['save'] else node.simulate, queue)
                output += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)


if __name__ == '__main__':
    main()

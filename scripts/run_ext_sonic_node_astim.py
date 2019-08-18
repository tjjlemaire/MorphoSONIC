# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-15 20:33:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-18 21:23:27

''' Run A-STIM simulations of an extended SONIC node with a specific point-neuron mechanism. '''

import matplotlib.pyplot as plt

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.parsers import EStimParser

from ExSONIC.core import ExtendedSonicNode
from ExSONIC.parsers import ExtSonicNodeAStimParser


def main():
    # Parse command line arguments
    parser = ExtSonicNodeAStimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    if args['mpi']:
        logger.warning('NEURON multiprocessing disabled')

    # Run A-STIM batch
    logger.info("Starting Extended Sonic Node A-STIM simulation batch")
    queue = PointNeuron.simQueue(*EStimParser.parseSimInputs(args), outputdir=args['outputdir'])
    output = []
    for a in args['radius']:
        for pneuron in args['neuron']:
            for Fdrive in args['freq']:
                for fs in args['fs']:
                    for deff in args['deff']:
                        for rs in args['rs']:
                            extnode = ExtendedSonicNode(pneuron, rs, a=a, Fdrive=Fdrive, fs=fs, deff=deff)
                            batch = Batch(extnode.simAndSave if args['save'] else extnode.simulate, queue)
                            output += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)


if __name__ == '__main__':
    main()

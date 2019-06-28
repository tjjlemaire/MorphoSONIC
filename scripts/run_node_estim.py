# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-28 17:35:12

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
    pkl_filepaths = []
    inputs = [args[k] for k in ['amp', 'tstim', 'toffset', 'PRF', 'DC']]
    queue = PointNeuron.simQueue(*inputs, outputdir=args['outputdir'])
    for pneuron in args['neuron']:
        node = IintraNode(pneuron)
        batch = Batch(node.runAndSave, queue)
        pkl_filepaths += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        scheme_plot = GroupedTimeSeries(pkl_filepaths, pltscheme=args['pltscheme'])
        scheme_plot.render(spikes=args['spikes'])
        plt.show()


if __name__ == '__main__':
    main()
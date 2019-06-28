# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-02-13 18:16:09
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-28 19:57:00

''' Run A-STIM simulations of a specific point-neuron. '''

import matplotlib.pyplot as plt

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.plt import GroupedTimeSeries
from PySONIC.parsers import AStimParser
from ExSONIC.core import SonicNode


def main():
    # Parse command line arguments
    parser = AStimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    if args['mpi']:
        logger.warning('NEURON multiprocessing disabled')

    # Run A-STIM batch
    logger.info("Starting A-STIM simulation batch")
    pkl_filepaths = []
    inputs = [args[k] for k in ['amp', 'tstim', 'toffset', 'PRF', 'DC']]
    queue = PointNeuron.simQueue(*inputs, outputdir=args['outputdir'])
    for a in args['radius']:
        for pneuron in args['neuron']:
            for Fdrive in args['freq']:
                for fs in args['fs']:
                    node = SonicNode(pneuron, a=a, Fdrive=Fdrive, fs=fs)
                    batch = Batch(node.runAndSave, queue)
                    pkl_filepaths += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        scheme_plot = GroupedTimeSeries(pkl_filepaths, pltscheme=args['pltscheme'])
        scheme_plot.render(spikes=args['spikes'])
        plt.show()


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-02-10 09:49:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-10 11:11:02

import os
import logging
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger, si_format
from ExSONIC.core import SonicFiber, myelinatedFiberReilly, unmyelinatedFiberSundt, PlanarDiskTransducerSource
from ExSONIC.plt import strengthDurationCurve

logger.setLevel(logging.INFO)


def computeSDcurve(source, fiber, durations, toffset, root='.'):
    ''' Compute strength-duration curve for a fiber-source combination.

        :param source: acoustic source object
        :param fiber: fiber model
        :param durations: array of pulse durations
        :param toffset: constant stimulus offset
        :param root: root for IO operations
        :return: array of threshold acoustic amplitudes for each pulse duration
    '''
    logger.info(f'Computing SD curve for {fiber}')

    # Get filename from parameters
    d_ratio = 1. if fiber.interL == 0 else 0.7
    base = 'myelinated'
    if fiber.interL == 0:
        base = 'un' + base
    fiber_str = f'{base}_{fiber.nodeD / d_ratio * 1e6:.2f}um'
    tstr = f'{si_format(durations.min(), 2, space="")}s_{si_format(durations.max(), 2, space="")}s_{durations.size}'
    fcode = f'SDcurve_{fiber_str}_{tstr}'
    fname = f'{fcode}.csv'
    fpath = os.path.join(root, fname)
    delimiter = '\t'
    labels = ['t stim (s)', 'Athr (Pa)']

    # Create log file if it does not exist
    if not os.path.isfile(fpath):
        logger.debug(f'creating log file: "{fpath}"')
        with open(fpath, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            writer.writerow(labels)
    else:
        logger.debug(f'existing log file: "{fpath}"')

    # For each duration
    for t in durations:
        # If entry not found in log file
        df = pd.read_csv(fpath, sep=delimiter)
        entries = df[labels[0]].values.astype(float)
#        entries = pd.read_csv(fpath, sep=delimiter)['t stim (s)'].values
        imatches = np.where(np.isclose(entries, t, rtol=1e-9, atol=1e-16))[0]
        if len(imatches) == 0:
            # Run titration and log output in file
            logger.debug(f'entry not found: "{t}"')
            pp = PulsedProtocol(t, toffset)
            print(fiber)
            print(source)
            print(pp)
            uthr = fiber.titrate(source, pp)                 # m/s
            Athr = uthr * source.relNormalAxisAmp(source.z)  # Pa
            with open(fpath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter)
                writer.writerow([t, Athr])
        else:
            logger.debug(f'existing entry: "{t}"')

    # Return log file outputs along with identifier
    Athrs = pd.read_csv(fpath, sep=delimiter)['Athr (Pa)'].values  # Pa
    return fiber_str, Athrs


def plotSDcurves(root='.'):
    ''' Compute and plot strength-duration curves of myelinated and unmyelinated fibers
        of various characteristic diameters with a single acoustic source.
    '''
    # US source
    diam = 19e-3  # m
    f = 500e3     # Hz
    source = PlanarDiskTransducerSource((0., 0., 'focus'), f, r=diam/2)
    logger.info(f'US source: {source}')

    # Durations and offset
    durations = np.logspace(-5, 0, 10)  # s
    toffset = 10e-3                      # s

    # Diameters
    u_diams = [0.8e-6]  # m
    m_diams = [10e-6]      # m

    # Fiber parameters
    a = 32e-9  # m
    fs = 1.0   # (-)

    fiber = None
    outputs = []

    # Myelinated fibers
    for d in m_diams:
        if fiber is not None:
            fiber.clear()
        fiber = myelinatedFiberReilly(SonicFiber, a=a, fs=fs, fiberD=d)
        outputs.append(computeSDcurve(source, fiber, durations, toffset, root=root))

    # Unmyelinated fibers
    for d in u_diams:
        if fiber is not None:
            fiber.clear()
        fiber = unmyelinatedFiberSundt(SonicFiber, a=a, fs=fs, fiberD=d)
        outputs.append(computeSDcurve(source, fiber, durations, toffset, root=root))

    # Plot strength-duration curve
    colors = plt.get_cmap('tab20c').colors
    colors = colors[:3][::-1] + colors[4:7][::-1]
    Athrs = {out[0]: out[1] for out in outputs}

    fig = strengthDurationCurve(
        f'Comparative SD curves - {diam * 1e3:.0f} mm diameter planar transducer @ {si_format(f)}Hz',
        durations, Athrs, scale='log', colors=colors,
        yname='amplitude', yfactor=1e-3, yunit='Pa', plot_chr=False)

    # Return figure
    return fig


if __name__ == '__main__':
    root = 'data'
    SD_fig = plotSDcurves(root=root)
    # SD_fig.savefig(os.path.join(root, 'SD_curves.pdf'))
    plt.show()

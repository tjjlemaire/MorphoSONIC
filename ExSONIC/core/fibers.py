# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-11-27 18:03:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-05 16:29:46

''' Constructor functions for different types of fibers. '''

import os
import numpy as np
import pandas as pd
import csv

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format
from PySONIC.postpro import boundDataFrame
from PySONIC.core import PulsedProtocol

from .senn import IintraFiber, IextraFiber, SonicFiber
from .sources import *


def myelinatedFiber(fiber_class, pneuron, fiberD, nnodes, rs, nodeL, d_ratio, inter_ratio=100., **kwargs):
    ''' Create a single-cable myelinated fiber model.

        :param fiber_class: class of fiber to be instanced
        :param pneuron: point-neuron model object
        :param fiberD: fiber outer diameter (m)
        :param nnodes: number of nodes
        :param rs: axoplasmic resistivity (Ohm.cm)
        :param nodeL: nominal node length (m)
        :param d_ratio: ratio of axon (inner-myelin) and fiber (outer-myelin) diameters
        :param inter_ratio: ratio of internode length to fiber diameter (default = 100)
        :param kwargs: other keyword arguments
        :return: myelinated fiber object
    '''
    if fiberD <= 0:
        raise ValueError('fiber diameter must be positive')
    if nodeL <= 0.:
        raise ValueError('node length must be positive')
    if d_ratio > 1. or d_ratio < 0.:
        raise ValueError('fiber-axon diameter ratio must be in [0, 1]')
    if inter_ratio <= 0.:
        raise ValueError('fiber diameter - internode length ratio must be positive')

    # Define fiber geometrical parameters
    nodeD = d_ratio * fiberD       # m
    nodeL = nodeL                  # m
    interD = d_ratio * fiberD      # m
    interL = inter_ratio * fiberD  # m

    # Create fiber model instance
    return fiber_class(pneuron, nnodes, rs, nodeD, nodeL, interD, interL, **kwargs)


def unmyelinatedFiber(fiber_class, pneuron, fiberD, rs, fiberL=5e-3, maxNodeL=None, **kwargs):
    ''' Create a single-cable unmyelinated fiber model.

        :param fiber_class: class of fiber to be instanced
        :param pneuron: point-neuron model object
        :param fiberD: fiber outer diameter (m)
        :param rs: axoplasmic resistivity (Ohm.cm)
        :param fiberL: fiber length (m)
        :param maxNodeL: maximal node length (m)
        :param kwargs: other keyword arguments
        :return: unmyelinated fiber object
    '''
    if fiberD <= 0:
        raise ValueError('fiber diameter must be positive')
    if fiberL <= 0:
        raise ValueError('fiber length must be positive')
    if fiberL <= 3e-3:
        logger.warning('fiber length is below the convergence threshold')
    if maxNodeL is None:
        nodelength_lin = fiberD * 16.3 + 9.1e-6  #um
        maxNodeL = min(nodelength_lin, 22e-6)
    if maxNodeL <= 0. or maxNodeL > fiberL:
        raise ValueError('maximum node length must be in [0, fiberL]')

    # Compute number of nodes (ensuring odd number) and node length from fiber length
    nnodes = int(np.ceil(fiberL / maxNodeL))
    if nnodes % 2 == 0:
        nnodes += 1
    nodeL = fiberL / nnodes

    # Define other fiber geometrical parameters (with zero-length internode)
    nodeD = fiberD   # m
    interD = fiberD  # m
    interL = 0.      # m

    # Create fiber model instance
    fiber = fiber_class(pneuron, nnodes, rs, nodeD, nodeL, interD, interL, **kwargs)
    return fiber

def unmyelinatedFiberConvergence(pneuron, fiberD, rs, fiberL, maxNodeL_range, pp, outdir='.'):
    ''' Simulate an unmyelinated fiber model the model upon intracellular current injection
        at the central node, for increasing spatial resolution (i.e. decreasing node length),
        and quantify the model convergence via 3 output metrics:
        - stimulation threshold
        - conduction velocity
        - spike amplitude

        :param pneuron: C-fiber membrane equations
        :param fiberD: fiber diameter (m)
        :param rs: intracellular resistivity (ohm.cm)
        :param fiberL: fiber length (m)
        :param maxNodeL_range: maximum node length range (m)
        :param pp: pulsed protocol object
        :param outdir: output directory
        :return: 5 columns dataframe with the following vectors:
         - 'maxNodeL (m)'
         - 'nodeL (m)'
         - 'Ithr (A)'
         - 'CV (m/s)'
         - 'dV (mV)'
    '''

    # Get filecode
    fiber = unmyelinatedFiber(IintraFiber, pneuron, fiberD, rs, fiberL, maxNodeL=fiberL / 3)
    filecodes = fiber.filecodes(IntracellularCurrent(fiber.nnodes // 2, 0.0), pp)
    for k in ['nnodes', 'nodeL', 'interD', 'interL', 'inode', 'I', 'nature', 'toffset', 'PRF', 'DC']:
        del filecodes[k]
    filecodes['tstim'] = si_format(pp.tstim, 1, space='') + 's'
    filecodes['nodeL_range'] = 'nodeL' + '-'.join(
        [f'{si_format(x, 1, "")}m' for x in [min(maxNodeL_range), max(maxNodeL_range)]])
    fcode = '_'.join(filecodes.values())

    # Output file and column names
    fname = f'{fcode}_convergence_results.csv'
    fpath = os.path.join(outdir, fname)
    delimiter = '\t'
    labels = ['maxNodeL (m)', 'nodeL (m)', 'Ithr (A)', 'CV (m/s)', 'dV (mV)']

    # Create log file if it does not exist
    if not os.path.isfile(fpath):
        logger.info(f'creating log file: "{fpath}"')
        with open(fpath, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            writer.writerow(labels)

    # Loop through parameter sweep
    logger.info('running maxNodeL parameter sweep ({}m - {}m)'.format(
        *si_format([maxNodeL_range.min(), maxNodeL_range.max()], 2)))
    for x in maxNodeL_range[::-1]:

        # If max node length not already in the log file
        df = pd.read_csv(fpath, sep=delimiter)
        entries = df[labels[0]].values.astype(float)
        is_entry = np.any(np.isclose(x, entries))
        if not is_entry:

            # Initialize fiber with specific max node length
            logger.info(f'creating model with maxNodeL = {x * 1e6:.2f} um ...')
            fiber = unmyelinatedFiber(IintraFiber, pneuron, fiberD, rs, fiberL, maxNodeL=x)
            logger.info(f'resulting node length: {fiber.nodeL * 1e6:.2f} um')

            # Perform titration to find threshold current
            psource = IntracellularCurrent(fiber.nnodes // 2)
            logger.info(f'running titration with intracellular current injected at node {psource.inode}')
            Ithr = fiber.titrate(psource, pp)  # A

            # If fiber is excited
            if not np.isnan(Ithr):
                logger.info(f'Ithr = {si_format(Ithr, 2)}A')

                # Simulate fiber at 1.1 times threshold current
                data, meta = fiber.simulate(psource.updatedX(1.1 * Ithr), pp)

                # Filter out stimulation artefact from dataframe
                data = {k: boundDataFrame(df, (pp.tstim, pp.tstim + pp.toffset)) for k, df in data.items()}

                # Compute CV and spike amplitude
                # ids = fiber.ids.copy()
                # del ids[fiber.nnodes // 2]
                cv = fiber.getConductionVelocity(data, out='median')  # m/s
                dV = fiber.getSpikeAmp(data, out='median')            # mV
                logger.info(f'CV = {cv:.2f} m/s')
                logger.info(f'dV = {dV:.2f} mV')
            else:
                # Otherwise, assign NaN values to them
                cv, dV = np.nan, np.nan

            # Log input-output pair into file
            logger.info('saving result to log file')
            with open(fpath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter)
                writer.writerow([x, fiber.nodeL, Ithr, cv, dV])

            # Clear fiber sections
            fiber.clear()

    logger.info('parameter sweep successfully completed')

    # Load results
    logger.info('loading results from log file')
    df = pd.read_csv(fpath, sep=delimiter)

    return df

def unmyelinatedFiberConvergence_fiberL(pneuron, fiberD, rs, fiberL_range, pp, outdir='.'):

    # Get filecode
    fiber = unmyelinatedFiber(IintraFiber, pneuron, fiberD, rs, 5e-3)
    filecodes = fiber.filecodes(IntracellularCurrent(fiber.nnodes // 2, 0.0), pp)
    for k in ['nnodes', 'nodeL', 'interD', 'interL', 'inode', 'I', 'nature', 'toffset', 'PRF', 'DC']:
        del filecodes[k]
    filecodes['tstim'] = si_format(pp.tstim, 1, space='') + 's'
    filecodes['fiberL_range'] = 'fiberL' + '-'.join(
        [f'{si_format(x, 1, "")}m' for x in [min(fiberL_range), max(fiberL_range)]])
    fcode = '_'.join(filecodes.values())
    fiber.clear()

    # Output file and column names
    fname = f'{fcode}_convergence_results.csv'
    fpath = os.path.join(outdir, fname)
    delimiter = '\t'
    labels = ['fiberL (m)', 'nodeL (m)', 'Ithr (A)', 'CV (m/s)', 'dV (mV)']

    # Create log file if it does not exist
    if not os.path.isfile(fpath):
        logger.info(f'creating log file: "{fpath}"')
        with open(fpath, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            writer.writerow(labels)

    # Loop through parameter sweep
    logger.info('running fiberL parameter sweep ({}m - {}m)'.format(
        *si_format([fiberL_range.min(), fiberL_range.max()], 2)))
    for x in fiberL_range:

        # If fiber length not already in the log file
        df = pd.read_csv(fpath, sep=delimiter)
        entries = df[labels[0]].values.astype(float)
        is_entry = np.any(np.isclose(x, entries))
        if not is_entry:

            # Initialize fiber with specific fiber length
            logger.info(f'creating model with fiberL = {x * 1e6:.2f} um ...')
            fiber = unmyelinatedFiber(IintraFiber, pneuron, fiberD, rs, fiberL=x)
            logger.info(f'resulting node length: {fiber.nodeL * 1e6:.2f} um')

            # Perform titration to find threshold current
            psource = IntracellularCurrent(fiber.nnodes // 2)
            logger.info(f'running titration with intracellular current injected at node {psource.inode}')
            Ithr = fiber.titrate(psource, pp)  # A

            # If fiber is excited
            if not np.isnan(Ithr):
                logger.info(f'Ithr = {si_format(Ithr, 2)}A')

                # Simulate fiber at 1.1 times threshold current
                data, meta = fiber.simulate(psource.updatedX(1.1 * Ithr), pp)

                # Filter out stimulation artefact from dataframe
#                data = {k: boundDataFrame(df, (pp.tstim, pp.tstim + pp.toffset)) for k, df in data.items()}

                # Compute CV and spike amplitude
                cv = fiber.getConductionVelocity(data, out='median')  # m/s
                dV = fiber.getSpikeAmp(data, out='median')            # mV
                logger.info(f'CV = {cv:.2f} m/s')
                logger.info(f'dV = {dV:.2f} mV')
            else:
                # Otherwise, assign NaN values to them
                cv, dV = np.nan, np.nan

            # Log input-output pair into file
            logger.info('saving result to log file')
            with open(fpath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter)
                writer.writerow([x, fiber.nodeL, Ithr, cv, dV])

            # Clear fiber sections
            fiber.clear()

    logger.info('parameter sweep successfully completed')

    # Load results
    logger.info('loading results from log file')
    df = pd.read_csv(fpath, sep=delimiter)

    return df

def myelinatedFiberReilly(fiber_class, fiberD=20e-6, **kwargs):
    '''  Create typical myelinated fiber model, using parameters from Reilly 1985.

        :param fiber_class: class of fiber to be instanced
        :param fiberD: fiber diameter (m)
        :param kwargs: other keyword arguments
        :return: fiber myelinated object

        Reference: Reilly, J.P., Freeman, V.T., and Larkin, W.D. (1985). Sensory effects
        of transient electrical stimulation--evaluation with a neuroelectric model.
        IEEE Trans Biomed Eng 32, 1001–1011.
    '''
    pneuron = getPointNeuron('FH')  # Frog myelinated node membrane equations
    nnodes = 21                     # number of nodes
    rs = 110.0                      # axoplasm resistivity (Ohm.cm, from McNeal 1976)
    nodeL = 2.5e-6                  # node length (m, from McNeal 1976)
    d_ratio = 0.7                   # axon / fiber diameter ratio (from McNeal 1976)
    return myelinatedFiber(fiber_class, pneuron, fiberD, nnodes, rs, nodeL, d_ratio, **kwargs)

def myelinatedFiberSweeney(fiber_class, fiberD=10e-6, **kwargs):
    '''  Create typical myelinated fiber model, using parameters from Sweeney 1987.

        :param fiber_class: class of fiber to be instanced
        :param fiberD: fiber diameter (m)
        :param kwargs: other keyword arguments
        :return: fiber myelinated object

        Reference: Sweeney, J.D., Mortimer, J.T., and Durand, D. (1987). Modeling of
        mammalian myelinated nerve for functional neuromuscular stimulation. IEEE 9th
        Annual Conference of the Engineering in Medicine and Biology Society 3, 1577–1578.
    '''
    pneuron = getPointNeuron('SW')  # mammalian fiber membrane equations
    nnodes = 19                     # number of nodes
    rs = 54.7                       # axoplasm resistivity (Ohm.cm)
    nodeL = 1.5e-6                  # node length (m)
    d_ratio = 0.6                   # axon / fiber diameter ratio
    return myelinatedFiber(fiber_class, pneuron, fiberD, nnodes, rs, nodeL, d_ratio, **kwargs)

def unmyelinatedFiberSundt(fiber_class, fiberD=0.8e-6, fiberL = 5e-3, **kwargs):
    ''' Create typical unmyelinated fiber model, using parameters from Sundt 2015.

        :param fiber_class: class of fiber to be instanced
        :param fiberD: fiber diameter (m)
        :param kwargs: other keyword arguments
        :return: fiber myelinated object

        Reference: Sundt D., Gamper N., Jaffe D. B., Spike propagation through the dorsal
        root ganglia in an unmyelinated sensory neuron: a modeling study.
        Journal of Neurophysiology (2015)
    '''
    pneuron = getPointNeuron('sundt')  # DRG peripheral axon membrane equations
    rs = 100.                          # axoplasm resistivity, from Sundt 2015 (Ohm.cm)
    return unmyelinatedFiber(fiber_class, pneuron, fiberD, rs, fiberL, **kwargs)


def strengthDuration(fiberType, fiberClass, fiberD, tstim_range, toffset=20e-3, outdir='.', zdistance=1e-3, Fdrive=500e3, a=32e-9, fs=1., r=2e-3):

    logger.info(f'creating model with fiberD = {fiberD * 1e6:.2f} um ...')
    if fiberClass == 'intracellular_electrical_stim':
        fiber_class = IintraFiber
        if fiberType =='sundt':
            fiber = unmyelinatedFiberSundt(fiber_class, fiberD)
        elif fiberType =='reilly':
            fiber = myelinatedFiberReilly(fiber_class, fiberD)
        else:
            raise ValueError('fiber type unknown')
        psource = IntracellularCurrent(fiber.nnodes // 2)
    elif fiberClass == 'extracellular_electrical_stim':
        fiber_class = IextraFiber
        if fiberType =='sundt':
            fiber = unmyelinatedFiberSundt(fiber_class, fiberD)
        elif fiberType =='reilly':
            fiber = myelinatedFiberReilly(fiber_class, fiberD)
        else:
            raise ValueError('fiber type unknown')
        psource = ExtracellularCurrent((0, zdistance), mode='cathode')
    elif fiberClass == 'acoustic_single_node':
        fiber_class = SonicFiber
        if fiberType =='sundt':
            fiber = unmyelinatedFiberSundt(fiber_class, fiberD, a=a)
        elif fiberType =='reilly':
            fiber = myelinatedFiberReilly(fiber_class, fiberD, a=a)
        else:
            raise ValueError('fiber type unknown')
        psource = NodeAcousticSource(fiber.nnodes//2, f)
    elif fiberClass == 'acoustic_planar_transducer':
        fiber_class = SonicFiber
        if fiberType =='sundt':
            fiber = unmyelinatedFiberSundt(fiber_class, fiberD, a=a, fs=fs)
        elif fiberType =='reilly':
            fiber = myelinatedFiberReilly(fiber_class, fiberD, a=a, fs=fs)
        else:
            raise ValueError('fiber type unknown')
        psource = PlanarDiskTransducerSource(0, 0, zdistance, Fdrive, r=r)
    else:
        raise ValueError('fiber class unknown')

    # Get filecode
    filecodes = fiber.filecodes(psource.updatedX(1), PulsedProtocol(toffset/100, toffset))
    for k in ['nnodes', 'nodeD', 'rs', 'nodeL', 'interD', 'interL', 'I', 'inode', 'nature', 'tstim', 'toffset', 'PRF', 'DC']:
        if k in filecodes:
            del filecodes[k]
    filecodes['fiberD'] = f'fiberD{(fiberD * 1e6):.2f}um'
    if fiberClass == 'extracellular_electrical_stim' or fiberClass == 'acoustic_planar_transducer':
        filecodes['zsource'] = f'zsource{(psource.z * 1e3):.2f}mm'
    if fiberClass == 'acoustic_single_node' or fiberClass == 'acoustic_planar_transducer':
        del filecodes['f']
    filecodes['tstim_range'] = 'tstim' + '-'.join(
        [f'{si_format(x, 1, "")}s' for x in [min(tstim_range), max(tstim_range)]])
    print(filecodes)
    fcode = '_'.join(filecodes.values())

    # Output file and column names
    fname = f'{fcode}_strengthduration_results.csv'
    fpath = os.path.join(outdir, fname)
    delimiter = '\t'
    labels = ['t stim (m)', 'Ithr (A)']

    # Create log file if it does not exist
    if not os.path.isfile(fpath):
        logger.info(f'creating log file: "{fpath}"')
        with open(fpath, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            writer.writerow(labels)

    # Loop through parameter sweep
    logger.info('running tstim parameter sweep ({}s - {}s)'.format(
        *si_format([tstim_range.min(), tstim_range.max()], 2)))
    for x in tstim_range:

        # If fiber length not already in the log file
        df = pd.read_csv(fpath, sep=delimiter)
        entries = df[labels[0]].values.astype(float)
        is_entry = np.any(np.isclose(x, entries))
        if not is_entry:

            # Perform titration to find threshold current
            pp = PulsedProtocol(x, toffset)
            Ithr = fiber.titrate(psource, pp)  # A

            # Log input-output pair into file
            logger.info('saving result to log file')
            with open(fpath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter)
                writer.writerow([x, Ithr])

    logger.info('parameter sweep successfully completed')
    fiber.clear()

    # Load results
    logger.info('loading results from log file')
    df = pd.read_csv(fpath, sep=delimiter)

    return df


def currentDistance(fiberType, fiberD, tstim, n_cur, cur_min, cur_max, n_z, z_min, z_max, outdir='.'):

    fiber_class = IextraFiber
    if fiberType =='sundt':
        fiber = unmyelinatedFiberSundt(fiber_class, fiberD)
    elif fiberType =='reilly':
        fiber = myelinatedFiberReilly(fiber_class, fiberD)
    else:
        raise ValueError('fiber type unknown')
    psource = ExtracellularCurrent((0, z_min), mode='cathode')
    toffset = 20e-3
    pp = PulsedProtocol(tstim, toffset)

    # Get filecode
    filecodes = fiber.filecodes(psource, 1, pp)
    for k in ['nnodes', 'nodeD', 'rs', 'nodeL', 'interD', 'interL', 'I', 'A', 'inode', 'nature', 'toffset', 'PRF', 'DC']:
        if k in filecodes:
            del filecodes[k]
    filecodes['fiberD'] = f'fiberD{(fiberD * 1e6):.2f}um'
    filecodes['cur'] = f'cur{(n_cur):.0f}_{(cur_min*1e3):.2f}mA-{(cur_max*1e3):.2f}mA'
    filecodes['z'] = f'z{(n_z):.0f}_{(z_min*1e3):.2f}mm-{(z_max*1e3):.2f}mm'
    fcode = '_'.join(filecodes.values())

    # Output file
    fname = f'{fcode}_strengthduration_results.txt'
    fpath = os.path.join(outdir, fname)

    # Computation of the current distance matrix if the file does not exist
    if not os.path.isfile(fpath):
        currents = np.linspace(cur_min, cur_max, n_cur)
        zdistances = np.linspace(z_min, z_max, n_z)
        ExcitationMatrix = [[0 for i in range(n_cur)] for j in range(n_z)]
        for i, I in enumerate(currents):
            for j, z in enumerate(zdistances):
                if I > 0:
                    psource = ExtracellularCurrent((0, z), I, mode='anode')
                elif I < 0:
                    psource = ExtracellularCurrent((0, z), I, mode='cathode')
                data, meta= fiber.simulate(psource, pp)
                ExcitationMatrix[j][i] = fiber.isExcited(data)
        # Save results
        np.savetxt(fpath, ExcitationMatrix)

    # Load results
    ExcitationMatrix = np.loadtxt(fpath)

    return ExcitationMatrix


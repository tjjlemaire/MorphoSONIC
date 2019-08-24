# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-27 14:38:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-23 22:28:03

import os
import pickle
import numpy as np
from neuron import h

from PySONIC.constants import *
from PySONIC.utils import si_format, logger


def loadData(fpath, frequency=1):
    ''' Load dataframe and metadata dictionary from pickle file. '''
    logger.info('Loading data from "%s"', os.path.basename(fpath))
    with open(fpath, 'rb') as fh:
        frame = pickle.load(fh)
        data = frame['data']
        data = {k: df.iloc[::frequency] for k, df in data.items()}
        meta = frame['meta']
        return data, meta


def getNmodlDir():
    ''' Return path to directory containing MOD files and compiled mechanisms files. '''
    selfdir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(selfdir, 'nmodl')


def getSpikesTimings(t, Vm_traces, tstart=None, tend=None, zcross=True):
    ''' Return an array containing occurence times of spikes detected on a collection of nodes.

        :param t: time vector (s)
        :param Vm_traces: vector of membrane potential traces
        :param tstart (optional): starting time for spike detection (s)
        :param tend (optional): end time for spike detection (s)
        :param zcross: boolean stating whether to consider ascending zero-crossings preceding peaks
            as temporal reference for spike occurence timings
        :return: nnodes x nspikes 2D matrix with occurence time (ms) per node and spike.
    '''

    # Discard start or end of traces if needed
    if tstart is not None:
        Vm_traces = [Vm[t >= tstart] for Vm in Vm_traces]
        t = t[t >= tstart]
    if tend is not None:
        Vm_traces = [Vm[t <= tend] for Vm in Vm_traces]
        t = t[t <= tend]

    dt = (t[1] - t[0]) * 1e3  # s
    mpd = int(np.ceil(SPIKE_MIN_DT / dt))
    nspikes = None
    tspikes = []
    for Vm in Vm_traces:

        # Detect spikes on current trace
        ispikes, *_ = findPeaks(Vm, SPIKE_MIN_VAMP, mpd, SPIKE_MIN_VPROM)

        # Assert consistency of spikes propagation
        if nspikes is None:
            nspikes = ispikes.size
            if nspikes == 0:
                print('Warning: no spikes detected')
                return None
        else:
            assert ispikes.size == nspikes, 'Inconsistent number of spikes in different nodes'

        if zcross:
            # Consider spikes as time of zero-crossing preceding each peak
            i_zcross = np.where(np.diff(np.sign(Vm)) > 0)[0]  # detect ascending zero-crossings
            slopes = (Vm[i_zcross + 1] - Vm[i_zcross]) / (t[i_zcross + 1] - t[i_zcross])  # slopes
            offsets = Vm[i_zcross] - slopes * t[i_zcross]  # offsets
            tzcross = -offsets / slopes  # interpolated times
            errmsg = 'Ascending zero crossing #{} (t = {:.2f} ms) not preceding peak #{} (t = {:.2f} ms)'
            for ispike, (tzc, tpeak) in enumerate(zip(tzcross, t[ispikes])):
                assert tzc < tpeak, errmsg.format(ispike, tzc, ispike, tpeak)
            tspikes.append(tzcross)
        else:
            tspikes.append(t[ispikes])

    return np.array(tspikes) * 1e3


def getConductionSpeeds(xcoords, tspikes):
    ''' Compute average conduction speed from simulation results.

        :param xcoords: vector of node longitudinal coordinates (um)
        :param tspikes: nnodes x nspikes 2D matrix with occurence time (ms) per node and spike.
        :return: (nnodes - 1) x nspikesconduction speed matrix (m/s).
    '''
    nnodes, nspikes = tspikes.shape
    dists = np.diff(xcoords)  # internodal distances (um)
    delays = np.abs(np.diff(tspikes, axis=0))  # node-to-node delays (ms)
    return (dists / delays.T).T * 1e-3  # node-to-node conduction velocities (m/s)


def getSpikeAmps(Vm_traces):
    ''' Return an array containing depolarization amplitudes of spikes detected on a collection of nodes.

        :param Vm_traces: vector of membrane potential traces
    '''
    for Vm in Vm_traces:
        print(np.ptp(Vm))

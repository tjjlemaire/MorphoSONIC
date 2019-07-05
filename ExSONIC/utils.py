# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-27 14:38:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-05 14:13:56

import os
import numpy as np
from neuron import h

from PySONIC.postpro import findPeaks
from PySONIC.constants import *
from PySONIC.utils import si_format


def getNmodlDir():
    ''' Return path to directory containing MOD files and compiled mechanisms files. '''
    selfdir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(selfdir, 'nmodl')


def sennGeometry(fiberD=20.):
    ''' Return SENN model geometrical parameters for a given fiber diameter. '''
    nodeD = 0.7 * fiberD  # um
    nodeL = 2.5  # um
    interD = fiberD  # um
    interL = 100 * fiberD  # um
    return [nodeD, nodeL, interD, interL]


def radialGeometry(deff, r1, r2=None, fc=None):
    ''' Return geometrical parameters of cylindrical sections to match ratios of
        membrane vs axial surface areas in a radial configuration between a central
        and a peripheral section.

        :param deff: effective submembrane depth (um)
        :param r1: radius of central section (um)
        :param r2: radius of peripheral section (um)
        :param fc: fraction of membrane surface area covered by central section
        :return: 3-tuple with sections common diameter and their respective lengths (um)
    '''

    # Check for inputs validity
    cond = (r2 is None) + (fc is None)
    if cond == 0 or cond == 2:
        raise ValueError('one of "r2" or "fc" values must be provided')

    # Compute r2 if fc is given
    if r2 is None:
        r2 = r1 / np.sqrt(fc)

    print('radial parameters: deff = {}m, r1 = {}m, r2 = {}m (fc = {:.2f} %)'.format(
        *si_format(np.array([deff, r1, r2]) * 1e-6, 2), fc * 1e2))

    # Compute parameters of cylindrical sections
    d = np.power(4 * deff * r2**2 / np.log((r1 + r2) / r1), 1 / 3)
    L1 = r1**2 / d
    L2 = r2**2 / d - L1

    print('equivalent linear parameters: d = {}m, L1 = {}m, L2 = {}m'.format(
        *si_format(np.array([d, L1, L2]) * 1e-6, 2)))

    # Return tuple
    return (d, [L1, L2])


def VextPointSource(I, r, rho=300.0):
    ''' Compute the extracellular electric potential generated by a given point-current source
        at a given distance in a homogenous, isotropic medium.

        :param I: stimulation current amplitude (uA)
        :param r: euclidian distance(s) between the source and the point(s) of interest (um)
        :param rho: extracellular medium resistivity (Ohm.cm)
        :return: computed extracellular potential(s) (mV)
    '''
    return rho * I / (4 * np.pi * r) * 1e1  # mV


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

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-17 11:59:50
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-05 17:36:33

import logging
import numpy as np
from neuron import h
import matplotlib.pyplot as plt

from PySONIC.utils import logger
from PySONIC.neurons import getPointNeuron
from PySONIC.core import PulsedProtocol

from ExSONIC.core.synapses import *
from ExSONIC.core.node import Node
from ExSONIC.constants import *

logger.setLevel(logging.INFO)


def getSynapseData(syn_model, sf, syn_delay=1., tstop=4.):

    logger.info(f'Simulating {name} synapse with {sf:.0f} Hz input')

    # Define input cell: artificial spike generator
    stim = h.NetStim()
    stim.number = 1e3
    stim.start = 0.
    stim.interval = S_TO_MS / sf

    # Define output cell: RS node
    node = Node(getPointNeuron('RS'))

    # Attach synapse model to output cell
    syn = syn_model.attach(node)

    # Connect input and output
    nc = h.NetCon(stim, syn)
    nc.delay = syn_delay
    nc.weight[0] = 1.

    # Record spike times and synaptic conductance
    tspikes = h.Vector()
    nc.record(tspikes)
    g = h.Vector()
    g.record(syn._ref_g)

    # Simulate
    data, meta = node.simulate(0., PulsedProtocol(tstop, 0.))

    # Retrieve spikes timings
    tspikes = np.array(tspikes.to_python())

    # Retrieve time and conductance vectors
    t = data['t'].values[1:] * S_TO_MS
    g = np.array(g.to_python())

    return tspikes, t, g


# -------------------------- MAIN --------------------------

# Synapse models
exp2syn_model = Exp2Synapse(tau1=0.1, tau2=3.0, E=0.0)
fexp2syn_model = FExp2Synapse(
    tau1=exp2syn_model.tau1, tau2=exp2syn_model.tau2, E=exp2syn_model.E, f=0.2, tauF=200.0)
fdexp2syn_model = FDExp2Synapse(
    tau1=exp2syn_model.tau1, tau2=exp2syn_model.tau2, E=exp2syn_model.E, f=0.5, tauF=94.0,
    d1=0.46, tauD1=380.0, d2=0.975, tauD2=9200.0)

syn_models = {x.__class__.__name__: x for x in [exp2syn_model, fexp2syn_model, fdexp2syn_model]}

# Spiking frequencies
spike_freqs = np.array([1., 10., 100.])  # Hz

for sf in spike_freqs:

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f'synapses behavior with {sf:.0f} Hz input')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('$g\ /\ g_0$')
    for name, syn_model in syn_models.items():
        tspikes, t, g = getSynapseData(syn_model, sf)
        ax.plot(t, g, label=name)
    ax.vlines(tspikes, -0.2, -0.1, color='k', label='$t_{spikes}$')
    ax.legend()

plt.show()

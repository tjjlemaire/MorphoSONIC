# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-13 19:51:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-01-14 17:24:17

import logging
import numpy as np
import matplotlib.pyplot as plt
from neuron import h

from PySONIC.core import PulsedProtocol
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger

from ExSONIC.plt import SectionCompTimeSeries
from ExSONIC.core import IintraNode, Network, Exp2Synapse

''' Create and simulate a small network of nodes. '''

logger.setLevel(logging.DEBUG)

# Nodes
pneurons = {k: getPointNeuron(k) for k in ['RS', 'FS', 'LTS']}
nodes = {k: IintraNode(v) for k, v in pneurons.items()}

# Synapse models
synapses = {
    'RS': Exp2Synapse(0.1, 3.0, 0.0),
    'FS': Exp2Synapse(0.5, 8.0, -85.0),
    'LTS': Exp2Synapse(0.5, 50.0, -85.0)
}

# Synaptic weights (uS)
weights = {
    'RS': {
        'RS': 0.002,
        'FS': 0.04,
        'LTS': 0.09
    },
    'FS': {
        'RS': 0.015,
        'FS': 0.135,
        'LTS': 0.86
    },
    'LTS': {
        'RS': 0.135,
        'FS': 0.02
    }
}

# Construct node network
network = Network(nodes, synapses, weights)

# Connect network nodes
for source, targets in weights.items():
    syn_model = synapses[source]
    for target, weight in targets.items():
        network.connect(source, target, syn_model, weight)

# Stimulation parameters
pp = PulsedProtocol(1., 1.)
I_Th_RS = 0.17  # nA
thalamic_drives = {  # drive currents (nA)
    'RS': I_Th_RS,
    'FS': 1.4 * I_Th_RS,
    'LTS': 0.0}
amps = {  # drive current density (mA/m2)
    k: (Idrive * 1e-6) / pneurons[k].area for k, Idrive in thalamic_drives.items()}

# Simulation
data, meta = network.simulate(amps, pp)

# Plot membrane potential traces for specific duration at threshold current
fig1 = SectionCompTimeSeries([(data, meta)], 'Vm', network.ids).render()

plt.show()

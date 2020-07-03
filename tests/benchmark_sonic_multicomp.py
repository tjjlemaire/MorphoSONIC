# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-06-29 18:11:24
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-07-03 12:23:42

import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.utils import logger

from PySONIC.neurons import getPointNeuron
from ExSONIC.core import SonicBenchmark
from ExSONIC.core import SennFiber, UnmyelinatedFiber

logger.setLevel(logging.INFO)

# Stimulation parameters
f = 500.             # US frequency (kHz)
rel_amps = (0.8, 0)  # relative capacitance oscillation amplitudes (active node as in Plaksin 2016)


# Benchmark 1: exploring the parameter space to identify conditions inducing
# a divergence between the full and SONIC pardigms
mechs = ['FHnode', 'SUseg']
pneurons = [getPointNeuron(k) for k in mechs]
for pneuron in pneurons:
    tau = pneuron.tau_pas * 1e3  # passive membrane time constant (ms)
    logger.info(f'passive {pneuron} model -> tau = {tau:.2f} ms')
    tstop = 5 * tau  # ms
    sb = SonicBenchmark(pneuron, 1e0, f, rel_amps, passive=True)
    ga_thr = sb.findThresholdAxialCoupling(tstop)
    gpas_ratio = ga_thr / sb.gPas
    logger.info(f'Threshold axial coupling: ga = {ga_thr:.2f} mS/cm2 ({gpas_ratio:.2f} * gpas)')
    fig = sb.benchmark(tstop)

# # Benchmark 2: with full membrane dynamics and axial coupling values
# # of realistic multicomp models

# full_benchmarks = []
# tstop = 10.  # ms

# # Fiber models
# fibers = [
#     SennFiber(10e-6, 11),                   # 10 um diameter SENN fiber
#     UnmyelinatedFiber(0.8e-6, fiberL=5e-3)  # 0.8 um diameter unmylinated fiber
# ]
# for fiber in fibers:
#     Ga = 1 / fiber.R_node_to_node    # S
#     Anode = fiber.nodes['node0'].Am  # cm2
#     ga = Ga / Anode * 1e3            # mS/cm2
#     logger.info(f'Node-to-node coupling in {fiber}: ga = {ga:.2f} mS/cm2')
#     full_benchmarks.append((fiber.pneuron, ga))

# for pneuron, ga in full_benchmarks:
#     fig = SonicBenchmark(pneuron, ga, f, rel_amps).benchmark(tstop)

plt.show()

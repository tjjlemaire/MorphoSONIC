# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 19:30:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-14 14:48:15

import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

from PySONIC.neurons import getPointNeuron
from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger, si_format
from ExSONIC.test import TestFiber
from ExSONIC.core import SonicFiber, myelinatedFiberReilly, unmyelinatedFiberSundt
from ExSONIC.core.sources import *
from ExSONIC.plt import SectionCompTimeSeries, strengthDurationCurve, strengthDistanceCurve
from ExSONIC.utils import chronaxie


class TestSennAstim(TestFiber):

    a = 32e-9       # sonophore diameter (m)
    fs = 1          # sonophore membrane coverage (-)
    Fdrive = 500e3  # US frequency (Hz)

    def test_node(self, is_profiled=False):
        ''' Run myelinated fiber ASTIM simulation with node source. '''
        logger.info('Test: node source on myelinated fiber')

        # Myelinated fiber model
        fiber = myelinatedFiberReilly(SonicFiber, a=self.a, fs=self.fs)

        # US stimulation parameters
        psource = NodeAcousticSource(fiber.nnodes // 2, self.Fdrive)
        pp = PulsedProtocol(3e-3, 3e-3)

        # Titrate for a specific duration and simulate fiber at threshold US amplitude
        logger.info(f'Running titration for {si_format(pp.tstim)}s pulse')
        Athr = fiber.titrate(psource, pp)  # Pa
        data, meta = fiber.simulate(psource.updatedX(1.2 * Athr), pp)

        # Compute conduction velocity and spike amplitude from resulting data
        sim_metrics = {
           'Athr': Athr,                             # Pa
           'cv': fiber.getConductionVelocity(data),  # m/s
           'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Plot membrane potential and membrane charge density traces
        fig1 = SectionCompTimeSeries([(data, meta)], 'Qm', fiber.ids).render()
        fig2 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

        # Comparative SD curve
        durations = np.logspace(-5, -3, 20)  # s
        toffset = 10e-3                     # s
        pps = [PulsedProtocol(t, toffset) for t in durations]
        Athrs = np.array([fiber.titrate(psource, pp) for pp in pps])

        # Plot strength-duration curve
        fig3 = strengthDurationCurve(
            fiber, durations, {'myelinated': Athrs}, scale='log',
            yname='amplitude', yfactor=1e-3, yunit='Pa', plot_chr=False)

        # Log output metrics
        self.logOutputMetrics(sim_metrics)

    def test_gaussian(self, is_profiled=False):
        ''' Run myelinated fiber ASTIM simulation with gaussian distribution source. '''
        logger.info('Test: gaussian distribution source on myelinated fiber')

        # Myelinated fiber model
        fiber = myelinatedFiberReilly(SonicFiber, a=self.a, fs=self.fs)

        # US stimulation parameters
        psource = GaussianAcousticSource(0., fiber.length() / 4., self.Fdrive)
        pp = PulsedProtocol(3e-3, 3e-3)

        # Titrate for a specific duration and simulate fiber at threshold US amplitude
        logger.info(f'Running titration for {si_format(pp.tstim)}s pulse')
        Athr = fiber.titrate(psource, pp)  # Pa
        data, meta = fiber.simulate(psource.updatedX(1.2 * Athr), pp)

        # Compute conduction velocity and spike amplitude from resulting data
        sim_metrics = {
           'Athr': Athr,                             # Pa
           'cv': fiber.getConductionVelocity(data),  # m/s
           'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Plot membrane potential and membrane charge density traces
        fig1 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

        # Comparative SD curve
        durations = np.logspace(-5, -3, 20)  # s
        toffset = 10e-3                     # s
        pps = [PulsedProtocol(t, toffset) for t in durations]
        Athrs = np.array([fiber.titrate(psource, pp) for pp in pps])

        # Plot strength-duration curve
        fig2 = strengthDurationCurve(
            fiber, durations, {'myelinated': Athrs}, scale='log',
            yname='amplitude', yfactor=1e-3, yunit='Pa', plot_chr=False)

        # Log output metrics
        self.logOutputMetrics(sim_metrics)

    def transducer(self, fiber, pp):
        ''' Run SENN fiber ASTIM simulations with a flat external transducer. '''
        # US source
        diam = 19e-3  # transducer diameter (m)
        source = PlanarDiskTransducerSource((0., 0., 'focus'), self.Fdrive, r=diam/2)

        # Titrate for a specific duration and simulate fiber at threshold particle velocity
        logger.info(f'Running titration for {si_format(pp.tstim)}s pulse')
        uthr = fiber.titrate(source, pp)  # m/s
        Athr = uthr * source.relNormalAxisAmp(source.z)  # Pa
        data, meta = fiber.simulate(source.updatedX(1.2 * uthr), pp)

        # Compute conduction velocity and spike amplitude from resulting data
        sim_metrics = {
            'uthr': uthr,                             # m/s
            'Athr': Athr,                             # Pa
            'cv': fiber.getConductionVelocity(data),  # m/s
            'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Plot membrane potential traces for specific duration at threshold current
        fig = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

        # Log output metrics
        self.logOutputMetrics(sim_metrics)

    def test_transducer1(self, is_profiled=False):
        logger.info('Test: transducer source on myelinated fiber')
        fiber = myelinatedFiberReilly(SonicFiber, a=self.a, fs=self.fs)
        pp = PulsedProtocol(100e-6, 3e-3)
        self.transducer(fiber, pp)

    def test_transducer2(self, is_profiled=False):
        logger.info('Test: transducer source on myelinated fiber')
        fiber = unmyelinatedFiberSundt(SonicFiber, a=self.a, fs=self.fs)
        pp = PulsedProtocol(10e-3, 3e-3)
        self.transducer(fiber, pp)

if __name__ == '__main__':
    tester = TestSennAstim()
    tester.main()
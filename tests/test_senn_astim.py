# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 19:30:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-05 16:55:30

import numpy as np

from PySONIC.neurons import getPointNeuron
from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger, si_format
from ExSONIC.test import TestFiber
from ExSONIC.core import SonicFiber, myelinatedFiberReilly, unmyelinatedFiberSundt
from ExSONIC.core import NodeAcousticSource, PlanarDiskTransducerSource
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

    def test_transducer(self, is_profiled=False):
        ''' Run SENN fiber ASTIM simulation for a flat external transducer. '''
        logger.info('Test: transducer source on myelinated fiber')

        # Myelinated fiber model
        fiber = myelinatedFiberReilly(SonicFiber, a=self.a, fs=self.fs)

        # US stimulation parameters
        pp = PulsedProtocol(3e-3, 3e-3)  # pulsing protocol
        x = (0., 0., -fiber.interL)      # transducer coordinates (m)

        # Create ultrasound source
        psource = PlanarDiskTransducerSource(x, self.Fdrive)

        # Titrate for a specific duration and simulate fiber at threshold particle velocity
        logger.info(f'Running titration for {si_format(pp.tstim)}s pulse')
        uthr = fiber.titrate(psource, pp)  # m/s
        data, meta = fiber.simulate(psource.updatedX(1.2 * uthr), pp)

        # Compute conduction velocity and spike amplitude from resulting data
        sim_metrics = {
            'uthr': uthr,                             # m/s
            'cv': fiber.getConductionVelocity(data),  # m/s
            'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Plot membrane potential traces for specific duration at threshold current
        fig1 = SectionCompTimeSeries([(data, meta)], 'Qm', fiber.ids).render()
        fig2 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

        # Comparative SD curve
        durations = np.logspace(-5, -3, 20)  # s
        toffset = 10e-3                     # s
        pps = [PulsedProtocol(t, toffset) for t in durations]
        uthrs = np.array([fiber.titrate(psource, pp) for pp in pps])

        # Plot strength-duration curve
        fig3 = strengthDurationCurve(
            fiber, durations, {'myelinated': uthrs}, scale='log',
            yname='particle velocity', yfactor=1e3, yunit='m/s', plot_chr=False)

        # Log output metrics
        self.logOutputMetrics(sim_metrics)


if __name__ == '__main__':
    tester = TestSennAstim()
    tester.main()
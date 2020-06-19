# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 19:30:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-06-19 15:33:26

import numpy as np

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger, si_format
from ExSONIC.test import TestFiber
from ExSONIC.core import SennFiber, UnmyelinatedFiber, MRGFiber
from ExSONIC.core.sources import *
from ExSONIC.plt import SectionCompTimeSeries, strengthDurationCurve


class TestFiberAstim(TestFiber):

    a = 32e-9       # sonophore diameter (m)
    fs = 1          # sonophore membrane coverage (-)
    Fdrive = 500e3  # US frequency (Hz)

    def test_node(self, is_profiled=False):
        ''' Run myelinated fiber ASTIM simulation with node source. '''
        logger.info('Test: local source on myelinated fiber node')

        # Myelinated fiber model
        fiberD = 20e-6  # m
        nnodes = 21
        fiber = SennFiber(fiberD, nnodes, a=self.a, fs=self.fs)

        # US stimulation parameters
        psource = SectionAcousticSource(fiber.central_ID, self.Fdrive)
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
        SectionCompTimeSeries([(data, meta)], 'Qm', fiber.nodeIDs).render()
        SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

        # Comparative SD curve
        durations = np.logspace(-5, -3, 20)  # s
        toffset = 10e-3                     # s
        pps = [PulsedProtocol(t, toffset) for t in durations]
        Athrs = np.array([fiber.titrate(psource, pp) for pp in pps])

        # Plot strength-duration curve
        strengthDurationCurve(
            fiber, durations, {'myelinated': Athrs}, scale='log',
            yname='amplitude', yfactor=PA_TO_KPA, yunit='Pa', plot_chr=False)

        # Log output metrics
        self.logOutputMetrics(sim_metrics)

    def gaussian(self, fiber, pp):
        ''' Run myelinated fiber ASTIM simulation with gaussian distribution source. '''
        # US source (gaussian distribution with 10 mm width)
        source = GaussianAcousticSource(0., 10e-3, self.Fdrive)

        # Titrate for a specific duration and simulate fiber at threshold US amplitude
        logger.info(f'Running titration for {si_format(pp.tstim)}s pulse')
        Athr = fiber.titrate(source, pp)  # Pa
        data, meta = fiber.simulate(source.updatedX(1.2 * Athr), pp)

        # Compute conduction velocity and spike amplitude from resulting data
        sim_metrics = {
           'Athr': Athr,                             # Pa
           'cv': fiber.getConductionVelocity(data),  # m/s
           'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Plot membrane potential and membrane charge density traces
        varkeys = ['Vm'] if not fiber.has_ext_mech else ['Vm', 'Vext', 'Vin']
        for stype, sdict in fiber.sections.items():
            for k in varkeys:
                fig = SectionCompTimeSeries([(data, meta)], k, sdict.keys()).render()
                fig.axes[0].set_title(f'{fiber} - {stype}s {k} traces')

        # # Comparative SD curve
        # durations = np.logspace(-5, -3, 20)  # s
        # toffset = 10e-3                     # s
        # pps = [PulsedProtocol(t, toffset) for t in durations]
        # Athrs = np.array([fiber.titrate(source, pp) for pp in pps])

        # # Plot strength-duration curve
        # fig2 = strengthDurationCurve(
        #     fiber, durations, {'myelinated': Athrs}, scale='log',
        #     yname='amplitude', yfactor=PA_TO_KPA, yunit='Pa', plot_chr=False)

        # Log output metrics
        self.logOutputMetrics(sim_metrics)

    def test_gaussian1(self, is_profiled=False):
        logger.info('Test: gaussian distribution source on myelinated fiber')
        fiber = SennFiber(20e-6, 21, a=self.a, fs=self.fs)
        pp = PulsedProtocol(3e-3, 3e-3, tstart=0.1e-3)
        return self.gaussian(fiber, pp)

    def test_gaussian2(self, is_profiled=False):
        logger.info('Test: gaussian distribution source on unmyelinated fiber')
        fiber = UnmyelinatedFiber(0.8e-6, a=self.a, fs=self.fs)
        pp = PulsedProtocol(10e-3, 3e-3, tstart=0.1e-3)
        return self.gaussian(fiber, pp)

    def test_gaussian3(self, is_profiled=False):
        logger.info('Test: gaussian distribution source on myelinated MRG fiber')
        fiber = MRGFiber(20e-6, 21, a=self.a, fs=self.fs)
        pp = PulsedProtocol(3e-3, 3e-3, tstart=0.1e-3)
        return self.gaussian(fiber, pp)

    def transducer(self, fiber, pp):
        ''' Run SENN fiber ASTIM simulations with a flat external transducer. '''
        # US source
        diam = 19e-3  # transducer diameter (m)
        source = PlanarDiskTransducerSource((0., 0., 'focus'), self.Fdrive, r=diam / 2)

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
        fig = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

        # Log output metrics
        self.logOutputMetrics(sim_metrics)

    def test_transducer1(self, is_profiled=False):
        logger.info('Test: transducer source on myelinated fiber')
        fiber = SennFiber(20e-6, 21, a=self.a, fs=self.fs)
        pp = PulsedProtocol(100e-6, 3e-3)
        self.transducer(fiber, pp)

    def test_transducer2(self, is_profiled=False):
        logger.info('Test: transducer source on unmyelinated fiber')
        fiber = UnmyelinatedFiber(0.8e-6, a=self.a, fs=self.fs)
        pp = PulsedProtocol(10e-3, 3e-3)
        self.transducer(fiber, pp)


if __name__ == '__main__':
    tester = TestFiberAstim()
    tester.main()

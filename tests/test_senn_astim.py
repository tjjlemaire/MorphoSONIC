# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 19:30:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-12-02 20:11:37

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

    refs = {
        'a': 32e-9,       # sonophore diameter (m)
        'Fdrive': 500e3,  # US frequency (Hz)
        'fs': 1           # sonophore membrane coverage (-)
    }

    def test_myelinated(self, is_profiled=False):
        ''' Run myelinated fiber ASTIM simulation. '''
        logger.info('Test: myelinated fiber')

        # Myelinated fiber model
        fiber = myelinatedFiberReilly(SonicFiber, **self.refs)

        # US stimulation parameters
        psource = NodeAcousticSource(fiber.nnodes // 2, self.refs['Fdrive'])
        pp = PulsedProtocol(3e-3, 3e-3)

        # Titrate for a specific duration and simulate fiber at threshold US amplitude
        logger.info(f'Running titration for {si_format(pp.tstim)}s pulse')
        Athr = fiber.titrate(psource, pp)  # Pa
        data, meta = fiber.simulate(psource, 1.2 * Athr, pp)

        # Compute conduction velocity and spike amplitude from resulting data
        sim_metrics = {
           'Athr': Athr,                             # Pa
           'cv': fiber.getConductionVelocity(data),  # m/s
           'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Log output metrics
        self.logOutputMetrics(sim_metrics)

        # Plot membrane potential and membrane charge density traces
        fig1 = SectionCompTimeSeries([(data, meta)], 'Qm', fiber.ids).render()
        fig2 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

    def test_unmyelinated(self, is_profiled=False):
        ''' Run unmyelinated C-fiber ASTIM simulation '''
        logger.info('Test: unmyelinated fiber')

        # Unmyelinated fiber model
        fiber = unmyelinatedFiberSundt(SonicFiber, **self.refs)

        # US stimulation parameters
        psource = NodeAcousticSource(fiber.nnodes // 2, self.refs['Fdrive'])
        pp = PulsedProtocol(10e-3, 10e-3)

        # Titrate for a specific duration and simulate fiber at threshold US amplitude
        logger.info(f'Running titration for {si_format(pp.tstim)}s pulse')
        Athr = fiber.titrate(psource, pp)  # Pa
        data, meta = fiber.simulate(psource, 1.2 * Athr, pp)

        # Compute conduction velocity and spike amplitude from resulting data
        sim_metrics = {
           'Athr': Athr,                             # Pa
           'cv': fiber.getConductionVelocity(data),  # m/s
           'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Log output metrics
        self.logOutputMetrics(sim_metrics)

        # Plot membrane potential and membrane charge density traces
        fig1 = SectionCompTimeSeries([(data, meta)], 'Qm', fiber.ids).render()
        fig2 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

    def test_SDcurveVsFiberType(self, is_profiled=False):
        logger.info('Comparison: strength-duration curves for different fiber types')

        # Create typical fiber models
        fibers = {
            'myelinated': myelinatedFiberReilly(SonicFiber, **self.refs),
            'unmyelinated': unmyelinatedFiberSundt(SonicFiber, **self.refs)
        }

        # US stimulation parameters
        psources = {k: NodeAcousticSource(v.nnodes // 2, self.refs['Fdrive'])
                    for k, v in fibers.items()}
        durations = np.logspace(-5, 0, 10)  # s
        toffset = 10e-3                      # s
        pps = [PulsedProtocol(t, toffset) for t in durations]

        # Comparative SD curves
        Athrs = {k: np.array([v.titrate(psources[k], pp) for pp in pps])
                 for k, v in fibers.items()}
        fig1 = strengthDurationCurve(
            'comparison', durations, Athrs, scale='log',
            yname='amplitude', yfactor=1e-3, yunit='Pa', plot_chr=False)

    def test_transducer(self, is_profiled=False):
        ''' Run SENN fiber ASTIM simulation for a flat external transducer. '''
        logger.info('Test: SENN model ASTIM stimulation from external transducer')

        # Fiber model parameters
        pneuron = getPointNeuron('FH')  # Frog myelinated node membrane equations
        fiberD = 10e-6                  # fiber diameter (m)
        nnodes = 15
        rho_a = 54.7                    # axoplasm resistivity (Ohm.cm)
        d_ratio = 0.6                   # axon / fiber diameter ratio
        nodeL = 1.5e-6                  # node length (m)
        a = 32e-9                       # sonophore diameter (m)
        Fdrive = 500e3                  # US frequency (Hz)
        fs = 1                          # sonophore membrane coverage (-)
        fiber = myelinatedFiber(SonicFiber, pneuron, fiberD, nnodes,
            rs=rho_a, nodeL=nodeL, d_ratio=d_ratio, a=a, Fdrive=Fdrive, fs=fs)

        # US stimulation parameters
        pp = PulsedProtocol(3e-3, 3e-3)

        # Transducer parameters
        z0 = fiber.interL  # default transducer z-location, m
        x0 = 0             # transducer initial x-location, m
        rho = 1204.1       # medium density (kg/m3)
        c = 1515.0         # speed of sound (m/s)
        l = 0.01           # length of a flat lxl transducer surface (m)
        theta = 0          # transducer angle of incidence (radians)
        r_tr = 1.27e-3     # transducer radius (m)

        # Create ultrasound source
        psource = PlanarDiskTransducerSource((x0, z0), Fdrive, rho=rho, c=c, theta=theta, r=r_tr)

        # Titrate for a specific duration and simulate fiber at threshold particle velocity
        logger.info(f'Running titration for {si_format(pp.tstim)}s pulse')
        uthr = fiber.titrate(psource, pp)  # m/s
        data, meta = fiber.simulate(psource, 1.2 * uthr, pp)

        # Compute conduction velocity and spike amplitude from resulting data
        sim_metrics = {
            'uthr': uthr,                             # m/s
            'cv': fiber.getConductionVelocity(data),  # m/s
            'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Log output metrics
        self.logOutputMetrics(sim_metrics)

        # Plot membrane potential traces for specific duration at threshold current
        fig1 = SectionCompTimeSeries([(data, meta)], 'Qm', fiber.ids).render()
        fig2 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()


if __name__ == '__main__':
    tester = TestSennAstim()
    tester.main()
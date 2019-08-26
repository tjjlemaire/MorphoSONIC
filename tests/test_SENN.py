# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 19:30:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-26 10:45:02

import numpy as np
import logging
import matplotlib.pyplot as plt

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format
from PySONIC.test import TestBase
from ExSONIC.core import VextSennFiber, CurrentPointSource
from ExSONIC.plt import SectionCompTimeSeries


class TestSenn(TestBase):

    def test_ESTIM(self, is_profiled=False):

        # Fiber model
        pneuron = getPointNeuron('FH')
        fiberD = 20e-6  # m
        nnodes = 11
        rs = 110.0  # Ohm.cm
        fiber = VextSennFiber(pneuron, fiberD, nnodes, rs=rs)

        # Electrode type and location
        z0 = fiber.interL  # one internodal distance away from fiber m
        x0 = 0.  # m
        mode = 'cathode'

        # Stimulation Parameters
        toffset = 3e-3  # s
        PRF = 100.
        DC = 1.

        # Create point-source electrode
        psource = CurrentPointSource(x0, z0, mode=mode)

        # Titrate for a specific duration and simulate fiber at threshold current
        tstim = 100e-6  # s
        Ithr = fiber.titrate(psource, tstim, toffset, PRF, DC)
        logger.info(f'tstim = {si_format(tstim)}s -> Ithr = {si_format(Ithr)}A')
        data, meta = fiber.simulate(psource, Ithr, tstim, toffset, PRF, DC)

        # Compute strength-duration curve
        durations = np.logspace(-4, 0, 20)  # s
        Ithrs = np.array([fiber.titrate(psource, x, toffset, PRF, DC) for x in durations])  # A

        # Plot membrane potential traces for specific duration at threshold current
        fig1 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

        # Plot strength-duration curve
        fs = 12
        fig2, ax = plt.subplots()
        ax.set_title(f'{fiber} - strength-duration curve', fontsize=fs)
        ax.set_xlabel('duration (ms)', fontsize=fs)
        ax.set_ylabel('threshold cathodic current (mA)', fontsize=fs)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(durations * 1e3, -Ithrs * 1e3)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)

        plt.show()



if __name__ == '__main__':
    tester = TestSenn()
    tester.main()
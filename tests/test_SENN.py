# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 19:30:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-23 20:35:09

import numpy as np
import logging
import matplotlib.pyplot as plt

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format
from PySONIC.test import TestBase
from ExSONIC.core import VextSennFiber, CurrentPointSource
from ExSONIC.plt import SectionCompTimeSeries


class TestSenn(TestBase):
    ''' Run IintraNode (ESTIM) and SonicNode (ASTIM) simulations with manually and automatically created
        NMODL membrane mechanisms, and compare results.
    '''

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
        tstim = 100e-6  # s
        toffset = 3e-3  # s
        PRF = 100.
        DC = 1.

        # Create point-source electrode
        psource = CurrentPointSource(x0, z0, mode=mode)

        # Titrate
        Ithr = fiber.titrate(psource, tstim, toffset, PRF, DC)
        logger.info(f'Ithr = {si_format(Ithr)}A')

        # Simulate fiber at threshold current
        data, meta = fiber.simulate(psource, Ithr, tstim, toffset, PRF, DC)

        # Plot resulting membrane potential traces
        sections = fiber.ids
        varname = 'Vm'
        filepath = [(data, meta)]
        fig = SectionCompTimeSeries(filepath, varname, sections).render()
        plt.show()



if __name__ == '__main__':
    tester = TestSenn()
    tester.main()
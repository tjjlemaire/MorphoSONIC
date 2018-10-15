# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-25 02:00:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-15 20:33:06

import numpy as np
from neuron import h


class SeriesConnector:
    ''' The SeriesConnector class allows to connect model sections in series through a
        by inserting axial current as a distributed membrane mechanism in those sections, thereby
        allowing to use any voltage variable (not necessarily 'v') as a reference to compute
        axial currents.

        :param mechname: name of the mechanism that compute axial current contribution
        :param vref: name of the reference voltage varibale to compute axial currents
        :param rmin (optional): lower bound for axial resistance density (Ohm.cm2)
    '''

    def __init__(self, mechname='Iax', vref='v', rmin=2e2):
        self.mechname = mechname
        self.vref = vref
        self.rmin = rmin  # Ohm.cm2

    def __repr__(self):
        return 'connect("{}"{})'.format(
            self.vref,
            ', rmin={:.2e}_Ohm.cm2'.format(self.rmin) if self.rmin is not None else ''
        )

    def __str__(self):
        return 'Series connector object: reference voltage variable = "{}"{}'.format(
            self.vref, ', minimal resistance density = {:.2e} Ohm.cm2'.format(self.rmin)
            if self.rmin is not None else '')

    def membraneArea(self, sec):
        ''' Compute section membrane surface area (in cm2) '''
        return np.pi * (sec.diam * 1e-4) * (sec.L * 1e-4)

    def axialArea(self, sec):
        ''' Compute section axial area (in cm2) '''
        return np.pi * (sec.diam * 1e-4)**2 / 4

    def resistance(self, sec):
        ''' Compute section axial resistance (in Ohm) '''
        return sec.Ra * (sec.L * 1e-4) / self.axialArea(sec)

    def attach(self, sec, verbose=False):
        ''' Insert density mechanism to section and set appropriate axial conduction parameters.

            :param sec: section to attach
            :param verbose: boolean indicating whether to output details of the connection process
        '''

        # Insert axial current density mechanism
        sec.insert(self.mechname)

        # Compute section properties
        Am = self.membraneArea(sec)  # membrane surface area (cm2)
        R = self.resistance(sec)  # section resistance (Ohm)

        # Optional: bound resistance to ensure (resistance * membrane area) is always above threshold,
        # in order to limit the magnitude of axial currents and thus ensure convergence of simulation
        if self.rmin is not None:
            if R < self.rmin / Am:
                if verbose:
                    print('R*Am = {:.1e} Ohm.cm2 -> bounded to {:.1e} Ohm.cm2'.format(
                        R * Am, self.rmin))
                R = self.rmin / Am
            elif verbose:
                print('R*Am = {:.1e} Ohm.cm2 -> not bounded'.format(R * Am))

        # Set section propeties to Iax mechanism
        setattr(sec, 'R_{}'.format(self.mechname), R)
        setattr(sec, 'Am_{}'.format(self.mechname), Am)
        h.setpointer(getattr(sec(0.5), '_ref_{}'.format(self.vref)), 'V',
                     getattr(sec(0.5), self.mechname))

        # While section not connected: set neighboring sections' properties (resistance and
        # membrane potential) as those of current section
        for suffix in ['prev', 'next']:
            setattr(sec, 'R{}_{}'.format(suffix, self.mechname), R)  # Ohm
            h.setpointer(getattr(sec(0.5), '_ref_{}'.format(self.vref)),
                         'V{}'.format(suffix), getattr(sec(0.5), self.mechname))

    def connect(self, sec1, sec2):
        ''' Connect two adjacent sections in series to enable trans-sectional axial current. '''

        # Inform sections about each other's axial resistance (in Ohm)
        setattr(sec1, 'Rnext_{}'.format(self.mechname), getattr(sec2, 'R_{}'.format(self.mechname)))
        setattr(sec2, 'Rprev_{}'.format(self.mechname), getattr(sec1, 'R_{}'.format(self.mechname)))

        # Set bi-directional pointers to sections about each other's membrane potential
        h.setpointer(getattr(sec1(0.5), '_ref_{}'.format(self.vref)),
                     'Vprev', getattr(sec2(0.5), self.mechname))
        h.setpointer(getattr(sec2(0.5), '_ref_{}'.format(self.vref)),
                     'Vnext', getattr(sec1(0.5), self.mechname))

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-04 13:25:53


class SerialConnectionScheme:
    ''' Connection scheme object spcifying the coupling variable for intracellular currents,
        and an optional lower bound for axial resistance normalized by membrane area.
    '''

    def __init__(self, vref='v', rmin=None):
        ''' Initialization.

            :param vref: reference coupling variable
            :param rmin: lower bound for axial resistance * membrane area (Ohm/cm2)

            ..note: The lower bound is defined here as membrane area normalized in order
            to directly limit the amplitude of axial currents compared to that of membrane
            currents, regardless of the sections' dimensions.
        '''
        self.vref = vref
        self.rmin = rmin

    def __repr__(self):
        s = f'{self.__class__.__name__}(vref={self.vref}'
        if self.rmin is not None:
            s = f'{s}, rmin={self.rmin:.2e} Ohm.cm2'
        return f'{s})'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.vref == other.vref and self.rmin == other.rmin

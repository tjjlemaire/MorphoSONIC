# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-03-06 09:06:17


class SerialConnectionScheme:

    def __init__(self, vref='v', rmin=None):
        self.vref = vref
        self.rmin = rmin

    def __repr__(self):
        s = f'{self.__class__.__name__}(vref={self.vref}'
        if self.rmin is not None:
            s = f'{s}, rmin={self.rmin:.2e} Ohm.cm2'
        return f'{s})'

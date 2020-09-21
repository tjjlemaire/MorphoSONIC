# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-09-21 16:45:38
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-09-21 17:20:17

import numpy as np

dataroot = 'C:\\Users\\lemaire\\Desktop\\morphoSONIC\\data'
figdir = 'C:\\Users\\lemaire\\Desktop\\morphoSONIC\\figs'
fontsize = 10


def setAxis(ax, precision, signed, axkey='y'):

    lim_getter = getattr(ax, f'get_{axkey}lim')
    lim_setter = getattr(ax, f'set_{axkey}lim')
    tick_setter = getattr(ax, f'set_{axkey}ticks')
    ticklabel_setter = getattr(ax, f'set_{axkey}ticklabels')

    rfactor = np.power(10, precision)
    lims = lim_getter()
    lims = [np.floor(lims[0] * rfactor) / rfactor, np.ceil(lims[1] * rfactor) / rfactor]
    fmt = f'{"+" if signed else ""}.{precision}f'
    lim_setter(*lims)
    tick_setter(lims)
    ticklabel_setter([f'{y:{fmt}}' for y in lims])

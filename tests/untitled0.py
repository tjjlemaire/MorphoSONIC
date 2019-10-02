# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:38:52 2019

@author: Maria
"""

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-11 15:58:38
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-12 11:15:17

import numpy as np

from ..core import PointNeuron


class Sundt(PointNeuron):
    ''' Template neuron class '''

    # Neuron name
    name = 'Sundt'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-6   # Membrane capacitance (F/m2)
    Vm0 = -60  # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 55.0     # Sodium
    EK = -90.0     # Potassium
    ELeak = -70.3  # Non-specific leakage ???

    # Maximal channel conductances (S/m2)
    gNabar = 560.0  # Sodium
    gKdbar = 60.0   # Delayed-rectifier Potassium
    gLeak = 0.205   # Non-specific leakage

    # Additional parameters
    VT = -56.2  # Spike threshold adjustment parameter (mV)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    @classmethod
    def alpham(cls, Vm):
        return 0.32 * cls.vtrap(13 - (Vm - cls.VT), 4) * 1e3  # s-1

    @classmethod
    def betam(cls, Vm):
        return 0.28 * cls.vtrap((Vm - cls.VT) - 40, 5) * 1e3  # s-1

    @classmethod
    def alphah(cls, Vm):
        return 0.128 * np.exp(-((Vm - cls.VT) - 17) / 18) * 1e3  # s-1

    @classmethod
    def betah(cls, Vm):
        return 4 / (1 + np.exp(-((Vm - cls.VT) - 40) / 5)) * 1e3  # s-1

    @classmethod
    def alphan(cls, Vm):
        return 0.032 * cls.vtrap(15 - (Vm - cls.VT), 5) * 1e3  # s-1

    @classmethod
    def betan(cls, Vm):
        return 0.5 * np.exp(-((Vm - cls.VT) - 10) / 40) * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'],
            'n': lambda Vm, x: cls.alphan(Vm) * (1 - x['n']) - cls.betan(Vm) * x['n']
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm)),
            'n': lambda Vm: cls.alphan(Vm) / (cls.alphan(Vm) + cls.betan(Vm))
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iNa(cls, m, h, Vm):
        ''' Sodium current '''
        return cls.gNabar * m**3 * h * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iKd(cls, n, Vm):
        ''' delayed-rectifier Potassium current '''
        return cls.gKdbar * n**4 * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, x: cls.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: cls.iKd(x['n'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }

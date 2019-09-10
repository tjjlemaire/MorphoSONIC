# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-09-03 15:52:49
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-10 17:29:16

import numpy as np
import matplotlib.pyplot as plt

from ExSONIC.utils import chronaxie, fitTauSD
from PySONIC.utils import rsquared

# S/D data
durations = np.array([10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500], dtype=float) * 1e-6  # s
currents = np.array([4.4, 2.8, 1.95, 1.6, 1.45, 1.35, 1.25, 1.2, 1.2, 1.2, 1.2, 1.2]) * 1e-9  # A
charges = currents * durations  # C

# Chronaxie and S/D time constant
I0 = min(currents)  # A
Q0 = min(charges)  # C
ch = chronaxie(durations, currents)  # s
tau_e = Q0 / I0  # s
print(f'chronaxie = {ch * 1e6:.1f} us')
print(f'SD time constant: tau_e = {tau_e * 1e6:.1f} us')

# Weiss SD fit
Weiss_taue, Weiss_Ithrs = fitTauSD(durations, currents, method='Weiss')
print(f'Weiss tau_e = {Weiss_taue * 1e6:.1f} us')
r2_Weiss = rsquared(currents, Weiss_Ithrs)

# Lapique SD fit
Lapique_taue, Lapique_Ithrs = fitTauSD(durations, currents, method='Lapique')
print(f'Lapique tau_e = {Lapique_taue * 1e6:.1f} us')
r2_Lapique = rsquared(currents, Lapique_Ithrs)

# Plot
fig, ax = plt.subplots()
ax.set_title('S/D curve fits')
ax.set_xlabel('duration (us)')
ax.set_ylabel('current (nA)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(0.9* I0 * 1e9, 1.1 * currents.max() * 1e9)
ax.plot(durations * 1e6, currents * 1e9, '.', color='k', label='S/D data')
ax.plot(durations * 1e6, Q0 / durations * 1e9, '--', color='C2', label='Q0 / t')
ax.plot(durations * 1e6, Weiss_Ithrs * 1e9, color='C0', label=f'Weiss fit (R2 = {r2_Weiss:.3f})')
ax.plot(durations * 1e6, Lapique_Ithrs * 1e9, color='C1', label=f'Lapique fit (R2 = {r2_Lapique:.3f})')
ax.axvline(ch * 1e6, linestyle='-.', color='k',  label='chronaxie')
ax.axvline(tau_e * 1e6, linestyle='-.', color='C2',  label='S/D time constant')
ax.axvline(Weiss_taue * 1e6, linestyle='-.', color='C0',  label='Weiss fit')
ax.axvline(Lapique_taue * 1e6, linestyle='-.', color='C1', label='Lapique fit')
ax.legend()

plt.show()
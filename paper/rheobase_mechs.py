# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-08-27 11:33:16
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-09-21 17:06:21

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.utils import logger, si_format, rescale, padleft
from PySONIC.neurons import getPointNeuron
from PySONIC.constants import NPC_DENSE
from PySONIC.core import BilayerSonophore, AcousticDrive, PulsedProtocol, PmCompMethod
from ExSONIC.core import Node, StrengthDurationBatch
from ExSONIC.constants import *

from utils import setAxis, figdir, dataroot, fontsize

logger.setLevel(logging.INFO)


def energy(y, dt):
    ''' Compute signal energy. '''
    return np.sum(y**2) * dt


# Durations and offset
durations = np.logspace(-5, 0, 20)  # s
toffset = 10e-3  # s

# Default US parameters
Fdrive = 500e3  # Hz
a = 32e-9       # m
fs = 0.8        # (-)
namps = 20
nplts = 4
iplts = np.round(np.linspace(0, namps - 1, nplts)).astype(int)

# Nodes
mechs = {'unmyelinated': 'SUseg', 'myelinated': 'FHnode'}
pneurons = {k: getPointNeuron(v) for k, v in mechs.items()}
nodes = {k: Node(v, a=a, fs=fs) for k, v in pneurons.items()}

# Plot parameters
ypad = -10

# SD curves
drive = AcousticDrive(Fdrive)
drive.key = 'A'
Athrs = {}
for k, node in nodes.items():
    sd_batch = StrengthDurationBatch('A (Pa)', drive, node, durations, toffset, root=dataroot)
    Athrs[k] = sd_batch.run()

# Extract rheobase threshold amplitudes
Arheo = {k: np.nanmin(v) for k, v in Athrs.items()}  # Pa

# Determine sub-threshold amplitude ranges
Asubthr = {k: 0.99 * v for k, v in Arheo.items()}
Amin = 1e3  # Pa
subthr_amps = {k: np.logspace(np.log10(Amin), np.log10(v), namps) for k, v in Asubthr.items()}  # Pa

# Durations
durations = {k: 50 * node.pneuron.tau_pas for k, node in nodes.items()}

# Simulate models for sub-threshold amplitude ranges
tvecs, Qvecs = {}, {}
for k, node in nodes.items():
    tvecs[k], Qvecs[k] = [], []
    pp = PulsedProtocol(durations[k], 0.)
    for i, Adrive in enumerate(subthr_amps[k]):
        drive = AcousticDrive(Fdrive, Adrive)
        data, meta = node.simulate(drive, pp)
        tvecs[k].append(data['t'].values)   # s
        Qvecs[k].append(data['Qm'].values)  # C/m2


# Compute steady-states and time constants of charge exponential convergence
Qss, tauQ = {}, {}
for k in nodes.keys():
    Qss[k] = np.array([Q[-1] for Q in Qvecs[k]])
    tauQ[k] = np.array([np.interp(0.63, rescale(Q), t) for t, Q in zip(tvecs[k], Qvecs[k])])
    Cm0 = nodes[k].pneuron.Cm0                     # F/m2
    Qm0 = nodes[k].pneuron.Qm0                     # C/cm2

# Compute sub-threshold steady-state effective capacitance and relative variations
Cmeff_ss, rel_Cmeff_drop, norm_dQss_approx = {}, {}, {}
for k in nodes.keys():
    Cmeff_ss[k] = np.array([Q / nodes[k].pylkp.projectN({'A': A, 'Q': Q})['V'] * V_TO_MV  # F/m2
                            for A, Q in zip(subthr_amps[k], Qss[k])])
    Cm0 = nodes[k].pneuron.Cm0                       # F/m2
    rel_Cmeff_drop[k] = (Cmeff_ss[k] - Cm0) / Cm0    # (-)

# Compute true and leakage-approximated normalized charge build-ups
norm_dQss = {'true': {}, 'approx': {}}
for k in nodes.keys():
    Cm0 = nodes[k].pneuron.Cm0                             # F/m2
    Qm0 = nodes[k].pneuron.Qm0                             # C/cm2
    ELeak = nodes[k].pneuron.ELeak                         # mV
    norm_dQss['true'][k] = (Qss[k] - Qm0) / Cm0 * V_TO_MV  # mV
    norm_dQss['approx'][k] = rel_Cmeff_drop[k] * ELeak     # mV

# Compute intra-cycle deflection and pressure profiles at threshold amplitudes
plabels = [
    'hydrostatic',
    'gaseous',
    'electrical',
    'molecular',
    'elastic',
    'viscous',
]
deflections, velocities, capcts, pressures, expansion_intervals = {}, {}, {}, {}, {}
Pmol_range, Pelec_range, deltas = {}, {}, {}

tcycle = np.linspace(0, 1 / Fdrive, NPC_DENSE)           # s
dt = tcycle[1] - tcycle[0]                               # s
Z_range = np.linspace(-BilayerSonophore.Delta_, 20e-9, 1000)  # m

Pm_met = PmCompMethod.direct
# Pm_met = PmCompMethod.predict

# For each node
for k, node in nodes.items():

    # Define corresponding bvilayer sonophore model
    bls = BilayerSonophore(a, node.pneuron.Cm0, node.pneuron.Qm0)
    deltas[k] = bls.Delta
    A, Qm = subthr_amps[k][-1], Qss[k][-1]

    # Compute range variables
    Pelec_range[k] = bls.Pelec(Z_range, Qm)
    if Pm_met == PmCompMethod.direct:
        R_range = bls.v_curvrad(Z_range)       # m
        S_range = bls.surface(Z_range)         # m2
        Pmol_range[k] = bls.v_PMavg(Z_range, R_range, S_range)
    else:
        Pmol_range[k] = bls.PMavgpred(Z_range)

    # Run simulation and extract deflection and gas profiles
    logger.info(f'{node}: getting cyclic pressure profiles at {A * PA_TO_KPA:.1f} kPa')
    drive = AcousticDrive(Fdrive, A)
    data = bls.simCycles(drive, Qm, Pm_comp_method=Pm_met).tail(NPC_DENSE)
    Z, ng = data['Z'].values, data['ng'].values

    # Compute cyclic profiles
    Cm = bls.v_capacitance(Z)     # F/m2
    U = padleft(np.diff(Z)) / dt  # m/s
    R = bls.v_curvrad(Z)          # m
    S = bls.surface(Z)            # m2
    pdict = {
        'hydrostatic': -np.ones(tcycle.size) * bls.P0,
        'gaseous': bls.gasmol2Pa(ng, bls.volume(Z)),
        'electrical': bls.Pelec(Z, Qm),
        'molecular': bls.v_PMavg(Z, R, S) if Pm_met == PmCompMethod.direct else bls.PMavgpred(Z),
        'elastic': bls.PEtot(Z, R),
        'viscous': bls.PVleaflet(U, R) + bls.PVfluid(U, R),
    }

    # Fill in dicts
    deflections[k] = Z
    velocities[k] = U
    capcts[k] = Cm
    pressures[k] = pdict

print(deltas)

# Compute energies from pressure profiles
energies = {}
for k, pdict in pressures.items():
    energies[k] = {}
    for pkey, pvec in pdict.items():
        energies[k][pkey] = energy(pvec, dt)


# Figure
fig = plt.figure(constrained_layout=True, figsize=(8, 6))
fig.canvas.set_window_title('rheobase_mechs')
gs = fig.add_gridspec(4, 5)
subplots = {
    'a': gs[:2, 0],
    'b': gs[:2, 1],
    'c': gs[:2, 2:4],
    'd': gs[0, 4],
    'e': gs[1, 4],
    'f': gs[2:4, 0],
    'g': gs[2:4, 1],
    'h': gs[2:4, 2:4],
    'i': gs[2, 4],
    'j': gs[3, 4],
}
axes = {k: fig.add_subplot(v) for k, v in subplots.items()}

# first row: charge build-ups
colors = plt.get_cmap('tab20c').colors[:8]
cdict = dict(zip(nodes.keys(), [colors[:4][::-1], colors[4:][::-1]]))
buildup_axes = [axes['a'], axes['b']]
for k, ax in zip(nodes.keys(), buildup_axes):
    ax.set_title(k, fontsize=fontsize)
    trep = np.power(10, np.floor(np.log10(durations[k])))
    ax.set_ylabel('Qm (nC/cm2)', fontsize=fontsize, labelpad=ypad)
    ax.set_xticks([])
    s = ax.spines['bottom']
    s.set_bounds(0, trep * S_TO_MS)
    s.set_position(('outward', 3))
    s.set_linewidth(3.0)
    ax.set_xlabel(f'{si_format(trep)}s', fontsize=fontsize)
    Qm0 = nodes[k].pneuron.Qm0
    ax.axhline(Qm0 * C_M2_TO_NC_CM2, c='k', linestyle='--')
    for i, c in zip(iplts, cdict[k]):
        t, Q = tvecs[k][i], Qvecs[k][i]
        ax.plot(t * S_TO_MS, Q * C_M2_TO_NC_CM2, c=c,
                label=f'{subthr_amps[k][i] * PA_TO_KPA:.0f} kPa')
        ax.axhline(Q[-1] * C_M2_TO_NC_CM2, c=c, linestyle='--')
    setAxis(ax, 0, True)
    tarrow = t[-1] * S_TO_MS
    Qarrow = [Qm0 * C_M2_TO_NC_CM2, Q[-1] * C_M2_TO_NC_CM2]
    xy1, xy2 = list(zip([tarrow] * 2, Qarrow))
    ax.annotate('', xy=xy1, xytext=xy2, arrowprops=dict(facecolor='k', arrowstyle='<|-|>'))

    xy_label = (tarrow, np.mean(Qarrow))
    ax_xy_label = ax.transAxes.inverted().transform(ax.transData.transform(xy_label))
    ax_xy_label = (ax_xy_label[0] - 0.05, ax_xy_label[1])
    ax.text(*ax_xy_label, '(DQm)inf', rotation='vertical', transform=ax.transAxes,
            horizontalalignment='center', verticalalignment='center')
    ax.legend(frameon=False, fontsize=fontsize, loc='center')

# normalized steady-state charge build-ups
ax = axes['c']
colors = list(plt.get_cmap('tab20').colors)
colors = dict(zip(nodes.keys(), [colors[:2], colors[2:]]))
linestyles = dict(zip(norm_dQss.keys(), ['-', '--']))
ax.set_xscale('log')
ax.set_xlim(Amin * PA_TO_KPA, 1e2)
ax.get_xaxis().get_minor_formatter().labelOnlyBase = True
ax.set_xlabel('amplitude (kPa)', fontsize=fontsize)
ax.set_ylabel('DQmss / Cm0 (mV)', fontsize=fontsize, labelpad=ypad)
for i, (mkey, norm_dQss_dict) in enumerate(norm_dQss.items()):
    ls = linestyles[mkey]
    for (k, amps) in subthr_amps.items():
        c = colors[k][i]
        ax.plot(amps * PA_TO_KPA, norm_dQss_dict[k], ls, c=c, label=f'{mkey} - {k}', linewidth=2)
        ax.scatter(Arheo[k] * PA_TO_KPA, norm_dQss_dict[k][-1], c=[c, ], zorder=2.5)
        if i == 0:
            ax.axvline(Arheo[k] * PA_TO_KPA, c=c, linestyle='--')
ax.legend(frameon=False, fontsize=fontsize)
setAxis(ax, 0, False)

cyclic_axes = [axes['d'], axes['e']]
colors = plt.get_cmap('tab10').colors

# Normalized capacitance profile at threshold
ax = axes['d']
ax.set_ylabel('Cm / Cm0', labelpad=ypad)
ax.axhline(1.0, linewidth=0.5, c='k')
ax.legend(frameon=False, fontsize=fontsize)
for c, (k, Cm) in zip(colors, capcts.items()):
    Cm0 = nodes[k].pneuron.Cm0
    Cmeff = 1 / np.mean(1 / Cm)
    print(f'{k}: DCm/Cm0 = {(Cm0 - Cmeff) / Cm0 * 1e2:.1f} %')
    ax.plot(tcycle * 1e6, Cm / Cm0, label=k, c=c)
    ax.axhline(Cmeff / Cm0, linestyle='--', c=c)
setAxis(ax, 1, True)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])

# Deflection profile at threshold
ax = axes['e']
ax.set_xlabel('time (us)')
ax.set_ylabel('Z (nm)', labelpad=ypad)
ax.axhline(0.0, linewidth=0.5, c='k')
for c, (k, Z) in zip(colors, deflections.items()):
    ax.plot(tcycle * 1e6, Z * M_TO_NM, label=k, c=c)
setAxis(ax, 1, True)

# Pressure profiles at threshold
colors = list(plt.get_cmap('Dark2').colors)[:len(plabels)]
pressure_axes = [axes['f'], axes['g']]
for ax, (k, pdict) in zip(pressure_axes, pressures.items()):
    ax.set_title(k, fontsize=fontsize)
    ax.set_ylabel('pressures (kPa)', fontsize=fontsize, labelpad=ypad)
    for (pkey, pvec), c in zip(pdict.items(), colors):
        ax.plot(tcycle * S_TO_US, pvec * PA_TO_KPA, c=c, linewidth=2, label=pkey)
    ax.axhline(0., c='k', linewidth=0.5)
ymin = min([ax.get_ylim()[0] for ax in pressure_axes])
ymax = max([ax.get_ylim()[1] for ax in pressure_axes])
for ax in pressure_axes:
    ax.set_ylim(ymin, ymax)
    setAxis(ax, 0, True)
pressure_axes[0].legend(frameon=False, fontsize=fontsize)

# Pressure energies at threshold
ax = axes['h']
ax.spines['bottom'].set_visible(False)
ax.tick_params(length=0, axis='x')
x = np.arange(len(plabels))
ax.set_xticks(x)
ax.set_xticklabels(plabels, rotation=30)
ax.set_ylabel('pressure energies (Pa2.ms)', fontsize=fontsize, labelpad=ypad)
ax.set_ylim(0, 30)
width = 0.35
offsets = dict(zip(energies.keys(), [- width / 2, + width / 2]))
bars = {}
for k, edict in energies.items():
    y = np.array(list(edict.values()))
    bars[k] = ax.bar(x + offsets[k], y / S_TO_MS, width, label=k,
                     color=colors, edgecolor='black')
for bar in bars['myelinated']:
    bar.set_hatch('//')
setAxis(ax, 0, False)

p_vs_Z_axes = [axes['i'], axes['j']]
colors = plt.get_cmap('tab10').colors
xlims = [-1, Z_range.max() * M_TO_NM]

# Electrical pressure over deflection range
ax = axes['i']
ax.set_ylabel('PQ (kPa)', fontsize=fontsize, labelpad=ypad)
ax.set_ylim(-100, 0)
for c, (k, Pelec) in zip(colors, Pelec_range.items()):
    ax.plot(Z_range * M_TO_NM, Pelec * PA_TO_KPA, label=k, c=c)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])


Qthrs = {k: v[-1] for k, v in Qss.items()}
Qratio = Qthrs['myelinated'] / Qthrs['unmyelinated']
print(np.mean(Qratio), np.std(Qratio))
print(np.mean(Qratio**2), np.std(Qratio**2))
Pelec_ratio = Pelec_range['myelinated'] / Pelec_range['unmyelinated']
print(np.mean(Pelec_ratio), np.std(Pelec_ratio))

# Molecular pressure over deflection range
ax = axes['j']
ax.set_ylabel('PM (kPa)', fontsize=fontsize, labelpad=ypad)
ax.set_ylim(-11, 11)
ax.axhline(0, linewidth=1, c='k')
ax.set_xlabel('Z (nm)', fontsize=fontsize, labelpad=ypad)
for c, (k, Pmol) in zip(colors, Pmol_range.items()):
    ax.plot(Z_range * M_TO_NM, Pmol * PA_TO_KPA, label=k, c=c)
ax.set_xticks(xlims)

for ax in p_vs_Z_axes:
    ax.axvline(0, linewidth=0.5, c='k')
    setAxis(ax, 0, False)
    ax.set_xlim(*xlims)

# Set t-scales
Tdrive = 1 / Fdrive  # s
for ax in [cyclic_axes[-1]] + pressure_axes:
    ax.set_xticks([])
    s = ax.spines['bottom']
    s.set_bounds(0, Tdrive * S_TO_US)
    s.set_position(('outward', 3))
    s.set_linewidth(3.0)
    ax.set_xlabel(f'TUS ({si_format(Tdrive)}s)', fontsize=fontsize)
    # for tstart, tend, _ in expansion_intervals:
    #     ax.axvspan(tstart * S_TO_US, tend * S_TO_US,
    #                facecolor='silver', edgecolor='none', alpha=0.2)

# Post-processing
for ax in axes.values():
    for sk in ['right', 'top']:
        ax.spines[sk].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fontsize)

fig.savefig(os.path.join(figdir, 'rheobase_mechs_raw.pdf'), transparent=True)

plt.show()

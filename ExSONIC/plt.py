# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-26 17:11:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-09-21 17:26:27

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from PySONIC.plt import GroupedTimeSeries, CompTimeSeries, mirrorAxis
from PySONIC.utils import logger, si_format, getPow10, rsquared, padleft, timeThreshold

from .core import *
from .utils import loadData, chronaxie
from .constants import *


class SectionGroupedTimeSeries(GroupedTimeSeries):
    ''' Plot the time evolution of grouped variables in a specific section. '''

    def __init__(self, section_id, outputs, pltscheme=None):
        ''' Constructor. '''
        self.section_id = section_id
        super().__init__(outputs, pltscheme=pltscheme)

    @staticmethod
    def getModel(meta):
        return getModel(meta)

    def figtitle(self, model, meta):
        return super().figtitle(model, meta) + f' - {self.section_id} section'

    def getData(self, entry, frequency=1, trange=None):
        if entry is None:
            raise ValueError('non-existing data')
        if isinstance(entry, str):
            data, meta = loadData(entry, frequency)
        else:
            data, meta = entry
        data = data[self.section_id]
        data = data.iloc[::frequency]
        if trange is not None:
            tmin, tmax = trange
            data = data.loc[(data['t'] >= tmin) & (data['t'] <= tmax)]
        return data, meta

    def render(self, *args, **kwargs):
        figs = super().render(*args, **kwargs)
        for fig in figs:
            title = fig.canvas.get_window_title()
            fig.canvas.set_window_title(title + f'_{self.section_id}')


class SectionCompTimeSeries(CompTimeSeries):
    ''' Plot the time evolution of a specific variable across sections, for a specific condition '''

    def __init__(self, filepath, varname, sections):
        self.entry = filepath[0]
        self.model = None
        self.ref_meta = None
        nsec = len(sections)
        if nsec > NTRACES_MAX:
            factor = int(np.ceil(nsec / NTRACES_MAX))
            sections = sections[::factor]
            logger.warning(f'Displaying only {len(sections)} traces out of {nsec}')
        super().__init__(sections, varname)

    def getModel(self, meta):
        if self.model is None:
            self.ref_meta = meta.copy()
            del self.ref_meta['section']
            self.model = getModel(meta)
        else:
            comp_meta = meta.copy()
            del comp_meta['section']
            if comp_meta != self.ref_meta:
                return getModel(meta)
        return self.model

    def getData(self, section, frequency=1, trange=None):
        if self.entry is None:
            raise ValueError('non-existing data')
        if isinstance(self.entry, str):
            data, meta = loadData(self.entry, frequency)
        else:
            data, meta = self.entry
        meta = meta.copy()
        meta['section'] = section
        data = data[section]
        data = data.iloc[::frequency]
        if trange is not None:
            tmin, tmax = trange
            data = data.loc[(data['t'] >= tmin) & (data['t'] <= tmax)]
        return data, meta

    def render(self, *args, cmap='sym_viridis_r', **kwargs):
        return super().render(*args, cmap=cmap, **kwargs)


def thresholdCurve(fiber, x, thrs, thrs2=None,
                   xname='duration', xfactor=S_TO_US, xunit='s',
                   yname='current', yfactor=1, yunit='A',
                   y2name='charge', y2factor=1, y2unit='C',
                   scale='log', plot_chr=True, fs=12, colors=None, limits=None, xlimits=None):
    if colors is None:
        colors = plt.get_cmap('tab10').colors

    fig, ax = plt.subplots()
    prefix = si_format(1 / yfactor, space='')[1:]
    ax.set_title(f'{fiber}', fontsize=fs)
    ax.set_xlabel(f'{xname} ({si_format(1 / xfactor, space="")[1:]}{xunit})', fontsize=fs)
    ax.set_ylabel(f'threshold {yname} ({prefix}{yunit})', fontsize=fs)
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
    testvalues = thrs[list(thrs.keys())[0]]
    testvalues = testvalues[np.logical_not(np.isnan(testvalues))]
    if np.all(testvalues < 0.):
        thrs = {k: -v for k, v in thrs.items()}
        if thrs2 is not None:
            thrs2 = {k: -v for k, v in thrs2.items()}
    to_add = []
    for i, k in enumerate(thrs.keys()):
        ax.plot(x * xfactor, thrs[k] * yfactor, label=k, color=colors[i])
        if any(np.isnan(thrs[k])):
            ilastnan = np.where(np.isnan(thrs[k]))[0][-1]
            to_add.append((x[ilastnan:ilastnan + 2], thrs[k][ilastnan + 1], colors[i]))
        if plot_chr:
            ax.axvline(chronaxie(x, thrs[k]) * xfactor, linestyle='-.', color=colors[i])
    if scale != 'log':
        if xlimits is None:
            ax.set_xlim(0., x.max() * xfactor)
            ax.set_ylim(0., ax.get_ylim()[1])
        else:
            ax.set_xlim(xlimits[0] * xfactor, xlimits[1] * xfactor)
    else:
        ax.set_xlim(x.min() * xfactor, x.max() * xfactor)
        if limits is None:
            ymin = np.nanmin([np.nanmin(v) for v in thrs.values()])
            ymax = np.nanmax([np.nanmax(v) for v in thrs.values()])
            ymin = getPow10(ymin * yfactor, 'down')
            ymax = getPow10(ymax * yfactor, 'up')
        else:
            ymin = limits[0] * yfactor
            ymax = limits[1] * yfactor
        ax.set_ylim(ymin, ymax)
    for xx, yy, cc in to_add:
        ax.plot(xx * xfactor, [ax.get_ylim()[1], yy * yfactor], '--', color=cc)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.legend(fontsize=fs / 1.8, frameon=False, loc='upper left', ncol=2)

    if thrs2 is not None:
        ax2 = ax.twinx()
        prefix = si_format(1 / y2factor, space='')[1:]
        ax2.set_ylabel(f'threshold {y2name} ({prefix}{y2unit})', fontsize=fs)
        if scale == 'log':
            ax2.set_yscale('log')
        for i, k in enumerate(thrs2.keys()):
            ax2.plot(x * xfactor, thrs2[k] * y2factor, linestyle='--', color=colors[i])
        if scale != 'log':
            ax2.set_ylim(0., ax2.get_ylim()[1])
        else:
            ymin2 = min([np.nanmin(v) for v in thrs2.values()])
            ymax2 = max([np.nanmax(v) for v in thrs2.values()])
            ymin2 = getPow10(ymin2 * y2factor, 'down')
            ymax2 = getPow10(ymax2 * y2factor, 'up')
            ax2.set_ylim(ymin2, ymax2)
    return fig


def strengthDurationCurve(fiber, durations, thrs, **kwargs):
    return thresholdCurve(fiber, durations, thrs, xname='duration',
                          xfactor=S_TO_US, xunit='s', **kwargs)


def strengthDistanceCurve(fiber, distances, thrs, **kwargs):
    return thresholdCurve(fiber, distances, thrs, xname='distance',
                          xfactor=M_TO_MM, xunit='m', plot_chr=False, **kwargs)


def plotConvergenceResults(df, inkey, outkeys, rel_eps_thr_Ithr=0.05, rel_eps_thr=0.01,
                           axesdirection='d'):
    ''' Plot output metrics of convergence study.

        :param df: dataframe with input values (parameter of interest) and output metrics
        :param inkey: key of the input parameter
        :param outkeys: keys of the output parameters
        :param direction: direction of the x axes used also to find the threshold
            ('a' ascending, 'd' descending)
        :param rel_eps_thr: relative error threshold for the output metrics
        :return: figure handle
    '''
    # Initialize dictionaries
    eps = {}      # relative errors of each output metrics
    xin_thr = {}  # threshold input values according to each output metrics

    # Extract input range and figure out if it must be reversed
    xin = df[inkey].values
    # reverse = xin[-1] < xin[0]

    # Create figure backbone
    fig, axes = plt.subplots(len(outkeys) + 1, 1, figsize=(6, 9))
    ax = axes[-1]
    ax.set_xlabel(inkey)
    ax.set_ylabel('relative errors (%)')
    ax.axhline(rel_eps_thr * 100, linestyle=':', color='k',
               label=f'{rel_eps_thr * 1e2:.1f} % threshold')
    ax.axhline(rel_eps_thr_Ithr * 100, linestyle='-.', color='k',
               label=f'{rel_eps_thr_Ithr * 1e2:.1f} % threshold')

    # For each output
    for i, k in enumerate(outkeys):
        xout = df[k].values

        # Plot output evolution
        axes[i].set_ylabel(k)
        axes[i].plot(xin, xout, c='k')
        ymin, ymax, yconv = np.nanmin(xout), np.nanmax(xout), xout[-1]
        yptp, ydelta = ymax - ymin, 0.8 * yconv
        if ymax - yconv > yconv - ymin:
            ytopaxis = min(yconv + ydelta, ymax + 0.05 * yptp)
            axes[i].set_ylim(
                ymin - 0.08 * (ytopaxis - ymin), ytopaxis)
        else:
            ybottomaxis = max(yconv - ydelta, ymin - 0.05 * yptp)
            axes[i].set_ylim(
                ybottomaxis, ymax + 0.08 * (ymax - ybottomaxis))

        # Compute and plot relative error w.r.t. reference (last) value
        xref = xout[-1]
        eps[k] = np.abs((xout - xref) / xref)
        axes[-1].plot(xin, eps[k] * 100, label=k, c=f'C{i}')

        # Compute and plot input value yielding threshold relative error
        j = eps[k].size - 1
        if i == 0:
            rel_thr = rel_eps_thr_Ithr
        else:
            rel_thr = rel_eps_thr
        while eps[k][j] <= rel_thr and j > 0:
            j -= 1
        xin_thr[k] = xin[j + 1]
        axes[-1].axvline(xin_thr[k], linestyle='dashed', color=f'C{i}')

    # Compute minimal required input value to satisfy all relative error threshold on all inputs
    # logger.info(f'Relative error threshold Ithr = {rel_eps_thr_Ithr * 1e2:.1f} %')
    # logger.info(f'Relative error threshold for CV and dV = {rel_eps_thr * 1e2:.1f} %')
    if axesdirection == 'd':
        logger.info(f'max {inkey} = {min(xin_thr.values()):.2e}')
    else:
        logger.info(f'To reach convergence {inkey} = {max(xin_thr.values()):.2e}')
    logger.info(f'Convergence excitation current threshold = {(df.values[-1,2] * A_TO_NA):.2f} nA')
    logger.info(f'Convergence conduction velocity = {df.values[-1,3]:.2f} m/s')
    logger.info(f'Convergence spike amplitude = {df.values[-1,4]:.2f} mV')

    # Post-process figure
    axes[-1].set_ylim(-5, 30)
    axes[-1].legend(frameon=False)
    for ax in axes:
        ax.set_xscale('log')
        if axesdirection == 'd':
            ax.invert_xaxis()
    fig.tight_layout()

    return fig


def plotFiberXCoords(fiber, fs=12):
    ''' Plot the x coordinates of a fiber model, per section type. '''
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_title(f'{fiber} - x-coordinates per section type', fontsize=fs)
    ax.set_xlabel('section mid-point x-coordinate (mm)', fontsize=fs)
    ax.set_ylabel('section type', fontsize=fs)
    ax.set_yticks(range(len(fiber.sectypes)))
    ax.set_yticklabels(fiber.sectypes)
    for i, (k, xcoords) in enumerate(fiber.getXCoords().items()):
        ax.plot(xcoords * M_TO_MM, np.ones(xcoords.size) * i, '|', markersize=15, label=k)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    fig.tight_layout()
    return fig


def plotFieldDistribution(fiber, source, fs=12):
    ''' Plot a source's field distribution over a fiber, per section type. '''
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_title(f'{fiber} - field distribution from {source}', fontsize=fs)
    ax.set_xlabel('section mid-point x-coordinate (mm)', fontsize=fs)
    if isinstance(source, (AcousticSource)):
        ylbl = 'Acoustic amplitude (kPa)'
        yfactor = PA_TO_KPA
    else:
        ylbl = 'Extracellular voltage (mV)'
        yfactor = 1e0
    ax.set_ylabel(ylbl, fontsize=fs)
    field_dict = source.computeDistributedAmps(fiber)
    xcoords = fiber.getXCoords()
    ndists = len(list(xcoords.keys()))
    colors = plt.get_cmap('tab10').colors[:ndists] if ndists > 1 else ['k']
    for c, (k, xcoords) in zip(colors, xcoords.items()):
        ax.plot(xcoords * M_TO_MM, field_dict[k] * yfactor, '.', label=k, c=c)
    ylims = ax.get_ylim()
    xvar = source.xvar
    if (isinstance(xvar, float) and xvar < 0.) or (isinstance(xvar, np.ndarray) and any(xvar < 0.)):
        ax.set_ylim(ylims[0], -0.05 * ylims[0])
    else:
        ax.set_ylim(-0.05 * ylims[1], ylims[1])
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    if ndists > 1:
        ax.legend(fontsize=fs, frameon=False)
    fig.tight_layout()
    return fig


def plotMRGLookups(fiberD_range=None, interp_methods=None, fs=12):
    ''' Plot MRG morphological parameters interpolated over a fiber diameter range. '''

    # Define diameters ranges
    ref_diams = mrg_lkp.refs['fiberD']
    if fiberD_range is None:
        fiberD_range = bounds(ref_diams)
    diams = np.linspace(*fiberD_range, 100)

    # Define interpolation methods
    if interp_methods is None:
        interp_methods = mrg_lkp.interp_choices

    # Define factor function
    factor = lambda k: 1 if k == 'nlayers' else M_TO_UM

    # Create figure backbone
    nouts = len(mrg_lkp.outputs)
    fig, axes = plt.subplots(1, nouts, figsize=(nouts * 3, 2.5))
    for ax, k in zip(axes, mrg_lkp.keys()):
        yunit = '' if k == 'nlayers' else '(um)'
        ax.set_xlabel('fiber diameter (um)', fontsize=fs)
        ax.set_ylabel(f'{k} {yunit}', fontsize=fs)
        ax.plot(ref_diams * M_TO_UM, mrg_lkp[k] * factor(k), '.', c='k')
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)

    # Interpolate over fiber range with each method and plot resulting profiles
    default_method = mrg_lkp.interp_method
    for interp_method in interp_methods:
        mrg_lkp.interp_method = interp_method
        interp_mrg_lkp = mrg_lkp.project('fiberD', diams)
        label = f'{interp_method} method'
        for ax, (k, v) in zip(axes, interp_mrg_lkp.items()):
            ax.plot(diams * M_TO_UM, v * factor(k), label=label)

    # Set lookup interpolation method back to default
    mrg_lkp.interp_method = default_method

    axes[0].legend(frameon=False)
    title = fig.suptitle(f'MRG morphological parameters', fontsize=fs)
    fig.tight_layout()
    title.set_y(title._y + 0.03)
    return fig


def plotFiberDiameterDistributions(n=50, fs=12):
    ''' Plot the diameter distribution of different types of peripheral fibers. '''
    fibers_dict = {
        'Aα': {
            'bounds': (13, 20),
            'myelinated': True,
            'implemented': True,
            'label': 'myelinated'
        },
        'Aβ': {
            'bounds': (6, 12),
            'myelinated': True,
            'implemented': True
        },
        'Aδ': {
            'bounds': (1, 5),
            'myelinated': True,
            'implemented': False
        },
        'C': {
            'bounds': (0.2, 1.5),
            'myelinated': False,
            'implemented': True,
            'label': 'unmyelinated'
        }
    }

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.set_yticks([])
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('diameter (um)', fontsize=fs)
    for item in ax.get_xticklabels():
        item.set_fontsize(fs)
    for key in ['top', 'left', 'right']:
        ax.spines[key].set_visible(False)
    g = signal.gaussian(n, std=8)
    for k, d in fibers_dict.items():
        drange = np.linspace(*d['bounds'], n)
        color = 'royalblue' if d['myelinated'] else 'orangered'
        label = d.get('label', None)
        ax.plot(drange, g, color, linewidth=2.5, label=label)
        ax.text(np.mean(d['bounds']), 1.07, k, color=color, size=fs + 2, weight='bold',
                horizontalalignment='center')
        if d['implemented']:
            ax.fill_between(drange, 0, g, color=color, alpha=0.5)
    ax.legend(fontsize=fs, frameon=False, bbox_to_anchor=(.9, 1), loc='upper left')
    fig.tight_layout()

    return fig


def plotCVvsDiameter(diams, cv_dict, fs=14):
    ''' Plot conduction velocity of various fiber models as a function of fiber diameter
        along with linear fits.
    '''
    fig, ax = plt.subplots()
    ax.set_xlabel('diameter (um)', fontsize=fs)
    ax.set_ylabel('conduction velocity (m/s)', fontsize=fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    for icolor, (k, cv) in enumerate(cv_dict.items()):
        color = f'C{icolor}'
        ax.plot(diams * 1e6, cv, 'o-', c=color, label=f'{k} - data')
        a, b = np.polyfit(diams, cv, 1)
        cv_fit = np.poly1d((a, b))(diams)
        r2 = rsquared(cv, cv_fit)
        ax.plot(diams * 1e6, cv_fit, '--', c=color,
                label=f'{k} - linear fit: CV = {b:.1f} + {a * 1e-6:.1f}*D (R2 = {r2:.3f})')
    ax.legend(frameon=False)
    return fig


def plotTimeseries0Dvs1D(pneuron, a, cov, rs, deff, drive, pp, figsize=(8, 6), fs=12):

    # Simulate punctual SONIC model with specific membrane coverage
    punctual_model = Node(pneuron, a=a, fs=cov)
    punctual_data, _ = punctual_model.simulate(drive, pp)

    # Simulate extended SONIC model with specific membrane coverage
    ext_model = surroundedSonophore(pneuron, a, cov, rs, depth=deff)
    ext_data, _ = ext_model.simulate(SectionAcousticSource('center', drive.f, drive.A), pp)

    # Add onset to solutions
    tonset = -5e-3
    punctual_data = prependDataFrame(punctual_data, tonset=tonset)
    for k, df in ext_data.items():
        ext_data[k] = prependDataFrame(df, tonset=tonset)

    # Get stimulus patches
    t = punctual_data['t'].values  # s
    stimon = punctual_data['stimstate'].values
    pulse = CompTimeSeries.getStimPulses(t, stimon)[0]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for ax in axes:
        for key in ['right', 'top']:
            ax.spines[key].set_visible(False)
        ax.set_xlim(tonset * 1e3, (pp.tstop) * 1e3)
        # ax.set_ylim(-100, 50)
        # ax.set_yticks(ax.get_ylim())
        ax.axvspan(pulse[0] * 1e3, pulse[1] * 1e3, edgecolor='none', facecolor='#8A8A8A', alpha=0.2)
    ax = axes[0]
    ax.set_ylabel('$\\rm V_m^*\ (mV)$', fontsize=fs)
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax = axes[1]
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$\\rm Q_m\ (nC/cm^2)$', fontsize=fs)
    ax.set_xticks([0, (pp.tstop) * 1e3])
    ax.set_xticklabels(['{:.0f}'.format(x) for x in ax.get_xticks()])

    # Plot membrane potential and charge density profiles
    colors = plt.get_cmap('Paired').colors[:2]
    linestyles = ['-', '--']
    for i, (key, df) in enumerate(ext_data.items()):
        axes[0].plot(
            df['t'] * 1e3, df['Vm'], linestyles[i], c=colors[i], label=f'ext. model: {key}')
        axes[1].plot(df['t'] * 1e3, df['Qm'] * 1e5, linestyles[i], c=colors[i])
    axes[0].plot(t * 1e3, punctual_data['Vm'], c='dimgrey', label='punctual model')
    axes[1].plot(t * 1e3, punctual_data['Qm'] * 1e5, c='dimgrey')

    # Add legend
    axes[0].legend(
        frameon=False, fontsize=fs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=3, mode='expand', borderaxespad=0.)

    # Post-process figure
    for ax in axes:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)
    GroupedTimeSeries.shareX(axes)

    return fig


def mergeFigs(*figs, linestyles=None, alphas=None, inplace=False):
    ''' Merge the content of several figures in a single figure. '''
    if alphas is None:
        alphas = [1] * len(figs)
    if linestyles is None:
        linestyles = ['-'] * len(figs)
    new_fig, new_ax = plt.subplots(figsize=figs[0].get_size_inches())
    mirrorAxis(figs[0].axes[0], new_ax)
    for fig, ls, alpha in zip(figs, linestyles, alphas):
        for l in fig.axes[0].get_lines():
            new_ax.plot(l.get_data()[0], l.get_data()[1], ls, c=l.get_color(), alpha=alpha)
    if hasattr(figs[0], 'sm'):
        cbarax = new_fig.add_axes([0.85, 0.15, 0.03, 0.8])
        mirrorAxis(figs[0].axes[1], cbarax)
        nvalues = len(figs[0].axes[0].get_lines())
        comp_values = list(range(nvalues))
        cbar_kwargs = {}
        bounds = np.arange(nvalues + 1) + min(comp_values) - 0.5
        ticks = bounds[:-1] + 0.5
        if nvalues > 10:
            ticks = [ticks[0], ticks[-1]]
        cbar_kwargs.update({'ticks': ticks, 'boundaries': bounds, 'format': '%1i'})
        cbarax.tick_params(axis='both', which='both', length=0)
        new_fig.colorbar(figs[0].sm, cax=cbarax, **cbar_kwargs)
        cbarax.set_ylabel('node index')
    if inplace:
        for fig in figs:
            plt.close(fig)
    return new_fig


def plotPassiveCurrents(fiber, df):
    # Extract time and currents vectors
    t = df['t'].values
    currents = fiber.getCurrentsDict(df)
    inet = currents.pop('Net')

    # Find time interval required to reach threshold charge build-up
    dQnorm_thr = 5.  # mV
    tthr = timeThreshold(t, df['Qm'].values / fiber.pneuron.Cm0 * V_TO_MV, dQnorm_thr)

    # Plot currents temporal profiles
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.subplots_adjust(left=0.1, right=0.8, bottom=0.15, top=0.95, hspace=0.5)
    for sk in ['top', 'right']:
        ax.spines[sk].set_visible(False)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('currents (A/m2)')
    tonset = t.min() - 0.05 * np.ptp(t)
    tplt = np.insert(t, 0, tonset)
    for k, i in currents.items():
        ax.plot(tplt * S_TO_MS, padleft(i) * MA_TO_A, label=k)
    ax.plot(tplt * S_TO_MS, padleft(inet) * MA_TO_A, label='Net', c='k')
    ax.axvline(tthr * S_TO_MS, c='k', linestyle='--')
    ax.legend(frameon=False)
    pulse = GroupedTimeSeries.getStimPulses(t, df['stimstate'].values)[0]
    ax.axvspan(pulse[0] * S_TO_MS, pulse[1] * S_TO_MS,
               edgecolor='none', facecolor='#8A8A8A', alpha=0.2)
    if fiber.pneuron.name == 'FHnode':
        ylims = [-140, 50]
    else:
        ylims = [-0.9, 0.7]
    ax.set_ylim(*ylims)
    ax.set_yticks(ylims)

    # Plot charge accumulation bar chart
    buildup_charges_norm = fiber.getBuildupContributions(df, tthr)
    colors = plt.get_cmap('tab10').colors
    ax = fig.add_axes([0.85, 0.15, 0.13, 0.8])
    for sk in ['top', 'right']:
        ax.spines[sk].set_visible(False)
    x = np.arange(len(buildup_charges_norm))
    ax.set_xticks(x)
    # ax.set_yscale('symlog')
    ax.set_ylabel('Normalized sub-threshold charge accumulation (mV)')
    ax.set_xticklabels(list(buildup_charges_norm.keys()))
    ax.bar(x, list(buildup_charges_norm.values()), color=colors)
    ax.set_ylim(-1, dQnorm_thr)
    ax.set_yticks([-1, 0, dQnorm_thr])
    ax.axhline(0, c='k', linewidth=0.5)

    return fig


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
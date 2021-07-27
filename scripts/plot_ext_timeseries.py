# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 07:03:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:46:55

''' Plot section-specific temporal profiles of specific simulation output variables
    for spatially-extended models. '''

import matplotlib.pyplot as plt

from PySONIC.utils import logger, loadData
from MorphoSONIC.plt import SectionCompTimeSeries, SectionGroupedTimeSeries
from MorphoSONIC.parsers import SpatiallyExtendedTimeSeriesParser


def main():
    # Parse command line arguments
    parser = SpatiallyExtendedTimeSeriesParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Parse sections (looking into first file for ids if not provided)
    if args['section'] == ['all']:
        data, _ = loadData(args['inputfiles'][0])
        args['section'] = list(data.keys())

    # Plot appropriate graph
    if args['compare']:
        if args['plot'] == ['all'] or args['plot'] is None:
            logger.error('Specific variables must be specified for comparative plots')
            return
        for pltvar in args['plot']:
            try:
                comp_plot = SectionCompTimeSeries(args['inputfiles'], pltvar, args['section'])
                comp_plot.render(
                    patches=args['patches'],
                    spikes=args['spikes'],
                    frequency=args['sr'],
                    trange=args['trange'],
                    prettify=args['pretty'],
                    cmap=args['cmap'],
                    cscale=args['cscale']
                )
            except KeyError as e:
                logger.error(e)
                return
    else:
        for key in args['section']:
            scheme_plot = SectionGroupedTimeSeries(key, args['inputfiles'], pltscheme=args['pltscheme'])
            scheme_plot.render(
                patches=args['patches'],
                spikes=args['spikes'],
                frequency=args['sr'],
                trange=args['trange'],
                prettify=args['pretty'],
                save=args['save'],
                outputdir=args['outputdir'],
                fig_ext=args['figext']
            )

    if not args['hide']:
        plt.show()


if __name__ == '__main__':
    main()

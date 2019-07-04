# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-03-18 18:06:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-05 00:26:29

import os

from PySONIC.utils import logger
from PySONIC.parsers import Parser
from ExSONIC.core import NmodlTranslator


def main():
    parser = Parser()
    parser.addNeuron()
    parser.addSave()
    parser.addOutputDir(dep_key='save')
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    for pneuron in args['neuron']:
        logger.info('generating %s neuron MOD file', pneuron.name)
        translator = NmodlTranslator(pneuron.__class__)
        if args['save']:
            outfile = '{}auto.mod'.format(pneuron.name)
            outpath = os.path.join(args['outputdir'], outfile)
            logger.info('dumping MOD file in "%s"', args['outputdir'])
            translator.dump(outpath)
        else:
            translator.print()


if __name__ == '__main__':
    main()

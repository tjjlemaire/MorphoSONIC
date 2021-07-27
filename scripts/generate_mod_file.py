# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-03-18 18:06:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:46:05

import os

from PySONIC.utils import logger
from PySONIC.parsers import Parser
from MorphoSONIC.core import NmodlTranslator


def main():
    parser = Parser()
    parser.addNeuron()
    parser.addSave()
    parser.addOutputDir(dep_key='save')
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    for pneuron in args['neuron']:
        logger.info(f'generating {pneuron.name} neuron MOD file')
        translator = NmodlTranslator(pneuron.__class__)
        if args['save']:
            outfile = f'{pneuron.name}.mod'
            outpath = os.path.join(args['outputdir'], outfile)
            logger.info(f'dumping MOD file in "{args["outputdir"]}"')
            translator.dump(outpath)
        else:
            translator.print()


if __name__ == '__main__':
    main()

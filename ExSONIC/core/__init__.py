# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-27 18:16:12

from PySONIC.neurons import getPointNeuron

from .node import *
from .ext_sonic_node import *
from .senn import *
from .connectors import *
from .pyhoc import *
from .pymodl import *
from .psource import *
from .fibers import *


def getModel(meta):
    ''' Return appropriate model object based on a dictionary of meta-information. '''
    simkey = meta['simkey']
    pneuron = getPointNeuron(meta['neuron'])

    if simkey == 'nano_ext_SONIC':
        model = ExtendedSonicNode(pneuron, meta['rs'], a=meta['a'], Fdrive=meta['Fdrive'],
                                  fs=meta['fs'], deff=meta['deff'])
    elif simkey.startswith('senn'):
        senn_args = [meta[x] for x in ['nnodes', 'rs', 'nodeD', 'nodeL', 'interD', 'interL']]
        if simkey == 'senn_Iextra':
            model = IextraFiber(pneuron, *senn_args)
        elif simkey == 'senn_Iintra':
            model = IintraFiber(pneuron, *senn_args)
        elif simkey == 'senn_SONIC':
            model = SonicFiber(pneuron, *senn_args, a=meta['a'], Fdrive=meta['Fdrive'], fs=meta['fs'])
        else:
            raise ValueError(f'Unknown SENN model type:{simkey}')
    else:
        raise ValueError(f'Unknown model type:{simkey}')
    return model
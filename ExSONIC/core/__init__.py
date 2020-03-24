# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-03-10 11:29:42

import sys
import inspect

from .node import *
from .ext_sonic_node import *
from .senn import *
from .mrg import *
from .connectors import *
from .pyhoc import *
from .pymodl import *
from .grids import *
from .nmodel import *
from .sources import *
from .fibers import *
from .synapses import *
from .network import *


def getModelsDict():
    ''' Construct a dictionary of all model classes, indexed by simulation key. '''
    current_module = sys.modules[__name__]
    models_dict = {}
    for _, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and hasattr(obj, 'simkey') and isinstance(obj.simkey, str):
            models_dict[obj.simkey] = obj
    return models_dict


models_dict = getModelsDict()


def getModel(meta):
    ''' Return appropriate model object based on a dictionary of meta-information. '''
    simkey = meta['simkey']
    try:
        return models_dict[simkey].initFromMeta(meta['model'])
    except KeyError:
        raise ValueError(f'Unknown model type:{simkey}')

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-09-24 22:21:39

import sys
import inspect

from .cgi_network import *
from .sonic import *
from .node import *
from .radial_model import *
from .senn import *
from .mrg import *
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
            for k in ['original', 'benchmark']:
                full_k = f'__{k}__'
                if hasattr(obj, full_k):
                    models_dict[getattr(obj, full_k).simkey] = getattr(obj, full_k)
    return models_dict


models_dict = getModelsDict()


def getModel(meta):
    ''' Return appropriate model object based on a dictionary of meta-information. '''
    simkey = meta['simkey']
    try:
        return models_dict[simkey].initFromMeta(meta['model'])
    except KeyError:
        raise ValueError(f'Unknown model type: {simkey}')

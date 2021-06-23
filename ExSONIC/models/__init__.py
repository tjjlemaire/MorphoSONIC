# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-23 17:39:38

import sys
import inspect

from .node import *
from .radial_model import *
from .single_cable import *
from .double_cable import *


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


def getModelClassAndMeta(meta):
    ''' Return appropriate model class based on a dictionary of meta-information. '''
    simkey = meta['simkey']
    try:
        model_class = models_dict[simkey]
    except KeyError:
        raise ValueError(f'Unknown model type: {simkey}')
    if 'model' in meta:
        meta = meta['model']
    return model_class, meta


def getModel(meta):
    ''' Return appropriate model object based on a dictionary of meta-information. '''
    model_class, meta = getModelClassAndMeta(meta)
    return model_class.initFromMeta(meta)

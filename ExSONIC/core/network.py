# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-13 20:15:35
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-18 15:12:16

import pandas as pd
from neuron import h

from PySONIC.neurons import getPointNeuron
from PySONIC.core import Model, getModel
from PySONIC.utils import si_prefixes, filecode, simAndSave
from PySONIC.postpro import prependDataFrame

from .pyhoc import *
from .node import Node, DrivenNode
from .synapses import *


prefix_map = {v: k for k, v in si_prefixes.items()}


class NodeCollection:

    simkey = 'node_collection'
    tscale = 'ms'  # relevant temporal scale of the model
    titration_var = None

    node_constructor_dict = {
        'ESTIM': (Node, [], []),
        'ASTIM': (Node, [], ['a', 'Fdrive', 'fs']),
        'DASTIM': (DrivenNode, ['Idrive'], ['a', 'Fdrive', 'fs']),
    }

    def __init__(self, nodes):
        ''' Constructor.

            :param nodes: dictionary of node objects
        '''
        # Assert consistency of inputs
        ids = list(nodes.keys())
        assert len(ids) == len(set(ids)), 'duplicate node IDs'

        # Assign attributes
        self.nodes = nodes
        self.ids = ids
        self.refnode = self.nodes[self.ids[0]]
        self.pneuron = self.refnode.pneuron
        unit, factor = [self.refnode.modality[k] for k in ['unit', 'factor']]
        self.unit = f'{prefix_map[factor]}{unit}'

    def strNodes(self):
        return f"[{', '.join([repr(x.pneuron) for x in self.nodes.values()])}]"

    def strAmps(self, amps):
        return f'A = [{", ".join([f"{A:.2f}" for A in amps.values()])}] {self.unit}'

    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return f'{self.refnode.__class__.__name__}{self.__class__.__name__}({self.strNodes()})'

    def __getitem__(self, key):
        return self.nodes[key]

    def __delitem__(self, key):
        del self.nodes[key]

    def __setitem__(self, key, value):
        self.nodes[key] = value

    def clear(self):
        for node in self.nodes.values():
            node.clear()

    @classmethod
    def getNodesFromMeta(cls, meta):
        nodes = {}
        for k, v in meta['nodes'].items():
            node_class, node_args, node_kwargs = cls.node_constructor_dict[v['simkey']]
            node_args = [getPointNeuron(v['neuron'])] + [v[x] for x in node_args]
            node_kwargs = {x: v[x] for x in node_kwargs}
            nodes[k] = node_class(*node_args, **node_kwargs)
        return nodes

    @classmethod
    def initFromMeta(cls, meta):
        return cls(cls.getNodesFromMeta(meta))

    def inputs(self):
        return self.refnode.pneuron.inputs()

    def setStimON(self, value):
        ''' Set stimulation ON or OFF.

            :param value: new stimulation state (0 = OFF, 1 = ON)
            :return: new stimulation state
        '''
        for node in self.nodes.values():
            node.setStimON(value)
        return value

    def setStimAmps(self, amps):
        ''' Set distributed stimulation amplitudes.

            :param amps: model-sized dictionary of stimulus amplitudes
        '''
        for id, node in self.nodes.items():
            node.setStimAmp(amps[id])
        self.amps = amps

    def initToSteadyState(self):
        ''' Initialize model variables to pre-stimulus resting state values. '''
        for id, node in self.nodes.items():
            node.section.v = node.pneuron.Qm0 * C_M2_TO_NC_CM2
        h.finitialize()
        if self.cvode.active():
            self.cvode.re_init()
        else:
            h.fcurrent()
        h.frecord_init()

    def toggleStim(self):
        return toggleStim(self)

    @Model.addMeta
    def simulate(self, amps, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param amps: amplitude dictionary with node ids (in modality units)
            :param pp: pulse protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method.
            :return: output dataframe
        '''

        logger.info(self.desc(self.meta(amps, pp)))

        # Set recording vectors
        t = self.setTimeProbe()
        stim = self.refnode.section.setStimProbe()
        probes = {k: v.section.setProbesDict(v.pneuron.statesNames()) for k, v in self.nodes.items()}

        # Set distributed stimulus amplitudes
        self.setStimAmps(amps)

        # Integrate model
        integrate(self, pp, dt, atol)

        # Store output in dataframes
        data = {}
        for id in self.nodes.keys():
            data[id] = pd.DataFrame({
                't': t.to_array() / S_TO_MS,  # s
                'stimstate': stim.to_array()
            })
            for k, v in probes[id].items():
                data[id][k] = v.to_array()
            data[id].loc[:, 'Qm'] /= C_M2_TO_NC_CM2  # C/m2

        # Prepend initial conditions (prior to stimulation)
        data = {id: prependDataFrame(df) for id, df in data.items()}

        return data

    def modelMeta(self):
        return {
            'simkey': self.simkey
        }

    def meta(self, amps, pp):
        return {
            **self.modelMeta(),
            'nodes': {k: v.meta(amps[k], pp) for k, v in self.nodes.items()},
            'amps': amps,
            'pp': pp
        }

    def desc(self, meta):
        return f'{self}: simulation @ {self.strAmps(meta["amps"])}, {meta["pp"].desc}'

    def modelCodes(self):
        return {
            'simkey': self.simkey,
            'neurons': '_'.join([x.pneuron.name for x in self.nodes.values()])
        }

    def filecodes(self, amps, pp):
        return {
            **self.modelCodes(),
            'amps': f'[{"_".join([f"{x:.1f}" for x in amps.values()])}]{self.unit.replace("/", "")}',
            'nature': 'CW' if pp.isCW() else 'PW',
            **pp.filecodes
        }

    def filecode(self, *args):
        return filecode(self, *args)

    def getPltVars(self, *args, **kwargs):
        ref_pltvars = self.refnode.pneuron.getPltVars(*args, **kwargs)
        keys = set(ref_pltvars.keys())
        for node in self.nodes.values():
            node_keys = list(node.pneuron.getPltVars(*args, **kwargs).keys())
            keys = keys.intersection(node_keys)
        return {k: ref_pltvars[k] for k in keys}

    def getPltScheme(self, *args, **kwargs):
        ref_pltscheme = self.refnode.pneuron.getPltScheme(*args, **kwargs)
        keys = set(ref_pltscheme.keys())
        for node in self.nodes.values():
            node_keys = list(node.pneuron.getPltScheme(*args, **kwargs).keys())
            keys = keys.intersection(node_keys)
        return {k: ref_pltscheme[k] for k in keys}

    def simAndSave(self, *args, **kwargs):
        return simAndSave(self, *args, **kwargs)


class NodeNetwork(NodeCollection):

    simkey = 'node_network'

    def __init__(self, nodes, connections, presyn_var='Qm'):
        ''' Construct network.

            :param nodes: dictionary of node objects
            :param connections: {presyn_node: postsyn_node} dictionary of (syn_weight, syn_model)
            :param presyn_var: reference variable for presynaptic threshold detection (Vm or Qm)
        '''
        # Construct node collection
        super().__init__(nodes)

        # Assert consistency of inputs
        for presyn_node_id, targets in connections.items():
            assert presyn_node_id in self.ids, f'invalid pre-synaptic node ID: "{presyn_node_id}"'
            for postsyn_node_id, (syn_weight, syn_model) in targets.items():
                assert postsyn_node_id in self.ids, f'invalid post-synaptic node ID: "{postsyn_node_id}"'
                assert isinstance(syn_model, Synapse), f'invalid synapse model: {syn_model}'

        # Assign attributes
        self.connections = connections
        self.presyn_var = presyn_var

        # Connect nodes
        self.syn_objs = []
        self.netcon_objs = []
        for presyn_node_id, targets in self.connections.items():
            for postsyn_node_id, (syn_weight, syn_model) in targets.items():
                self.connect(presyn_node_id, postsyn_node_id, syn_model, syn_weight)

    @classmethod
    def initFromMeta(cls, meta):
        return cls(cls.getNodesFromMeta(meta), meta['connections'], meta['presyn_var'])

    def connect(self, source_id, target_id, syn_model, syn_weight, delay=0.0):
        ''' Connect a source node to a target node with a specific synapse model
            and synaptic weight.

            :param source_id: ID of the pre-synaptic node
            :param target_id: ID of the post-synaptic node
            :param syn_model: synapse model
            :param weight: synaptic weight (uS)
            :param delay: synaptic delay (ms)
        '''
        for id in [source_id, target_id]:
            assert id in self.ids, f'invalid node ID: "{id}"'
        syn = syn_model.attach(self.nodes[target_id])
        if self.presyn_var == 'Vm':
            hoc_var = f'Vm_{self.nodes[source_id].mechname}'
        else:
            hoc_var = 'v'
        nc = h.NetCon(
            getattr(self.nodes[source_id].section(0.5), f'_ref_{hoc_var}'),
            syn,
            sec=self.nodes[source_id].section)

        # Normalize synaptic weight
        syn_weight *= self.nodes[target_id].getAreaNormalizationFactor()

        # Assign netcon attributes
        nc.threshold = syn_model.Vthr  # pre-synaptic voltage threshold (mV)
        nc.delay = syn_model.delay     # synaptic delay (ms)
        nc.weight[0] = syn_weight          # synaptic weight (uS)

        self.syn_objs.append(syn)
        self.netcon_objs.append(nc)

    def modelMeta(self):
        return {**super().modelMeta(), **{
            'connections': self.connections,
            'presyn_var': self.presyn_var
        }}

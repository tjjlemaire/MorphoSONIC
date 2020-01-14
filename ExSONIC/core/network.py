# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-13 20:15:35
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-01-14 17:06:24

import pandas as pd
from neuron import h

from PySONIC.neurons import getPointNeuron
from PySONIC.core import Model
from PySONIC.utils import si_prefixes, filecode, simAndSave
from PySONIC.postpro import prependDataFrame

from .pyhoc import *
from .node import IintraNode

prefix_map = {v: k for k, v in si_prefixes.items()}


class Network:

    simkey = 'network'
    tscale = 'ms'  # relevant temporal scale of the model
    titration_var = None

    def __init__(self, nodes, synapses, weights):
        ''' Construct network.

            :param nodes: dictionary of node objects
            :param synapses: dictionary of synapse models
            :param weights: 2D {source:target} dictionary of synaptic weights
        '''
        # Assert consistency of inputs
        ids = list(nodes.keys())
        assert len(ids) == len(set(ids)), 'duplicate node IDs'
        for k, v in synapses.items():
            assert k in ids, f'invalid synapse ID: "{k}"'
        for source, targets in weights.items():
            assert source in ids, f'invalid source ID: "{source}"'
            assert source in synapses, f'no synapse for source ID: "{source}"'
            for target in targets.keys():
                assert target in ids, f'invalid target ID: "{target}"'

        # Assign attributes
        self.nodes = nodes
        self.ids = ids
        self.synapses = synapses
        self.weights = weights
        self.refnode = self.nodes[self.ids[0]]
        self.nodekey = self.refnode.pneuron.simkey
        self.pneuron = self.refnode.pneuron
        unit, factor = [self.refnode.modality[k] for k in ['unit', 'factor']]
        self.unit = f'{prefix_map[factor]}{unit}'

        # Connect nodes
        self.synobjs = []
        self.connections = []
        for source, targets in self.weights.items():
            syn_model = self.synapses[source]
            for target, weight in targets.items():
                A0 = self.nodes[target].section(0.5).area() * 1e-12  # section area (m2)
                A = self.nodes[target].pneuron.area  # neuron membrane area (m2)
                norm_weight = weight * A0 / A
                self.connect(source, target, syn_model, norm_weight)

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

    @classmethod
    def initFromMeta(cls, meta):
        pneurons = {k: getPointNeuron(k) for k in meta['neurons']}
        nodes = {k: IintraNode(v) for k, v in pneurons.items()}
        return cls(nodes, meta['synapses'], meta['weights'])

    def inputs(self):
        return self.refnode.pneuron.inputs()

    def connect(self, source_id, target_id, synapse, weight, delay=0.0):
        ''' Connect a source node to a target node with a specific synapse model
            and synaptic weight.

            :param source_id: ID of the source node
            :param target_id: ID of the target node
            :param synapse: synapse model
            :param weight: synaptic weight (uS)
            :param delay: synaptic delay (ms)
        '''
        for id in [source_id, target_id]:
            assert id in self.ids, f'"{id}" not found in network IDs'
        syn = synapse.assign(self.nodes[target_id])
        nc = h.NetCon(
            self.nodes[source_id].section(0.5)._ref_v,
            syn,
            sec=self.nodes[source_id].section)
        nc.delay = delay       # synaptic delay (ms)
        nc.weight[0] = weight  # weight (uS)
        self.connections.append(nc)
        self.synobjs.append(syn)

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
            node.section.v = node.pneuron.Qm0() * 1e5
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
            :param atol: absolute error tolerance for adaptive time step method (default = 1e-3)
            :return: output dataframe
        '''

        logger.info(self.desc(self.meta(amps, pp)))

        # Set recording vectors
        t = setTimeProbe()
        stim = setStimProbe(self.refnode.section, self.refnode.mechname)
        probes = {k: v.setProbesDict(v.section) for k, v in self.nodes.items()}

        # Set distributed stimulus amplitudes
        self.setStimAmps(amps)

        # Integrate model
        integrate(self, pp, dt, atol)

        # Store output in dataframes
        data = {}
        for id in self.nodes.keys():
            data[id] = pd.DataFrame({
                't': vec_to_array(t) * 1e-3,  # s
                'stimstate': vec_to_array(stim)
            })
            for k, v in probes[id].items():
                data[id][k] = vec_to_array(v)
            data[id].loc[:,'Qm'] *= 1e-5  # C/m2

        # Prepend initial conditions (prior to stimulation)
        data = {id: prependDataFrame(df) for id, df in data.items()}

        return data

    def modelMeta(self):
        return {
            'simkey': self.simkey,
            'nodekey': self.nodekey,
            'neurons': [x.pneuron.name for x in self.nodes.values()],
            'synapses': self.synapses,
            'weights': self.weights
        }

    def meta(self, amps, pp):
        return {**self.modelMeta(), **{
            'amps': amps,
            'pp': pp
        }}

    def desc(self, meta):
        return f'{self}: simulation @ {self.strAmps(meta["amps"])}, {meta["pp"].pprint()}'

    def modelCodes(self):
        return {
            'simkey': self.simkey,
            'nodekey': self.nodekey,
            'neurons': '_'.join([x.pneuron.name for x in self.nodes.values()])
        }

    def filecodes(self, amps, pp):
        return {
            **self.modelCodes(),
            'amps': f'[{"_".join([f"{x:.1f}" for x in amps.values()])}]{self.unit.replace("/", "")}',
            'nature': 'CW' if pp.isCW() else 'PW',
            **pp.filecodes()
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





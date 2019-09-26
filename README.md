# Description

`ExSONIC` is a Python+NEURON implementation of **spatially extended representations** of the **multi-Scale Optimized Neuronal Intramembrane Cavitation (SONIC) model [1]** to simulate the distributed electrical response of morphologically realistic neuron representations to acoustic stimuli, as predicted by the *intramembrane cavitation* hypothesis.

This package expands features from the `PySONIC` package (https://c4science.ch/diffusion/4670/browse/master/PySONIC/).

## Content of repository

### Single-compartment (node) models

The package contains several **node** classes that define the punctual membrane sections:
- `Node` defines the generic interface of a generic punctual membrane section.
- `IintraNode` defines a punctual node that can be simulated with intracellular current injection.
- `SonicNode` defines a punctual node that can be simulated with ultrasound stimulation.

### Multi-compartment (spatially-extended) models

The package also contains several classes defining multi-compartmental  model expansions, at various spatial scales.

At the nanometer scale, an `ExtendedSonicNode` class that simulate the behavior of a **nanoscale spatially-extended SONIC model** with two coupled compartments (i.e. nodes with geometrical extents): an "ultrasound-responsive" sonophore and an "ultrasound-resistant" surrounding membrane. As this model is radially symmetric, some adaptation was needed in order to represent it within the *NEURON* environment (check [this link](NEURON_radial_geometry.md) for more details).

At the morphological scale, several **fiber** classes define spatially-extended fiber models:
- `SennFiber` defines ...
- `VextSennFiber` defines ...
- `IinjSennFiber` defines ...
- `SonicSennFiber` defines ...

Moreover, several **source** classes are implemented to compute spatially-distributed inputs from various source types, used to drive the models:
- `PointSource`
- ...

### Membrane mechanisms (NMODL)

The membrane mechanisms of several conductance-based point-neuron models are implemented in the **NMODL** neuron language:
- `CorticalRS`: cortical regular spiking (`RS`) neuron
- `CorticalFS`: cortical fast spiking (`FS`) neuron
- `CorticalLTS`: cortical low-threshold spiking (`LTS`) neuron
- `CorticalIB`: cortical intrinsically bursting (`IB`) neuron
- `ThalamicRE`: thalamic reticular (`RE`) neuron
- `ThalamoCortical`: thalamo-cortical (`TC`) neuron
- `OstukaSTN`: subthalamic nucleus (`STN`) neuron
- `FrankenhaeuserHuxley`: Xenopus myelinated fiber node (`FH`)

### Other modules

- `pyhoc`: defines utilities for Python-NEURON communication
- `pymodl`: defines a parser to translate point-neuron models defined in Python into NMODL membrane mechanisms
- `parsers`: command line parsing utilities
- `plt`: graphing utilities
- `constants`: algorithmic constants
- `utils`: generic utilities

# Requirements

- Python 3.6+
- NEURON 7.x
- PySONIC package
- Other dependencies (numpy, scipy, ...) are installed automatically upon installation of the package.

# Installation

## NEURON 7.x

### Windows

- Go to the NEURON website https://neuron.yale.edu/neuron/download/ and download the appropriate *NEURON* installer for Windows

- Run the *NEURON* installer and follow the procedure:
  - Confirm installation directory (`c:\nrn`)
  - Check the option "Set DOS environment"
  - Click on "Install"
After completion you should see a new folder named `NEURON 7.x x86_64` on your Desktop.

- Open the `NEURON 7.x x86_64` folder and run the "NEURON Demo" executable. You should see the NEURON GUI appearing.
- Click on "Release" on the "NEURON Demonstrations" window. A number of windows should appear.
- Click on "Init & Run" button from the "RunControl" window. You should see the trace of a single action potential on the first Graph window.
- Exit *NEURON* by clicking on "File->Quit" in the "NEURON Main Menu" window
- Log out and back in to make sure your environment variables are updated.

### Ubuntu

In the following instructions, replace 7.x by the appropriate *NEURON* version.

- Install ncurses LINUX package:
``` $ apt install ncurses-dev ```

- Download the NEURON source code archive:

https://neuron.yale.edu/ftp/neuron/versions/v-7.x/nrn-7.x.tar.gz

- Unzip the archive:
``` $ tar xzf nrn-7.x.tar.gz ```

- Install NEURON (without GUI)
```
$ cd nrn-7.x
$ ./configure --prefix=/usr/local/nrn-7.x --without-iv --with-nrnpython=<path/to/python/executable>
$ make
$ make install
$ make clean
```
Example of path to Anaconda3 Python3.6 executable: `/opt/apps/anaconda3/bin/python3.6`

- Add *NEURON* executables to the global environment file:
```
$ vim /etc/environment
    PATH=<old_path>:/usr/local/nrn-7.x/x86_64/bin
    exit
```

- Check that *NEURON* has been properly installed:
```
$ nrniv -python
NEURON -- VERSION 7.x master (...) ...
Duke, Yale, and the BlueBrain Project -- Copyright 1984-xxxx
See http://neuron.yale.edu/neuron/credits

>>> quit()
```

- Go back to unzipped archive directory
``` $ cd <path/to/unzipped/archive> ```

- Install the neuron package for Python 3.x:
```
$ cd src/nrnpython
$ <path/to/python/executable> setup.py install
```

## PySONIC package

- Download the PySONIC package (https://c4science.ch/diffusion/4670/browse/master/PySONIC/)
- Follow installation instruction written in the README file

## ExSONIC package

- Open a terminal.

- Activate a Python3 environment if needed, e.g. on the tnesrv5 machine:

```$ source /opt/apps/anaconda3/bin activate```

- Check that the appropriate version of pip is activated:

```$ pip --version```

- Go to the package directory (where the setup.py file is located):

```$ cd <path_to_directory>```

- Install the package and all its dependencies:

```$ pip install -e .```

# Usage

## Python scripts

You can easily run simulations of any implemented point-neuron model under both electrical and ultrasonic stimuli, and visualize the simulation results, in just a few lines of code:

```python
import logging
import matplotlib.pyplot as plt

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger
from PySONIC.plt import GroupedTimeSeries

from ExSONIC.core import IintraNode, SonicNode

logger.setLevel(logging.INFO)

# Stimulation parameters
a = 32e-9        # m
Fdrive = 500e3   # Hz
Adrive = 100e3   # Pa
Astim = 10.      # mA/m2
tstim = 250e-3   # s
toffset = 50e-3  # s
PRF = 100.       # Hz
DC = 0.5         # -

# Point-neuron model and corresponding Iintra and SONIC node models
pneuron = getPointNeuron('RS')
estim_node = IintraNode(pneuron)
sonic_node = SonicNode(pneuron, a=a, Fdrive=Fdrive)

# Run simulation upon electrical stimulation, and plot results
data, meta = estim_node.simulate(Astim, tstim, toffset, PRF, DC)
fig1 = GroupedTimeSeries([(data, meta)]).render()

# Run simulation upon ultrasonic stimulation, and plot results
data, meta = sonic_node.simulate(Adrive, tstim, toffset, PRF, DC)
fig2 = GroupedTimeSeries([(data, meta)]).render()

plt.show()
```

Similarly, you can run simulations of SENN-type myelinated fiber models under extracellular electrical stimulation, and visualize the simulation results:

```python
import logging
import matplotlib.pyplot as plt

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format
from ExSONIC.core import VextSennFiber, CurrentPointSource
from ExSONIC.plt import SectionCompTimeSeries

logger.setLevel(logging.INFO)

# Fiber model
fiberD = 20e-6  # m
nnodes = 11
rs = 110.0      # Ohm.cm
pneuron = getPointNeuron('FH')
fiber = VextSennFiber(pneuron, fiberD, nnodes, rs=rs)

# Point-source electrode object
z0 = fiber.interL  # z-position (m): one internodal distance away from fiber
x0 = 0.            # x-position (m): aligned with fiber's central node
elec_psource = CurrentPointSource(x0, z0, mode='cathode')

# Stimulation Parameters
tstim = 100e-6  # s
toffset = 3e-3  # s
PRF = 100.      # HZ
DC = 1.         # -

# Titrate for a specific duration and simulate fiber at threshold current
Ithr = fiber.titrate(elec_psource, tstim, toffset, PRF, DC)
logger.info(f'tstim = {si_format(tstim)}s -> Ithr = {si_format(Ithr)}A')
data, meta = fiber.simulate(elec_psource, Ithr, tstim, toffset, PRF, DC)

# Plot membrane potential traces across nodes for stimulation at threshold current
fig = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

plt.show()
```

## From the command line

### Running simulations

You can easily run simulations of punctual and spatially-extended models using the dedicated command line scripts. To do so, open a terminal in the `scripts` directory.

- Use `run_node_estim.py` for simulations of **point-neuron models** upon **intracellular electrical stimulation**. For instance, a regular-spiking (RS) neuron injected with 10 mA/m2 intracellular current for 30 ms:

```$ python run_node_estim.py -n RS -A 10 --tstim 30 -p Vm```

- Use `run_node_astim.py` for simulations of **point-neuron models** upon **ultrasonic stimulation**. For instance, a 32 nm radius bilayer sonophore within a regular-spiking (RS) neuron membrane, sonicated at 500 kHz and 100 kPa for 150 ms:

```$ python run_node_astim.py -n RS -a 32 -f 500 -A 100 --tstim 150 --method sonic -p Qm```

- Use `run_ext_node_astim.py` for simulations of the **nanometer-scale, spatially-extended SONIC model** of any neuron type upon **ultrasonic stimulation**. For instance, a 32 nm radius bilayer sonophore surrounded by a circular "US-insensitive" patch, within a regular-spiking (RS) neuron membrane with 30% sonophore coverage, sonicated at 500 kHz and 100 kPa for 150 ms:

```$ python run_ext_sonic_node_astim.py -n RS -a 32 --fs 30 -f 500 -A 100 --tstim 150 --method sonic -p Qm --compare --section all```

- Use `run_senn_estim.py` for simulations of a **SENN-type myelinated fiber model** of any diameter and with any number of nodes upon **extracellular electrical stimulation**. For instance, a 20 um diameter, 11 nodes fiber with Frankenhaeuser-Huxley (FH) nodal membrane dynamics, stimulated at 0.6 mA for 0.1 ms by a cathodal point-source electrode located one internodal distance above the central node:

```$ python run_senn_estim.py -n FH -d 20 --nnodes 11 -A -0.6 --tstim 0.1 -p Vm --compare --section all```

### Saving and visualizing results

By default, simulation results are neither shown, nor saved.

To view results directly upon simulation completion, you can use the `-p [xxx]` option, where `[xxx]` can be `all` (to plot all resulting variables) or a given variable name (e.g. `Z` for membrane deflection, `Vm` for membrane potential, `Qm` for membrane charge density).

To save simulation results in binary `.pkl` files, you can use the `-s` option. You will be prompted to choose an output directory, unless you also specify it with the `-o <output_directory>` option. Output files are automatically named from model and simulation parameters to avoid ambiguity.

When running simulation batches, it is highly advised to specify the `-s` option in order to save results of each simulation. You can then visualize results at a later stage.

To visualize results, use the `plot_ext_timeseries.py` script. You will be prompted to select the output files containing the simulation(s) results. By default, separate figures will be created for each simulation, showing the time profiles of all resulting variables in the default morphological section of the model. Here again, you can choose to show only a subset of variables using the `-p [xxx]` option, and specify morphological sections of interest with the `--section [xxx]` option. Moreover, if you select a subset of variables, you can visualize resulting profiles across sections in comparative figures wih the `--compare` option.

## On the integration of SONIC model representations within *NEURON*

The SONIC model predicts a time-varying capacitance introducing a nonlinear discrepancy between membrane potential and charge density, something that *NEURON* is not natively equipped to deal with. Therefore, several workaround strategies were designed in order to adapt *NEURON* for the numerical simulation of this model, as well as its spatial extension. You can check [this link](NEURON_integration_strategies.md) for fore details.

## Future developments

Here is a list of future developments:

- [ ] Spatial expansion into morphological realistic fiber models
- [ ] Model validation against experimental data (leech neurons)

# Authors

Code written and maintained by Theo Lemaire (theo.lemaire@epfl.ch).

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# References

[1] Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019). Understanding ultrasound neuromodulation using a computationally efficient and interpretable model of intramembrane cavitation. J. Neural Eng.
# Description

`MorphoSONIC` is a Python+NEURON implementation of **spatially extended representations** of the **multi-Scale Optimized Neuronal Intramembrane Cavitation (SONIC) model [1]** to simulate the distributed electrical response of morphologically realistic neuron representations to acoustic stimuli, as predicted by the *intramembrane cavitation* hypothesis.

This package expands features from the `PySONIC` package (https://github.com/tjjlemaire/PySONIC).

## Content of repository

### Single-compartment (node) models

The package contains a `Node` class that provides a NEURON wrapper around the models defined in the PySONIC package. This class defines a generic section object with a specific membrane dynamics that can be simulated with both punctual electrical and acoustic drives.

### Multi-compartment (spatially-extended) models

The package also contains several classes defining multi-compartmental model expansions, at various spatial scales.

At the nanometer scale, a `RadialModel` class that simulate the behavior of a **nanoscale radially-symmetric model** with central and peripheral compartments. It can be used to model the coupling between an "ultrasound-responsive" sonophore and an "ultrasound-resistant" surrounding membrane (see `usrroundedSonophore` function. As this model is radially symmetric, some adaptation was needed in order to represent it within the *NEURON* environment (check [this link](NEURON_radial_geometry.md) for more details).

At the morphological scale, several models of **unmyelinated and myelinated peripheral fibers** are implemented:
- `SennFiber` implements a spatially-extended nonlinear node (SENN) myelinated fiber model, as defined in Reilly 1985.
- `SweeneyFiber` implements the SENN model variant defined in Sweeney 1987.
- `MRGFiber` implements the double-cable myelinated fiber model defined as in McIntyre 2002.
- `UnmyelinatedFiber` implements an unmyelinated fiber model defined as in Sundt 2015.

Those fiber models can be simulate upon stimulation by different types of **source** objects:
- `IntracellularCurrent` for local intracellular current injection at a specific section.
- `ExtracellularCurrent` for distributed voltage perturbation resulting from current injection at a distant point-source electrode.
- `SectionAcousticSource` for local acoustic perturbation at a specific section.
- `PlanarDiskTransducerSource` for distributed acoustic perturbation resulting from sonication by a distant planar acoustic transducer.

### Membrane mechanisms (NMODL)

Most point-neuron models defined in the PySONIC package have been translated to equivalent membrane mechanisms in **NMODL** language. Please refer to the PySONIC package for a list of these membrane mechanisms.

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

**In the following instructions, replace 7.x by the appropriate *NEURON* version.**

### Windows

1. Go to the [NEURON website](https://neuron.yale.edu/neuron/download/) and download the appropriate *NEURON* installer for Windows
2. Run the *NEURON* installer and follow the procedure:
  - Confirm installation directory (`c:\nrn`)
  - Check the option "Set DOS environment"
  - Click on "Install". After completion you should see a new folder named `NEURON 7.x x86_64` on your Desktop.
3. Check that *NEURON* has been properly installed by running the demo:
  - Open the `NEURON 7.x x86_64` folder and run the "NEURON Demo" executable. You should see the NEURON GUI appearing.
  - Click on "Release" on the "NEURON Demonstrations" window. A number of windows should appear.
  - Click on "Init & Run" button from the "RunControl" window. You should see the trace of a single action potential on the first Graph window.
  - Exit *NEURON* by clicking on "File->Quit" in the "NEURON Main Menu" window
4. Log out and back in to make sure your environment variables are updated.
5. Open a terminal and check that the *NEURON* import in Python works correctly (no error should be raised):
```
python
>>> from neuron import h
>>> quit()
```

### Mac

1. Go to the [NEURON website](https://neuron.yale.edu/neuron/download/) and download the appropriate *NEURON* installer for Mac
2. Run the *NEURON* installer
3. Check that *NEURON* has been properly installed by running the demo:
  - Try to run the *neurondemo* (in the NEURON folder). If the NEURON GUI appears correctly, go to 4. Otherwise, follow the steps:
  - Install *XQuartz* from https://www.xquartz.org
  - Restart the computer
  - Try to run the *neurondemo* again. It should execute properly.
4. If you don’t have *XCode*, install it from the App Store
5. Go to https://developer.apple.com/downloads, sign with the ID Apple, download the right Command Line Tools based on your OS X and XCode versions, and install it
6. Open a terminal and add *NEURON* python package to your python path: `export PYTHONPATH=/Applications/NEURON-7.x/nrn/lib/python`
7. Restart the computer
8. Open a terminal and check that the *NEURON* import in Python works correctly (no error should be raised):
```
python
>>> from neuron import h
>>> quit()
```

### Ubuntu

1. Open a terminal
2. Install the *ncurses* LINUX package: `apt install ncurses-dev`
3. Download the [NEURON source code archive](https://neuron.yale.edu/ftp/neuron/versions/v-7.x/nrn-7.x.tar.gz)
4. Unzip the archive: `tar xzf nrn-7.x.tar.gz`
5. Install NEURON (without GUI):
```
cd nrn-7.x
./configure --prefix=/usr/local/nrn-7.x --without-iv --with-nrnpython=<path/to/python/executable>
make
make install
make clean
```
6. Add *NEURON* executables to the global environment file:
```
vim /etc/environment
PATH=<old_path>:/usr/local/nrn-7.x/x86_64/bin
exit
```
7. Check that *NEURON* has been properly installed:
```
nrniv -python
NEURON -- VERSION 7.x master (...) ...
Duke, Yale, and the BlueBrain Project -- Copyright 1984-xxxx
See http://neuron.yale.edu/neuron/credits
>>> quit()
```
8. Go back to unzipped archive directory: `cd <path/to/unzipped/archive>`
9. Install the neuron package for Python 3.x:
```
cd src/nrnpython
<path/to/python/executable> setup.py install
```
10. Open a terminal and check that the *NEURON* import in Python works correctly (no error should be raised):
```
python
>>> from neuron import h
>>> quit()
```

## PySONIC package

- Download the PySONIC package (https://github.com/tjjlemaire/PySONIC)
- Follow installation instruction written in the README file

## MorphoSONIC package

- Open a terminal.
- Activate a Python3 environment if needed, e.g. on the tnesrv5 machine: `source /opt/apps/anaconda3/bin activate`
- Check that the appropriate version of pip is activated: `pip --version`
- Go to the package directory (where the setup.py file is located): `cd <path_to_directory>`
- Install the package and all its dependencies: `pip install -e .`

## Compile NEURON membrane mechanisms

In order to use the package, you will need to compile a specific set of equations describing the membrane dynamics of the different neuron types.

### Windows

- In the folder named `NEURON 7.x x86_64` on your Desktop, run the `mknrndll` executable.

- In the displayed window, select the directory containing the source files for the membrane mechanisms: *.../MorphoSONIC/MorphoSONIC/nmodl/*

- Click on "make nrnmech.dll"

- Upon completion, hit enter in the terminal to close it.

### Mac OSx and Ubuntu

- Open a terminal window and move to the directory containing the source files for the membrane mechanisms:

```cd <path_to_MorphoSONIC_package/MorphoSONIC/nmodl/```

- Run the *nrnivmodl* executable:

```nrnivmodl```

# Usage

## Python scripts

You can easily run simulations of any implemented point-neuron model under both electrical and ultrasonic stimuli, and visualize the simulation results, in just a few lines of code:

```python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:45:54

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol, ElectricDrive, AcousticDrive
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger
from PySONIC.plt import GroupedTimeSeries
from MorphoSONIC.models import Node

# Set logging level
logger.setLevel(logging.INFO)

# Define point-neuron model
pneuron = getPointNeuron('RS')

# Define sonophore parameters
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)

# Create node model
node = Node(pneuron, a=a, fs=fs)

# Define electric and ultrasonic drives
EL_drive = ElectricDrive(20.)  # mA/m2
US_drive = AcousticDrive(
    500e3,  # Hz
    100e3)  # Pa

# Set pulsing protocol
tstim = 250e-3   # s
toffset = 50e-3  # s
PRF = 100.       # Hz
DC = 0.5         # -
pp = PulsedProtocol(tstim, toffset, PRF, DC)

# Simulate model with each drive modality and plot results
for drive in [EL_drive, US_drive]:
    data, meta = node.simulate(drive, pp)
    GroupedTimeSeries([(data, meta)]).render()

# Show figures
plt.show()

```

Similarly, you can run simulations of myelinated and unmyelinated fiber models under extracellular electrical and ultrasonic stimulation, and visualize the simulation results:

```python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:45:43

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol, BalancedPulsedProtocol
from PySONIC.utils import logger

from MorphoSONIC.models import SennFiber
from MorphoSONIC.sources import *
from MorphoSONIC.plt import SectionCompTimeSeries

# Set logging level
logger.setLevel(logging.INFO)

# Define sonophore parameters
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)

# Define fiber parameters
fiberD = 20e-6  # m
nnodes = 21

# Create fiber model
fiber = SennFiber(fiberD, nnodes, a=a, fs=fs)

# Define various sources
iintra_source = IntracellularCurrent(
    sec_id=fiber.central_ID,  # target section
    I=3.0e-9)                 # current amplitude (A)
iextra_source = ExtracellularCurrent(
    x=(0., fiber.interL),  # point-source electrode position (m)
    I=-0.70e-3)            # current amplitude (A)
voltage_source = GaussianVoltageSource(
    0,                   # gaussian center (m)
    fiber.length / 10.,  # gaussian width (m)
    Ve=-80.)             # peak extracellular voltage (mV)
section_US_source = SectionAcousticSource(
    fiber.central_ID,  # target section
    500e3,             # US frequency (Hz)
    A=100e3)           # peak acoustic amplitude (Pa)
gaussian_US_source = GaussianAcousticSource(
    0,                   # gaussian center (m)
    fiber.length / 10.,  # gaussian width (m)
    500e3,               # US frequency (Hz)
    A=100e3)             # peak acoustic amplitude (Pa)
transducer_source = PlanarDiskTransducerSource(
    (0., 0., 'focus'),  # transducer position (m)
    500e3,              # US frequency (Hz)
    r=2e-3,             # transducer radius (m)
    u=0.04)             # m/s

# Define pulsing protocols
tpulse = 100e-6  # s
xratio = 0.2     # (-)
toffset = 3e-3   # s
standard_pp = PulsedProtocol(tpulse, toffset)                  # (for US sources)
balanced_pp = BalancedPulsedProtocol(tpulse, xratio, toffset)  # (for electrical sources)

# Define source-protocol pairs
pairs = [
    (iintra_source, balanced_pp),
    (iextra_source, balanced_pp),
    (voltage_source, balanced_pp),
    (section_US_source, standard_pp),
    (gaussian_US_source, standard_pp),
    (transducer_source, standard_pp)
]

# Simulate model with each source-protocol pair, and plot results
for source, pp in pairs:
    data, meta = fiber.simulate(source, pp)
    SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

plt.show()

```

## From the command line

### Running simulations

You can easily run simulations of punctual and spatially-extended models using the dedicated command line scripts. To do so, open a terminal in the `scripts` directory.

- Use `run_node_estim.py` for simulations of **point-neuron models** upon **intracellular electrical stimulation**. For instance, a regular-spiking (RS) neuron injected with 10 mA/m2 intracellular current for 30 ms:

```python run_node_estim.py -n RS -A 10 --tstim 30 -p Vm```

- Use `run_node_astim.py` for simulations of **point-neuron models** upon **ultrasonic stimulation**. For instance, a 32 nm radius bilayer sonophore within a regular-spiking (RS) neuron membrane, sonicated at 500 kHz and 100 kPa for 150 ms:

```python run_node_astim.py -n RS -a 32 -f 500 -A 100 --tstim 150 --method sonic -p Qm```

- Use `run_fiber_iextra.py` for simulations of a **peripheral fiber models** (myelinated or unmyelinated) of any diameter and with any number of nodes upon **extracellular electrical stimulation**. For instance, a 20 um diameter, 11 nodes SENN-type myelinated fiber, stimulated at 0.6 mA for 0.1 ms by a cathodal point-source electrode located one internodal distance above the central node:

```python run_fiber_iextra.py --type senn -d 20 --nnodes 11 -A -0.6 --tstim 0.1 -p Vm --compare --section all```

- Use `run_fiber_iintra.py` for simulations of a **peripheral fiber models** (myelinated and unmyelinated) of any diameter and with any number of nodes upon **intracellular electrical stimulation**. For instance, a 20 um diameter, 11 nodes SENN-type fiber, stimulated at 3 nA for 0.1 ms by a anodic current injected intracellularly at the central node:

```python run_fiber_iintra.py --type senn -d 20 --nnodes 11 -A 3 --tstim 0.1 -p Vm --compare --section all```

### Saving and visualizing results

By default, simulation results are neither shown, nor saved.

To view results directly upon simulation completion, you can use the `-p [xxx]` option, where `[xxx]` can be `all` (to plot all resulting variables) or a given variable name (e.g. `Z` for membrane deflection, `Vm` for membrane potential, `Qm` for membrane charge density).

To save simulation results in binary `.pkl` files, you can use the `-s` option. You will be prompted to choose an output directory, unless you also specify it with the `-o <output_directory>` option. Output files are automatically named from model and simulation parameters to avoid ambiguity.

When running simulation batches, it is highly advised to specify the `-s` option in order to save results of each simulation. You can then visualize results at a later stage.

To visualize results, use the `plot_ext_timeseries.py` script. You will be prompted to select the output files containing the simulation(s) results. By default, separate figures will be created for each simulation, showing the time profiles of all resulting variables in the default morphological section of the model. Here again, you can choose to show only a subset of variables using the `-p [xxx]` option, and specify morphological sections of interest with the `--section [xxx]` option. Moreover, if you select a subset of variables, you can visualize resulting profiles across sections in comparative figures wih the `--compare` option.

### morphoSONIC paper simulations

The `/paper/` subfolder contains the scripts used to generate figures reported in the morphoSONIC paper.

Before running the scripts, you will need to specify an output directory to save the figures in. To do so, update the `figdir` variable in the `root.py` module of this subfolder:

`figdir = '/path/to/your/output/directory'`

Once that is done, you can run each script from the command line:

```python figX.py ```

replacing `X` with the appropriate number.

## On the integration of SONIC model representations within *NEURON*

The SONIC model predicts a time-varying capacitance introducing a nonlinear discrepancy between membrane potential and charge density, something that *NEURON* is not natively equipped to deal with. Therefore, several workaround strategies were designed in order to adapt *NEURON* for the numerical simulation of this model, as well as its spatial extension. You can check [this link](NEURON_integration_strategies.md) for fore details.

## Future developments

Here is a list of future developments:

- [ ] Spatial expansion into more complex neuronal morphologies (axons, soma, dendrites)
- [ ] Experimental validation of the fiber models

# Authors

Code written and maintained by Theo Lemaire (theo.lemaire@epfl.ch).

# License

At the moment, access to this project is granted strictly for the purpose of reviews of the associated paper. As such, the code should not be re-used or distributed - see the LICENSE file for details.

# References

[1] Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019). Understanding ultrasound neuromodulation using a computationally efficient and interpretable model of intramembrane cavitation. J. Neural Eng.
# Description

`ExSONIC` is a Python+NEURON implementation of **spatially extended representations** of the **multi-Scale Optimized Neuronal Intramembrane Cavitation (SONIC) model [1]** to simulate the distributed electrical response of morphologically realistic neuron representations to acoustic stimuli, as predicted by the *intramembrane cavitation* hypothesis.

This package expands features from the `PySONIC` package (https://c4science.ch/diffusion/4670/browse/master/PySONIC/).

## Content of repository

### Models

The package contains several **node** classes that define the punctual membrane sections:
- `Node` defines the generic interface of a generic punctual membrane section.
- `IintraNode` defines a punctual node that can be simulated with intracellular current injection.
- `SonicNode` defines a punctual node that can be simulated with ultrasound stimulation.

It also contain an `ExtendedSonicNode` class that simulate the behavior of a nanoscale spatially-extended SONIC model with two coupled compartments (i.e. nodes with geometrical extents): an "ultrasound-reponsive" sonophore and an "ultrasound-resistant" surrounding membrane.

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
- Exit *NEURON* by cliking on "File->Quit" in the "NEURON Main Menu" window
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

### ExSONIC package

- Open a terminal.

- Activate a Python3 environment if needed, e.g. on the tnesrv5 machine:

```$ source /opt/apps/anaconda3/bin activate```

- Check that the appropriate version of pip is activated:

```$ pip --version```

- Go to the package directory (where the setup.py file is located):

```$ cd <path_to_directory>```

- Insall the package and all its dependencies:

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

## From the command line

### Running simulations and batches

You can easily run simulations of all 3 model types using the dedicated command line scripts. To do so, open a terminal in the `scripts` directory.

- Use `run_node_estim.py` for simulations of **point-neuron models** upon **intracellular electrical stimulation**. For instance, a regular-spiking (RS) neuron injected with 10 mA/m2 intracellular current for 30 ms:

```$ python run_node_estim.py -n RS -A 10 --tstim 30 -p Vm```

- Use `run_node_astim.py` for simulations of **point-neuron models** upon **ultrasonic stimulation**. For instance, for a coarse-grained simulation of a 32 nm radius bilayer sonophore within a regular-spiking (RS) neuron membrane, sonicated at 500 kHz and 100 kPa for 150 ms:

```$ python run_node_astim.py -n RS -a 32 -f 500 -A 100 --tstim 150 --method sonic -p Qm```

Additionally, you can run batches of simulations by specifying more than one value for any given stimulation parameter (e.g. `-A 100 200` for sonication with 100 and 200 kPa respectively). These batches can be parallelized using multiprocessing to optimize performance, with the extra argument `--mpi`.

### Saving and visualizing results

By default, simulation results are neither shown, nor saved.

To view results directly upon simulation completion, you can use the `-p [xxx]` option, where `[xxx]` can be `all` (to plot all resulting variables) or a given variable name (e.g. `Z` for membrane deflection, `Vm` for membrane potential, `Qm` for membrane charge density).

To save simulation results in binary `.pkl` files, you can use the `-s` option. You will be prompted to choose an output directory, unless you also specify it with the `-o <output_directory>` option. Output files are automatically named from model and simulation parameters to avoid ambiguity.

When running simulation batches, it is highly advised to specify the `-s` option in order to save results of each simulation. You can then visualize results at a later stage.

**TODO: add visualization script**

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
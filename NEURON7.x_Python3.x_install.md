# How to install and run *NEURON* with Python 3.x

In the following instructions, replace `7.x` by the appropriate *NEURON* version, and `3.x.x` by the appropriate Python version.

## Install Python 3.x

### Windows

- Clean environment variables refering to *NEURON*, if any

- Go to https://www.anaconda.com/download/ and download the Anaconda installer (*Python 3.x* version)

- Run the Anaconda installer and follow the procedure (preferably not adding python to your *PATH*)

- Open the Anaconda prompt and check that Python 3.x is working:
```
$ python
Python 3.x.x |Anaconda, Inc.| (default, ...) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> print('hello')
hello
>>> quit()
$
```

### Ubuntu

TODO


## Install NEURON 7.x

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

### Use *NEURON* from Python

Run Python 3.x from the terminal and check that you can properly import *NEURON*:

```
$ python
Python 3.x.x |Anaconda, Inc.| (default, ...) [...] on ...
Type "help", "copyright", "credits" or "license" for more information.
>>> import neuron
NEURON -- VERSION 7.x master (...) ...
Duke, Yale, and the BlueBrain Project -- Copyright 1984-20xx
See http://neuron.yale.edu/neuron/credits

>>> from neuron import h
>>> soma = h.Section(name='soma')
>>> soma.insert('pas')
soma
<class 'nrn.Section'>
>>> print(type(soma))
>>> quit()
$
```
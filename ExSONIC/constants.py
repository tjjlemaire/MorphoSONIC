# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-03 16:49:43

# Numerical integration
FIXED_DT = 1e-5  # default time step for fixed integration (s)

# Titration ranges
IINJ_RANGE = (1e-12, 1e-7)  # intracellular current range allowed at the fiber level (A)
VEXT_RANGE = (1e0, 1e5)     # extracellular potential range allowed at the fiber level (mV)

# Model construction
MAX_CUSTOM_CON = 2  # maximum number of custom intracellular connections to a given section

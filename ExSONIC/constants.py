# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-05 17:28:59

# Conversions
V_TO_MV = 1e3         # V to  mV
PA_TO_KPA = 1e-3      # Pa -> kPa
HZ_TO_KHZ = 1e-3      # Hz -> kHz
C_M2_TO_NC_CM2 = 1e5  # C/m2 -> nC/cm2
S_TO_MS = 1e3         # s -> ms
S_TO_US = 1e6         # s -> us
M_TO_CM = 1e2         # m -> cm
M_TO_MM = 1e3         # m -> mm
M_TO_UM = 1e6         # m to um
M_TO_NM = 1e9         # m to nm
CM_TO_UM = 1e4        # cm -> um
A_TO_NA = 1e9         # A -> nA
MA_TO_A = 1e-3        # mA -> A
MA_TO_NA = 1e6        # mA -> nA
F_M2_TO_UF_CM2 = 1e2  # F/m2 -> uF/cm2
OHM_TO_MOHM = 1e-6    # Ohm-> MOhm

# Numerical integration
FIXED_DT = 1e-5        # default time step for fixed integration (s)
TRANSITION_DT = 1e-12  # time step for ON-OFF and OFF-ON sharp transitions (s)

# Titration
IINJ_RANGE = (1e-12, 1e-7)  # intracellular current range allowed at the fiber level (A)
VEXT_RANGE = (1e0, 1e5)     # extracellular potential range allowed at the fiber level (mV)
REL_EPS_THR = 1e-2          # relative convergence threshold
REL_START_POINT = 0.2       # relative position of starting point within search range

# Model construction
MAX_CUSTOM_CON = 2  # maximum number of custom intracellular connections to a given section

# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2019-01-23 18:06:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-01-23 18:06:13


US_AMP_MAX = 600.  # upper limit for acoustic pressure amplitude (kPa)
DELTA_US_AMP_MIN = .1  # refinement threshold for titration with acoustic pressure amplitude (kPa)
IINJ_INTRA_MAX = 1e5  # upper limit for intracellular current amplitude (mA/m2)
IINJ_EXTRA_CATHODAL_MAX = 1e4  # upper limit for cathodal stimulation current magnitude (uA)
IINJ_EXTRA_ANODAL_MAX = 1e4  # upper limit for anodal stimulation current magnitude (uA)
DELTA_IINJ_INTRA_MIN = 1e3  # refinement threshold for titration with intracellular current (mA/m2)
DELTA_IINJ_EXTRA_MIN = 1e1  # refinement threshold for titration with extracellular current (uA)

TITLE Custom passive current

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
   SUFFIX custom_pas
   NONSPECIFIC_CURRENT iPas : passive leakage current
   RANGE Adrive, Vm : section specific
   RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
   stimon         : Stimulation state
   Adrive (kPa)   : Stimulation amplitude
   gPas   (S/cm2)
   EPas   (mV)
}

ASSIGNED {
   v    (nC/cm2)
   Vm   (mV)
   iPas (mA/cm2)
}

FUNCTION_TABLE V(A(kPa), Q(nC/cm2)) (mV)

BREAKPOINT {
   Vm = V(Adrive * stimon, v)
   Vm = v
   iPas = gPas * (Vm - EPas)
}
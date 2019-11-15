TITLE sundt membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a Unmyelinated C-fiber model.
upon electrical / ultrasonic stimulation, based on the SONIC model.

Reference: Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019).
Understanding ultrasound neuromodulation using a computationally efficient
and interpretable model of intramembrane cavitation. J. Neural Eng.

@Author: Theo Lemaire, EPFL
@Date: 2019-11-07
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
   SUFFIX sundtauto
   NONSPECIFIC_CURRENT iNa : Sodium current.  Gating formalism from Migliore 1995, using 3rd power for m in order to reproduce thinner AP waveform (half-width of ca. 1 ms)
   NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
   NONSPECIFIC_CURRENT iM : slow non-inactivating Potassium current
   NONSPECIFIC_CURRENT iLeak : non-specific leakage current
   RANGE Adrive, Vm : section specific
   RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
   stimon       : Stimulation state
   Adrive (kPa) : Stimulation amplitude
   gNabar = 0.04 (S/cm2)
   ENa = 55.0 (mV)
   gKdbar = 0.04 (S/cm2)
   EK = -90.0 (mV)
   gMbar = 0.00031 (S/cm2)
   gLeak = 0.0001 (S/cm2)
   ELeak = -52.94442210426802 (mV)
}

STATE {
   m : iNa activation gate
   h : iNa inactivation gate
   n : iKd activation gate
   l : iKd inactivation gate
   p : iM gate
}

ASSIGNED {
   v  (nC/cm2)
   Vm (mV)
   iNa (mA/cm2)
   iKd (mA/cm2)
   iM (mA/cm2)
   iLeak (mA/cm2)
}

FUNCTION_TABLE V(A(kPa), Q(nC/cm2)) (mV)
FUNCTION_TABLE alpham(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betam(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphah(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betah(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphan(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betan(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphal(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betal(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE pinf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE taup(A(kPa), Q(nC/cm2)) (ms)

INITIAL {
   m = alpham(Adrive * stimon, v) / (alpham(Adrive * stimon, v) + betam(Adrive * stimon, v))
   h = alphah(Adrive * stimon, v) / (alphah(Adrive * stimon, v) + betah(Adrive * stimon, v))
   n = alphan(Adrive * stimon, v) / (alphan(Adrive * stimon, v) + betan(Adrive * stimon, v))
   l = alphal(Adrive * stimon, v) / (alphal(Adrive * stimon, v) + betal(Adrive * stimon, v))
   p = pinf(Adrive * stimon, v)
}

BREAKPOINT {
   SOLVE states METHOD cnexp
   Vm = V(Adrive * stimon, v)
   iNa = gNabar * m * m * m * h * (Vm - ENa)
   iKd = gKdbar * n * n * n * l * (Vm - EK)
   iM = gMbar * p * (Vm - EK)
   iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
   m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
   h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
   n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
   l' = alphal(Adrive * stimon, v) * (1 - l) - betal(Adrive * stimon, v) * l
   p' = (pinf(Adrive * stimon, v) - p) / taup(Adrive * stimon, v)
}
TITLE RS membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a Cortical regular spiking neuron
upon electrical / ultrasonic stimulation, based on the SONIC model.

Reference: Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019).
Understanding ultrasound neuromodulation using a computationally efficient
and interpretable model of intramembrane cavitation. J. Neural Eng.

@Author: Theo Lemaire, EPFL
@Date: 2019-07-05
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
   SUFFIX RSauto
   NONSPECIFIC_CURRENT iNa : Sodium current
   NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
   NONSPECIFIC_CURRENT iM : slow non-inactivating Potassium current
   NONSPECIFIC_CURRENT iLeak : non-specific leakage current
   RANGE Adrive, Vm : section specific
   RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
   stimon       : Stimulation state
   Adrive (kPa) : Stimulation amplitude
   gNabar = 0.056 (S/cm2)
   ENa = 50.0 (mV)
   gKdbar = 0.006 (S/cm2)
   EK = -90.0 (mV)
   gMbar = 7.500000000000001e-05 (S/cm2)
   gLeak = 2.05e-05 (S/cm2)
   ELeak = -70.3 (mV)
}

STATE {
   m : iNa activation gate
   h : iNa inactivation gate
   n : iKd gate
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
FUNCTION_TABLE pinf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE taup(A(kPa), Q(nC/cm2)) (ms)

INITIAL {
   m = alpham(0, v) / (alpham(0, v) + betam(0, v))
   h = alphah(0, v) / (alphah(0, v) + betah(0, v))
   n = alphan(0, v) / (alphan(0, v) + betan(0, v))
   p = pinf(0, v)
}

BREAKPOINT {
   SOLVE states METHOD cnexp
   Vm = V(Adrive * stimon, v)
   iNa = gNabar * m * m * m * h * (Vm - ENa)
   iKd = gKdbar * n * n * n * n * (Vm - EK)
   iM = gMbar * p * (Vm - EK)
   iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
   m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
   h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
   n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
   p' = (pinf(Adrive * stimon, v) - p) / taup(Adrive * stimon, v)
}
TITLE LTS membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a Cortical low-threshold spiking neuron
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
   SUFFIX LTSauto
   NONSPECIFIC_CURRENT iNa : Sodium current
   NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
   NONSPECIFIC_CURRENT iM : slow non-inactivating Potassium current
   NONSPECIFIC_CURRENT iLeak : non-specific leakage current
   NONSPECIFIC_CURRENT iCaT : low-threshold (T-type) Calcium current
   RANGE Adrive, Vm : section specific
   RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
   stimon       : Stimulation state
   Adrive (kPa) : Stimulation amplitude
   gNabar = 0.05 (S/cm2)
   ENa = 50.0 (mV)
   gKdbar = 0.004 (S/cm2)
   EK = -90.0 (mV)
   gMbar = 2.8000000000000003e-05 (S/cm2)
   gLeak = 1.9e-05 (S/cm2)
   ELeak = -50.0 (mV)
   gCaTbar = 0.0004 (S/cm2)
   ECa = 120.0 (mV)
}

STATE {
   m : iNa activation gate
   h : iNa inactivation gate
   n : iKd gate
   p : iM gate
   s : iCaT activation gate
   u : iCaT inactivation gate
}

ASSIGNED {
   v  (nC/cm2)
   Vm (mV)
   iNa (mA/cm2)
   iKd (mA/cm2)
   iM (mA/cm2)
   iLeak (mA/cm2)
   iCaT (mA/cm2)
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
FUNCTION_TABLE sinf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE taus(A(kPa), Q(nC/cm2)) (ms)
FUNCTION_TABLE uinf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE tauu(A(kPa), Q(nC/cm2)) (ms)

INITIAL {
   m = alpham(Adrive * stimon, v) / (alpham(Adrive * stimon, v) + betam(Adrive * stimon, v))
   h = alphah(Adrive * stimon, v) / (alphah(Adrive * stimon, v) + betah(Adrive * stimon, v))
   n = alphan(Adrive * stimon, v) / (alphan(Adrive * stimon, v) + betan(Adrive * stimon, v))
   p = pinf(Adrive * stimon, v)
   s = sinf(Adrive * stimon, v)
   u = uinf(Adrive * stimon, v)
}

BREAKPOINT {
   SOLVE states METHOD cnexp
   Vm = V(Adrive * stimon, v)
   iNa = gNabar * m * m * m * h * (Vm - ENa)
   iKd = gKdbar * n * n * n * n * (Vm - EK)
   iM = gMbar * p * (Vm - EK)
   iLeak = gLeak * (Vm - ELeak)
   iCaT = gCaTbar * s * s * u * (Vm - ECa)
}

DERIVATIVE states {
   m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
   h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
   n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
   p' = (pinf(Adrive * stimon, v) - p) / taup(Adrive * stimon, v)
   s' = (sinf(Adrive * stimon, v) - s) / taus(Adrive * stimon, v)
   u' = (uinf(Adrive * stimon, v) - u) / tauu(Adrive * stimon, v)
}
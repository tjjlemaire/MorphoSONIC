TITLE IB membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a Cortical intrinsically bursting neuron
upon electrical / ultrasonic stimulation, based on the SONIC model.

Reference: Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019).
Understanding ultrasound neuromodulation using a computationally efficient
and interpretable model of intramembrane cavitation. J. Neural Eng.

@Author: Theo Lemaire, EPFL
@Date: 2019-07-04
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
   SUFFIX IB
   NONSPECIFIC_CURRENT iNa : Sodium current
   NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
   NONSPECIFIC_CURRENT iM : slow non-inactivating Potassium current
   NONSPECIFIC_CURRENT iLeak : non-specific leakage current
   NONSPECIFIC_CURRENT iCaL : high-threshold (L-type) Calcium current
   RANGE Adrive, Vm : section specific
   RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
   stimon       : Stimulation state
   Adrive (kPa) : Stimulation amplitude
   cm = 1.0 (uF/cm2)
   gNabar = 0.05 (S/cm2)
   ENa = 50.0 (mV)
   gKdbar = 0.005 (S/cm2)
   EK = -90.0 (mV)
   gMbar = 3e-05 (S/cm2)
   gLeak = 1e-05 (S/cm2)
   ELeak = -70 (mV)
   gCaLbar = 0.0001 (S/cm2)
   ECa = 120.0 (mV)
}

STATE {
   m : iNa activation gate
   h : iNa inactivation gate
   n : iKd gate
   p : iM gate
   q : iCaL activation gate
   r : iCaL inactivation gate
}

ASSIGNED {
   v  (nC/cm2)
   Vm (mV)
   iNa (mA/cm2)
   iKd (mA/cm2)
   iM (mA/cm2)
   iLeak (mA/cm2)
   iCaL (mA/cm2)
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
FUNCTION_TABLE alphaq(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betaq(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphar(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betar(A(kPa), Q(nC/cm2)) (/ms)

INITIAL {
   r = alphar(0, v) / (alphar(0, v) + betar(0, v))
   m = alpham(0, v) / (alpham(0, v) + betam(0, v))
   h = alphah(0, v) / (alphah(0, v) + betah(0, v))
   n = alphan(0, v) / (alphan(0, v) + betan(0, v))
   p = pinf(0, v)
   q = alphaq(0, v) / (alphaq(0, v) + betaq(0, v))
}

BREAKPOINT {
   SOLVE states METHOD cnexp
   Vm = V(Adrive * stimon, v)
   iKd = gKdbar * n * n * n * n * (Vm - EK)
   iCaL = gCaLbar * q * q * r * (Vm - ECa)
   iNa = gNabar * m * m * m * h * (Vm - ENa)
   iM = gMbar * p * (Vm - EK)
   iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
   r' = alphar(Adrive * stimon, v) * (1 - r) - betar(Adrive * stimon, v) * r
   m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
   h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
   n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
   p' = (pinf(Adrive * stimon, v) - p) / taup(Adrive * stimon, v)
   q' = alphaq(Adrive * stimon, v) * (1 - q) - betaq(Adrive * stimon, v) * q
}
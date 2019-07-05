TITLE TC membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a Thalamo-cortical neuron
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
   SUFFIX TCauto
   NONSPECIFIC_CURRENT iNa : Sodium current
   NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
   NONSPECIFIC_CURRENT iCaT : low-threshold (Ts-type) Calcium current
   NONSPECIFIC_CURRENT iLeak : non-specific leakage current
   NONSPECIFIC_CURRENT iKLeak : Potassium leakage current
   NONSPECIFIC_CURRENT iH : outward mixed cationic current
   RANGE Adrive, Vm : section specific
   RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
   stimon       : Stimulation state
   Adrive (kPa) : Stimulation amplitude
   gNabar = 0.09000000000000001 (S/cm2)
   ENa = 50.0 (mV)
   gKdbar = 0.01 (S/cm2)
   EK = -90.0 (mV)
   gCaTbar = 0.002 (S/cm2)
   ECa = 120.0 (mV)
   gLeak = 1e-05 (S/cm2)
   ELeak = -70.0 (mV)
   gKLeak = 1.3800000000000002e-05 (S/cm2)
   gHbar = 1.75e-05 (S/cm2)
   EH = -40.0 (mV)
   Cai_min = 5e-08 (M)
   taur_Cai = 5.0 (ms)
   current_to_molar_rate_Ca = 0.0005182136553443892 (1e7 mol.m-1.C-1)
   k2 = 0.0004 (/ms)
   k1 = 2.5e+19 (/ms)
   nCa = 4 ()
   k3 = 0.1 (/ms)
   k4 = 0.001 (/ms)
}

STATE {
   m : iNa activation gate
   h : iNa inactivation gate
   n : iKd gate
   s : iCaT activation gate
   u : iCaT inactivation gate
   Cai : submembrane Ca2+ concentration (M)
   P0 : proportion of unbound iH regulating factor
   C1 : iH gate closed state
   O1 : iH gate open state
}

ASSIGNED {
   v  (nC/cm2)
   Vm (mV)
   iNa (mA/cm2)
   iKd (mA/cm2)
   iCaT (mA/cm2)
   iLeak (mA/cm2)
   iKLeak (mA/cm2)
   iH (mA/cm2)
}

FUNCTION_TABLE V(A(kPa), Q(nC/cm2)) (mV)
FUNCTION_TABLE alpham(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betam(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphah(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betah(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphan(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betan(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE sinf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE taus(A(kPa), Q(nC/cm2)) (ms)
FUNCTION_TABLE uinf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE tauu(A(kPa), Q(nC/cm2)) (ms)
FUNCTION_TABLE betao(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphao(A(kPa), Q(nC/cm2)) (/ms)

FUNCTION OL(O, C) {
    OL = 1 - O - C
}

FUNCTION npow(x, n) {
    npow = x^n
}

FUNCTION P0inf(Cai) {
    P0inf = k2 / (k2 + k1 * npow(Cai, nCa))
}

FUNCTION Oinf(Cai, Vm) {
    Oinf = k4 / (k3 * (1 - P0inf(Cai)) + k4 * (1 + betao(0, v) / alphao(0, v)))
}

FUNCTION Cinf(Cai, Vm) {
    Cinf = betao(0, v) / alphao(0, v) * Oinf(Cai, Vm)
}

INITIAL {
   m = alpham(0, v) / (alpham(0, v) + betam(0, v))
   h = alphah(0, v) / (alphah(0, v) + betah(0, v))
   n = alphan(0, v) / (alphan(0, v) + betan(0, v))
   s = sinf(0, v)
   u = uinf(0, v)
   iCaT = gCaTbar * s * s * u * (Vm - ECa)
   Cai = Cai_min - taur_Cai * current_to_molar_rate_Ca * iCaT
   P0 = P0inf(Cai)
   C1 = Cinf(Cai, Vm)
   O1 = Oinf(Cai, Vm)
}

BREAKPOINT {
   SOLVE states METHOD cnexp
   Vm = V(Adrive * stimon, v)
   iNa = gNabar * m * m * m * h * (Vm - ENa)
   iKd = gKdbar * n * n * n * n * (Vm - EK)
   iCaT = gCaTbar * s * s * u * (Vm - ECa)
   iLeak = gLeak * (Vm - ELeak)
   iKLeak = gKLeak * (Vm - EK)
   iH = gHbar * (O1 + 2 * OL(O1, C1)) * (Vm - EH)
}

DERIVATIVE states {
   m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
   h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
   n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
   s' = (sinf(Adrive * stimon, v) - s) / taus(Adrive * stimon, v)
   u' = (uinf(Adrive * stimon, v) - u) / tauu(Adrive * stimon, v)
   Cai' = ((Cai_min - Cai) / taur_Cai - current_to_molar_rate_Ca * iCaT)
   P0' = k2 * (1 - P0) - k1 * P0 * npow(Cai, nCa)
   C1' = betao(Adrive * stimon, v) * O1 - alphao(Adrive * stimon, v) * C1
   O1' = (alphao(Adrive * stimon, v) * C1 - betao(Adrive * stimon, v) * O1 - k3 * O1 * (1 - P0) + k4 * (1 - O1 - C1))
}
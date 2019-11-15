TITLE STN membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a Sub-thalamic nucleus neuron
upon electrical / ultrasonic stimulation, based on the SONIC model.

Reference: Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019).
Understanding ultrasound neuromodulation using a computationally efficient
and interpretable model of intramembrane cavitation. J. Neural Eng.

@Author: Theo Lemaire, EPFL
@Date: 2019-07-17
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

CONSTANT {
   Z_Ca = 2
   FARADAY = 96485.3
   Rg = 8.31342
}

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
   SUFFIX STNauto
   NONSPECIFIC_CURRENT iNa : Sodium current
   NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
   NONSPECIFIC_CURRENT iA : A-type Potassium current
   NONSPECIFIC_CURRENT iCaT : low-threshold (T-type) Calcium current
   NONSPECIFIC_CURRENT iCaL : high-threshold (L-type) Calcium current
   NONSPECIFIC_CURRENT iKCa : Calcium-activated Potassium current
   NONSPECIFIC_CURRENT iLeak : non-specific leakage current
   RANGE Adrive, Vm : section specific
   RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
   stimon       : Stimulation state
   Adrive (kPa) : Stimulation amplitude
   gNabar = 0.049 (S/cm2)
   ENa = 60.0 (mV)
   gKdbar = 0.057 (S/cm2)
   EK = -90.0 (mV)
   gAbar = 0.005 (S/cm2)
   gCaTbar = 0.005 (S/cm2)
   Cao = 0.002 (M)
   T = 306.15 ()
   gCaLbar = 0.015000000000000001 (S/cm2)
   gKCabar = 0.001 (S/cm2)
   gLeak = 0.00035 (S/cm2)
   ELeak = -60.0 (mV)
   tau_d2 = 130.0 (ms)
   thetax_d2 = 1e-07 ()
   kx_d2 = 2e-08 ()
   tau_r = 2.0 (ms)
   thetax_r = 1.7e-07 ()
   kx_r = -8e-08 ()
   current_to_molar_rate_Ca = 5.062862990011234e-06 (1e7 mol.m-1.C-1)
   taur_Cai = 0.5 (ms)
   Cai0 = 5e-09 (M)
}

STATE {
   m : iNa activation gate
   h : iNa inactivation gate
   n : iKd gate
   a : iA activation gate
   b : iA inactivation gate
   p : iCaT activation gate
   q : iCaT inactivation gate
   c : iCaL activation gate
   d1 : iCaL inactivation gate 1
   d2 : iCaL inactivation gate 2
   r : iCaK gate
   Cai : submembrane Calcium concentration (M)
}

ASSIGNED {
   v  (nC/cm2)
   Vm (mV)
   iNa (mA/cm2)
   iKd (mA/cm2)
   iA (mA/cm2)
   iCaT (mA/cm2)
   iCaL (mA/cm2)
   iKCa (mA/cm2)
   iLeak (mA/cm2)
}

FUNCTION_TABLE V(A(kPa), Q(nC/cm2)) (mV)
FUNCTION_TABLE ainf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE taua(A(kPa), Q(nC/cm2)) (ms)
FUNCTION_TABLE binf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE taub(A(kPa), Q(nC/cm2)) (ms)
FUNCTION_TABLE cinf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE tauc(A(kPa), Q(nC/cm2)) (ms)
FUNCTION_TABLE d1inf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE taud1(A(kPa), Q(nC/cm2)) (ms)
FUNCTION_TABLE minf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE taum(A(kPa), Q(nC/cm2)) (ms)
FUNCTION_TABLE hinf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE tauh(A(kPa), Q(nC/cm2)) (ms)
FUNCTION_TABLE ninf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE taun(A(kPa), Q(nC/cm2)) (ms)
FUNCTION_TABLE pinf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE taup(A(kPa), Q(nC/cm2)) (ms)
FUNCTION_TABLE qinf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE tauq(A(kPa), Q(nC/cm2)) (ms)

FUNCTION nernst(z_ion, Cion_in, Cion_out, T) {
    nernst = (Rg * T) / (z_ion * FARADAY) * log(Cion_out / Cion_in) * 1e3
}

FUNCTION xinf(var, theta, k) {
    xinf = 1 / (1 + exp((var - theta) / k))
}

FUNCTION d2inf(Cai) {
    d2inf = xinf(Cai, thetax_d2, kx_d2)
}

FUNCTION rinf(Cai) {
    rinf = xinf(Cai, thetax_r, kx_r)
}

FUNCTION derCai(p, q, c, d1, d2, Cai, Vm) {
    LOCAL iCa_tot
    iCa_tot = iCaT + iCaL
    derCai = - current_to_molar_rate_Ca * iCa_tot - Cai / taur_Cai
}

FUNCTION Caiinf(p, q, c, d1, Vm) {
    Caiinf = Cai0
}

INITIAL {
   m = minf(Adrive * stimon, v)
   h = hinf(Adrive * stimon, v)
   n = ninf(Adrive * stimon, v)
   a = ainf(Adrive * stimon, v)
   b = binf(Adrive * stimon, v)
   p = pinf(Adrive * stimon, v)
   q = qinf(Adrive * stimon, v)
   c = cinf(Adrive * stimon, v)
   d1 = d1inf(Adrive * stimon, v)
   d2 = d2inf(Cai)
   r = rinf(Cai)
   Cai = Caiinf(p, q, c, d1, Vm)
}

BREAKPOINT {
   SOLVE states METHOD cnexp
   Vm = V(Adrive * stimon, v)
   iNa = gNabar * m * m * m * h * (Vm - ENa)
   iKd = gKdbar * n * n * n * n * (Vm - EK)
   iA = gAbar * a * a * b * (Vm - EK)
   iCaT = gCaTbar * p * p * q * (Vm - nernst(Z_Ca, Cai, Cao, T))
   iCaL = gCaLbar * c * c * d1 * d2 * (Vm - nernst(Z_Ca, Cai, Cao, T))
   iKCa = gKCabar * r * r * (Vm - EK)
   iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
   m' = (minf(Adrive * stimon, v) - m) / taum(Adrive * stimon, v)
   h' = (hinf(Adrive * stimon, v) - h) / tauh(Adrive * stimon, v)
   n' = (ninf(Adrive * stimon, v) - n) / taun(Adrive * stimon, v)
   a' = (ainf(Adrive * stimon, v) - a) / taua(Adrive * stimon, v)
   b' = (binf(Adrive * stimon, v) - b) / taub(Adrive * stimon, v)
   p' = (pinf(Adrive * stimon, v) - p) / taup(Adrive * stimon, v)
   q' = (qinf(Adrive * stimon, v) - q) / tauq(Adrive * stimon, v)
   c' = (cinf(Adrive * stimon, v) - c) / tauc(Adrive * stimon, v)
   d1' = (d1inf(Adrive * stimon, v) - d1) / taud1(Adrive * stimon, v)
   d2' = (d2inf(Cai) - d2) / tau_d2
   r' = (rinf(Cai) - r) / tau_r
   Cai' = derCai(p, q, c, d1, d2, Cai, Vm)
}
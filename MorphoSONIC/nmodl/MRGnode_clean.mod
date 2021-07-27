TITLE MRG model nodal membrane dynamics

COMMENT
Equations governing the nodal membrane dynamics of a motor axon fiber, based on the MRG model.

This mod file is based on the original AXNODE.mod file found on ModelDB
(https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=3810),
but the equations have been cleaned up to facilitate readibility.

Reference: McIntyre CC, Richardson AG, and Grill WM. Modeling the excitability
of mammalian nerve fibers: influence of afterpotentials on the recovery cycle.
Journal of Neurophysiology 87:995-1006, 2002.

@Author: Theo Lemaire, EPFL
@Date: 2020-03-25
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX MRGnode
	NONSPECIFIC_CURRENT inaf
	NONSPECIFIC_CURRENT inap
	NONSPECIFIC_CURRENT iks
	NONSPECIFIC_CURRENT il
	RANGE gnafbar, gnapbar, gksbar, gl, ena, ek, el
	RANGE Qm : section specific
	RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	stimon       : Stimulation state
	celsius			(degC)
	q10_mp
	q10_h
	q10_s
	gnafbar	= 3.0	(mho/cm2)
	gnapbar = 0.01	(mho/cm2)
	gksbar = 0.08	(mho/cm2)
	gl = 0.007		(mho/cm2)
	ena = 50.0		(mV)
	ek = -90.0		(mV)
	el = -90.0		(mV)
	mhshift = 3.	(mV)
	vtraub = -80.	(mV)
}

STATE {
	m h p s
}

ASSIGNED {
	Qm (nC/cm2)
	inaf	(mA/cm2)
	inap	(mA/cm2)
	iks		(mA/cm2)
	il      (mA/cm2)
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	Qm = v
	inaf = gnafbar * m * m * m * h * (v - ena)
	inap = gnapbar * p * p * p * (v - ena)
	iks = gksbar * s * (v - ek)
	il = gl * (v - el)
}

DERIVATIVE states {
	m' = alpham(v) * (1 - m) - betam(v) * m
	h' = alphah(v) * (1 - h) - betah(v) * h
	p' = alphap(v) * (1 - p) - betap(v) * p
	s' = alphas(v) * (1 - s) - betas(v) * s
}

INITIAL {
	q10_mp = 2.2^((celsius - 20) / 10)
	q10_h = 2.9^((celsius - 20) / 10)
	q10_s = 3.0^((celsius - 36)/ 10)
	m = alpham(v) / (alpham(v) + betam(v))
	h = alphah(v) / (alphah(v) + betah(v))
	p = alphap(v) / (alphap(v) + betap(v))
	s = alphas(v) / (alphas(v) + betas(v))
}

FUNCTION vtrap(x, y) {
    vtrap = x / (exp(x / y) - 1)
}

FUNCTION alpham (v(mV)) {
	alpham = q10_mp * 1.86 * vtrap(-(v + mhshift + 18.4), 10.3)
}

FUNCTION betam (v(mV)) {
	betam = q10_mp * 0.086 * vtrap(v + mhshift + 22.7, 9.16)
}

FUNCTION alphah (v(mV)) {
	alphah = q10_h * 0.062 * vtrap(v + mhshift + 111.0, 11.0)
}

FUNCTION betah (v(mV)) {
	betah = q10_h * 2.3 / (1 + exp(-(v + mhshift + 28.8) / 13.4))
}

FUNCTION alphap (v(mV)) {
	alphap = q10_mp * 0.01 * vtrap(-(v + 27.), 10.2)
}

FUNCTION betap (v(mV)) {
	betap = q10_mp * 0.00025 * vtrap(v + 34., 10.)
}

FUNCTION alphas (v(mV)) {
	alphas = q10_s * 0.3 / (1 + exp(-(v - vtraub - 27.) / 5.))
}

FUNCTION betas (v(mV)) {
	betas = q10_s * 0.03 / (1 + exp(-(v - vtraub + 10.) / 1.))
}

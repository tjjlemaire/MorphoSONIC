## On the use of the extracellular mechanism for myelinated neurites in *NEURON*

### *NEURON*'s default behavior

By default, *NEURON* does not explicitly model the extracellular field around a neurite. Concretely, that means that in any given section, the extracellular potential is assumed to be zero (i.e. the circuit's ground), and the intracellular potential thus equals the transmembrane potential, allowing to use a single variable `v` to compute transmembrane and intracellular currents.

### The extracellular mechanism

In cases where the neural dynamics is directly affected by an extracellular electric field, or influenced by incomplete seals in the myelin sheath along with current flow in the space between the myelin and the axon, the extracellular potential can no longer be neglected and must be explicitly modeled. This is done by inserting a so-called *extracellular* mechanism into concerned sections, which effectively introduces an additional double layer network between the neurite sections and the ground, in order to model the extracellular field.

This mechanism is useful for simulating the stimulation with extracellular electrodes, response in the presence of an extracellular potential boundary condition computed by some external program, leaky patch clamps, incomplete seals in the myelin sheath along with current flow in the space between the myelin and the axon. It is described below:

```text
             Ra
/`----o----'\/\/`----o----'\/\/`----o----'\/\/`----o----'\ vext + v           intracellular space
      |              |              |              |
     ---            ---            ---            ---
    |   |          |   |          |   |       cm |   | i_membrane             axolamellar membrane
     ---            ---            ---            ---
      |  xraxial[0]  |              |              |
/`----o----'\/\/`----o----'\/\/`----o----'\/\/`----o----'\ vext[0]            periaxonal space
      |              |              |              |
     ---            ---            ---            --- 
    |   |          |   |          |   |    xc[0] |   | xg[0]                  myelin 
     ---            ---            ---            ---
      |  xraxial[1]  |              |              |
/`----o----'\/\/`----o----'\/\/`----o----'\/\/`----o----'\ vext[1]            adjacent extracellular space
      |              |              |              |
     ---            ---            ---            ---     
    |   |          |   |          |   |          |   | xg[1]
    |  ---         |  ---         |  ---   xc[1] |  ---  
    |   -          |   -          |   -          |   - e_extracellular
     ---            ---            ---            ---     
      |              |              |              |
-------------------------------------------------------- ground
```
<br>

Each layer is composed of axial resistors (referred to as `xraxial[0]` and `xraxial[1]`) representing some level of longitudinal extracellular coupling along a cylindrical shell surrounding the neurite, and a transverse RC circuit composed of a capacitor (namely `xc[0]` and `xc[1]`) and conductor (namely `xg[0]` and `xg[1]`) representing some level of coupling between the extracellular space and the ground. Moreover, a voltage source is introduced in series with the resistor of the second network layer in order to model the time-and-space varying extracellular potential imposed by the user (`e_extracellular`).

Effectively, this additional circuitry introduces 2 additional "extracellular" voltage nodes (referred to as `vext[0]` and `vext[1]`) above each "intracellular" section node (now representing `v + vext[0]`), at which the the voltage equation is solved simultaneously by applying of Kirchoff's law.

Note that in this circuit, axial resistors are specified in units of resistance per unit length (MOhm/cm), and therefore correspond to a longitudinal resistivity (in Ohm.cm) divided by a specific cross-sectional area (in cm2), thus they only need to be integrated along a length separating two nodes in order to yield an absolute resistance value, (in Ohm).
Oppositely, transverse elements are specified in units per unit area (i.e. uF/cm2 and S/cm2), hence they must be integrated over a given transverse surface area (in cm2) in order to yield an absolute capacitance (in F) and conductance (in S). This has important implications, which will be discussed below.

### Simple use-case: unmyelinated neurite

In the case where one simply wishes to model the influence of an extracellular potential field on a unmyelinated neurite, the extracellular voltage just outside of each section can be controlled by:
- Setting both transverse capacitors (`xc[0]` and `xc[1]`) to zero, thereby eliminating capacitive currents in both extracellular layers.
- Setting both axial resistors (`xraxial[0]` and `xraxial[1]`) to a "pseudo-infinite" value (i.e. 1e10 Ohm/cm), thereby removing any kind of longitudinal coupling in the extracellular layers.
- Setting both transverse conductances (`xg[0]` and `xg[1]`) to a "pseudo-infinite" value (i.e. 1e10 S/cm2), thereby forcing `vext[1]` and `vext[0]` to equilibrate instantaneously with `e_extracellular`.

### More complex use-case: myelinated neurite

In the case where one wishes to model the influence of an extracellular potential field on a (partially) myelinated neurite and / or the longitudinal current flow in the periaxonal space, the situation becomes more complex.

Specifically, while parameters of the second layer (`xraxial[1]`, `xc[1]` and `xg[1]`) can still be set in such a way to ensure a direct synchronization of vext[1] to e_extracellular without longitudinal coupling, that is not true anymore for the first layer since:
- `xraxial[0]` now represents the longitudinal resistance per unit length in the periaxonal space
- `xc[0]` and `xg[0]` now represent the transverse capacitance and conductance per unit area of the myelin sheath

Hence, these parameters must be given finite impedance values.

In this situation, `xc[0]` and `xg[0]` are integrated over a given surface area meant to represent the myelin.
In theory, that area should be defined from the fiber outer diameter:
- `C_myelin = xc[0] * pi * fiberD * section.L`
- `G_myelin = xg[0] * pi * fiberD * section.L`

However, since myelin compartments are not explicitly represented in this model (they're only virtually represented by the use of the extracellular mechanism), they have no geometrical dimensions of their own. Consequently, *NEURON* relies here on the dimensions provided for the underlying neurite section to integrate myelin properties, i.e.:
- `C_myelin = xc[0] * pi * section.diam * section.L`
- `G_myelin = xg[0] * pi * section.diam * section.L`

While the length of a neurite compartment and its associated embedding myelin can be reasonably thought of as equal, that is not the case for their respective diameters. This is especially true for models in which a myelin of constant cross-sectional area is wrapped around a neurite of varying diameter, as is the case in multiple models.

Consequently, in order to ensure unicity of myelin conductance and capacitance across sections, one can think of two ways to proceed:
1. Correct the values of the myelin transverse intensive properties on a per section basis, to compensate for the difference in neurite and myelin outer diameters:
    - `xc[0] = c_myelin * fiberD / section.diam`
    - `xg[0] = g_myelin * fiberD / section.diam`
2. Set the diameter of all myelinated neurite sections to that of the outer fiber (`sec.diam = fiberD`), and adapt the values of their membrane capacitance, membrane conductance and internal longitudinal resistivity accordingly (as done in the original MRG model implementation):
    - `cm = c_membrane * section.diam / fiberD`
    - `g = g_membrane * section.diam / fiberD`
    - `Ra = rho_a * (fiberD / section.diam)^2`

Note that this correction problem does not apply to the extracellular axial resistors, since those are provided as resistances per unit length. Hence, they do not need to be integrated over a surface area but rather only along the given length (which is consistent between the neurite section and its surrounding myelin), in order to compute their equivalent extensive property.

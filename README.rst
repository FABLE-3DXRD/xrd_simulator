======================================
Gaussian models
======================================

Gaussian splatting [Kerbl2023] is a quite trendy set of algorithms for modeling and rendering 3D scenes from a series of images with varying viewpoint.
Par of the appeal of these new methods is that the reconstructed scenes can be rendered very efficiently with a rasterization-approach.

We can model a polycrystal as a set of Gaussian subgrains, and under a set approximations, each such subgrain will cause a 2D-Gaussian gaussian-shaped
diffraction peak on the detector. The approach has been demonstrated and tested already in a serial-crystallography setting [Brehm2023].

There are a number of peak-broadedning effects that we could consider to include, and as long as we assume that all of these have a Gaussian profile 
and are small enough that non-linear terms can be discarded, they will continue to produce a Gaussian spot on the detector.

* Grain size
* Detector point-spread
* Incidident beam bandwidth
* Incident beam angular divergence
* lattice misorientation spread (aka. mosaicity)
* Strain-broadening (probably modeled by dislocation-concentrations and contrast factors)

To limit the scope here, I implement a model with Grain-size and mosaicity only. But I will implement anisotropic misorientation densities.
The approach to derive and implement the model with more effect is essentially the same, but if you have several very-small terms, you can get into numerical
stability problems.

The geometric model of XRD
--------------------------

Given an incident beam descibed by some phase-space density, :math:`p(\mathbf{k})`, centered on a point
:math:`\mathbf{k}_0` with length :math:`2\pi/\lambda`. And given a crystallite with a "reciprocal space map" 
:math:`f(\mathbf{q})` around a specific reciprocal lattice vector :math:`G_0`. We want to compute 
the phase-space distribution of the scattered beam which is given by an integral:

..math::
   I(\mathbf{p}) = \int \mathrm{d}\mathbf{k}\int \mathrm{d}\mathbf{q}
   p(\mathbf{k})f(\mathbf{q})\delta(|\mathbf{k}|-|\mathbf{p}|)\delta{\mathbf{k}+\mathbf{q}-\mathbf{p}}
   


.. rubric:: References

.. [Kerbl2023] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering, 2023.
.. [Brehm2023] Brehm, W., White, T. & Chapman, H. N. (2023). Crystal diffraction prediction and partiality estimation using Gaussian basis functions. Acta Cryst. A79
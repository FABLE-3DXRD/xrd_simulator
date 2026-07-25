Gaussian models
===============

Gaussian splatting [Kerbl2023] is a quite trendy set of algorithms for modeling and rendering 3D scenes from a series of images with varying viewpoint.
Par of the appeal of these new methods is that the reconstructed scenes can be rendered very efficiently with a rasterization-approach.

We can model a polycrystal as a set of Gaussian subgrains, and under a set approximations, each such subgrain will cause a 2D-Gaussian gaussian-shaped
diffraction peak on the detector. The approach has been demonstrated and tested already in a serial-crystallography setting [Brehm2023].

There are a number of peak-broadedning effects that we could consider to include. As long as we assume that all of these have a Gaussian profile and are small enough that non-linear terms can be discarded, they will produce a Gaussian spot on the detector.

* Grain size
* Detector point-spread
* Incidident beam bandwidth
* Incident beam angular divergence
* lattice misorientation spread (aka. mosaicity)
* Strain-broadening (probably modeled by dislocation-concentrations and contrast factors)

To limit the scope here, I implement a model with grain-size and mosaicity only.

Scattering theory
-----------------

Given an incident beam descibed by some phase-space density, $p(\mathbf{k})$, centered on a point
$\mathbf{k}_0$ with length $k=2\pi/\lambda_0$. And given a crystallite with a "reciprocal space map" (RSM)
$f(\mathbf{q})$ around a specific reciprocal lattice vector $\mathbf{G}_0$. We want to compute 
the phase-space distribution of the scattered beam which is given by an integral:

$$
   I(\mathbf{p}) = \int \mathrm{d}\mathbf{k}\int \mathrm{d}\mathbf{q}
    p(\mathbf{k})f(\mathbf{q})\delta(|\mathbf{k}|-|\mathbf{p}|)\delta(\mathbf{k}+\mathbf{q}-\mathbf{p})
$$

the two delta-Dirac function ensure energy- and momentum conservation respectively. 

You can plug in various combinations of gaussians and delta-Dirac function in for 
the two functions and go to town, but for a little bit of extra phyical understanding we
introduce a specific choise of basis vectors.

$$
   \mathbf{k} = \mathbf{k}_0 + \varepsilon\hat{\mathbf{\mathbf{k}_0}}
    + \zeta_{||}\hat{\mathbf{k}_{||}}
     + \zeta_\perp\hat{\mathbf{k}_\perp}
$$

where hat denoes the normalized vector. $\hat{\mathbf{k}_{||}}$ is 
a vector orthogonal to $\mathbf{k}_0$ that lies in the span of $\mathbf{G}$ and $\mathbf{k}\_0$ with the sign chosen such that $\mathbf{G}\cdot\hat{\mathbf{k}_{||}}>0$ . The last unit vector completes a right hand basis $\hat{\mathbf{k}_\perp}=\hat{\mathbf{k}_0}\times\hat{\mathbf{k}_{||}}$ .

We compute a nominal scattering vector $\theta_0=\arcsin(|\mathbf{G}_0|/2k)$ and

$$
   \mathbf{Q} = 2k\sin\theta_0[\cos\theta_0 \hat{\mathbf{k}_{||}} - \sin\theta_0\hat{\mathbf{k}_0}] = 2k\sin\theta_0\hat{\mathbf{Q}}
$$

Importantly this vector is not quite equal to $\mathbf{G}$ but should be close the difference
between the two is to first order parrallel to the unit-vector $\hat{\mathbf{q}}_{\mathrm{rock}} = [\cos\theta_0\hat{\mathbf{k}_0} + \sin\theta_0 \hat{\mathbf{k}_{||}}]$

which completes the basis for $\mathbf{q}$:

$$
   \mathbf{q} = \mathbf{G}_0 + q_{\mathrm{rock}}\hat{\mathbf{q}_{\mathrm{rock}}}
    + q_{\mathrm{strain}}\hat{\mathbf{Q}}
     + q_{\mathrm{roll}}\hat{\mathbf{k}_\perp} \\\\
     \approx \mathbf{Q} + (q_{\mathrm{rock}} - \delta q)\hat{\mathbf{q}_{\mathrm{rock}}}
      + q_{\mathrm{strain}}\hat{\mathbf{Q}}
       + q_{\mathrm{roll}}\hat{\mathbf{k}_\perp}
$$


where $\delta q = (\mathbf{Q} - \mathbf{G})\cdot\hat{\mathbf{q}_{\mathrm{rock}}}$ is a measure of how far the reflection is out of alignment. 

Now finally we can choose a parametrization of the outgoing ray. First we define $\mathbf{k}_h = \mathbf{k}_0 + \mathbf{Q}$
which by construction is $\mathbf{k}_h = k [\cos2\theta_0 \hat{\mathbf{k}_0} + \sin2\theta_0\hat{\mathbf{k}_{||}}]$.
Again we need to final unit vector normal to this: $\hat{\mathbf{k}_{\mathrm{up}}} = [\cos2\theta_0\hat{\mathbf{k}_{||}}-\sin2\theta_0 \hat{\mathbf{k}_0}]$.

$$
   \mathbf{p} = \mathbf{k}_h + \varepsilon'\hat{\mathbf{\mathbf{k}_h}}
    + \psi_{\mathrm{rad}}\hat{\mathbf{k}_{\mathrm{rad}}}
     + \psi_{\mathrm{azim}}\hat{\mathbf{k}_\perp}
$$

Assuming all deviations from the nominal directions are small, we see that $|\mathbf{k}|\approx k+\epsilon$ 
and $|\mathbf{p}|\approx k+\epsilon'$ so the energy-conservation integral can be removed by
setting $\varepsilon = \varepsilon'$ and raising an corresponding integral.

In practive we are not interested in the energy-distribution of the outgoing wave, 
so we will integrate out the outgoing energy also.

The momentum-conservation factor can be used to raise either the $\mathbf{k}$ or the $\mathbf{q}$
integral. The one you don't raise will become your integration variable, so depeding on what
model you want to construct one choise might be better that the other.

In either case, the equations enforced by momentum conservation is

$$
   \mathbf{q} = \mathbf{p} - \mathbf{k} \\\\
   \Leftrightarrow
    (q_{\mathrm{rock}} - \delta q)\hat{\mathbf{q}}_{\mathrm{rock}}
      + q_{\mathrm{strain}}\hat{\mathbf{Q}}
       + q_{\mathrm{roll}}\hat{\mathbf{k}_\perp} = 
    2\sin\theta_0 \varepsilon \hat{\mathbf{Q}}
    + \psi_{\mathrm{rad}}\hat{\mathbf{k}_{\mathrm{rad}}}
     + \psi_{\mathrm{azim}}\hat{\mathbf{k}_\perp}
      + \zeta_{||}\hat{\mathbf{k}_{||}}
       + \zeta_\perp\hat{\mathbf{k}_\perp}
$$    


In the simple model I choose, the incident beams is monochromatic and
collimated ($\varepsilon = \zeta_{||} = \zeta_{\perp} = 0$) and there
is no strain-broadening $q_{\mathrm{strain}}=0$ leading to a significant simplification:

$$
    (q_{\mathrm{rock}} - \delta q)\hat{\mathbf{q}_{\mathrm{rock}}}
       + q_{\mathrm{roll}}\hat{\mathbf{k}_\perp} = 
    + \psi_{||}\hat{\mathbf{k}_{\mathrm{up}}}
     + \psi_{\mathrm{azim}}\hat{\mathbf{k}_\perp}
$$

which can be rearanged to the three scalar equations:

$$
    q_{\mathrm{roll}} = \psi_{\mathrm{azim}} \text{ and } q_{\mathrm{rock}} = \delta q \text{ and } \psi_{||}=0
$$


So the scattered beam is simply a 1D Gaussian that samples the RSM though a line offset by $\delta q$ from the center.

Computing the RSM from an anisotropic Gaussian texture model
------------------------------------------------------------

Some more derivations...

Testing
-------

I simulate a 1 degree rotation of a single crystal of quartz in the shape of a symmetric tetrahedron. The crystal is much larger than the pixels and  the largest scattering angles are over 90 degrees to see the perspective effect at large angles.

The gaussian simulation uses seven gaussians to approximate the tetragedron (on symmetric in the center and six prolate ones along the edges) and has low misorientation.

The position and shapes of the peaks match well. The gaussian model includes some extra weak peaks because it is integrating a gaussian shaped time-window where the tetrahedron model is integrating a top hat time window.

![image](docs/_static/single_crystal_quartz.png)



Potential improvements
----------------------

Include incident beam divergence and bandwidth.

Improve performance of the rasterizer.


### References


[Kerbl2023] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering, 2023.

[Brehm2023] Brehm, W., White, T. & Chapman, H. N. (2023). Crystal diffraction prediction and partiality estimation using Gaussian basis functions. Acta Cryst. A79
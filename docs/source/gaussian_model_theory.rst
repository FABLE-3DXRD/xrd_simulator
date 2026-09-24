:orphan:

.. _gaussian-model-theory:

Gaussian model: theory and references
=====================================

This page describes the scattering model and approximations underlying
:mod:`xrd_simulator.gaussian_crystal_model`. A worked example is provided in
the module documentation.

.. contents:: On this page
   :local:
   :depth: 2

Gaussian subgrain representation
--------------------------------

Gaussian splatting represents a three-dimensional scene using Gaussian
primitives whose projected contributions can be rendered efficiently by
rasterization [Kerbl2023]_. Here, a polycrystal is represented by Gaussian
grains or subgrains, each with a Gaussian real-space density profile and a
narrow Gaussian distribution of lattice orientations. Under the approximations
below, each subgrain produces Gaussian-shaped diffraction spots on the
detector. Related Gaussian-basis methods for diffraction prediction and
partiality estimation have been demonstrated in serial crystallography
[Brehm2023]_.

Several effects can contribute to diffraction-spot broadening: grain size,
detector point-spread, incident-beam bandwidth, incident-beam angular
divergence, lattice misorientation (mosaicity), and distributions of strain.
When these effects have Gaussian profiles and their deviations are small
enough to neglect nonlinear terms, they can be combined within a Gaussian
model.

The model described here includes the real-space extent of the illuminated
subgrains and their mosaicity. The incident beam is monochromatic and
collimated, and each subgrain has a uniform strain tensor. A uniform strain
can shift a reflection but does not describe a distribution of lattice
spacings within that subgrain. Small sample rotations during an exposure are
approximated by additional Gaussian orientation broadening.

.. note::

   The Gaussian-splat rendering path does not apply a separate detector
   point-spread convolution. The ``gaussian_sigma`` and
   ``max_gaussian_kernel_radius`` detector settings retained in the worked
   example therefore do not add this broadening to its Gaussian spots.

Scattering theory
-----------------

Scattered-beam distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let the incident beam have a phase-space density :math:`p(\mathbf{k})`
centred on :math:`\mathbf{k}_0`, with nominal wavenumber

.. math::
   :label: gaussian-wavenumber

   k = \lvert\mathbf{k}_0\rvert = \frac{2\pi}{\lambda_0}.

For a particular reflection, let :math:`f(\mathbf{q})` describe the subgrain's
reciprocal-space map (RSM) around a reciprocal-lattice vector
:math:`\mathbf{G}_0`. The scattered-beam distribution is written as

.. math::
   :label: gaussian-scattering-integral

   I(\mathbf{p})
   = \int \mathrm{d}^{3}\mathbf{k}
     \int \mathrm{d}^{3}\mathbf{q}\;
     p(\mathbf{k})\,f(\mathbf{q})\,
     \delta\!\left(\lvert\mathbf{k}\rvert-\lvert\mathbf{p}\rvert\right)
     \delta^{(3)}\!\left(\mathbf{k}+\mathbf{q}-\mathbf{p}\right).

The scalar Dirac delta enforces elastic scattering, while the
three-dimensional Dirac delta enforces momentum conservation.

Incident-beam coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~

Following the coordinate construction used by Poulsen et al.
[Poulsen2017]_, write the incident wavevector as

.. math::
   :label: gaussian-incident-wavevector

   \mathbf{k}
   = \mathbf{k}_0
     + k\left(
         \varepsilon\,\hat{\mathbf{k}}_0
         + \zeta_{\parallel}\,\hat{\mathbf{k}}_{\parallel}
         + \zeta_{\perp}\,\hat{\mathbf{k}}_{\perp}
       \right).

A hat denotes a unit vector. The vector
:math:`\hat{\mathbf{k}}_{\parallel}` is perpendicular to
:math:`\mathbf{k}_0` and lies in the plane spanned by :math:`\mathbf{k}_0`
and :math:`\mathbf{G}_0`. Its sign is chosen so that
:math:`\mathbf{G}_0\cdot\hat{\mathbf{k}}_{\parallel}>0`. Complete the
right-handed basis with

.. math::

   \hat{\mathbf{k}}_{\perp}
   = \hat{\mathbf{k}}_0\times\hat{\mathbf{k}}_{\parallel}.

The coordinate :math:`\varepsilon` describes the relative wavenumber
deviation, and :math:`\zeta_{\parallel}` and :math:`\zeta_{\perp}` describe
small angular deviations of the incident beam.

Reciprocal-space coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define the nominal Bragg angle and the exactly aligned scattering vector by

.. math::
   :label: gaussian-nominal-scattering-vector

   \begin{aligned}
   \theta_0
   &= \arcsin\!\left(\frac{\lvert\mathbf{G}_0\rvert}{2k}\right), \\
   \mathbf{Q}
   &= 2k\sin\theta_0\left(
       \cos\theta_0\,\hat{\mathbf{k}}_{\parallel}
       - \sin\theta_0\,\hat{\mathbf{k}}_0
     \right) \\
   &= 2k\sin\theta_0\,\hat{\mathbf{Q}}.
   \end{aligned}

The vectors :math:`\mathbf{G}_0` and :math:`\mathbf{Q}` coincide when the
reflection is exactly aligned. For a small misalignment, their difference is
parallel, to first order, to the rocking direction

.. math::

   \hat{\mathbf{q}}_{\mathrm{rock}}
   = \cos\theta_0\,\hat{\mathbf{k}}_0
     + \sin\theta_0\,\hat{\mathbf{k}}_{\parallel}.

Use dimensionless rocking, strain, and rolling coordinates to write

.. math::
   :label: gaussian-reciprocal-coordinates

   \begin{aligned}
   \mathbf{q}
   &= \mathbf{G}_0
      + 2k\sin\theta_0\left(
          q_{\mathrm{rock}}\,\hat{\mathbf{q}}_{\mathrm{rock}}
          + q_{\mathrm{strain}}\,\hat{\mathbf{Q}}
          + q_{\mathrm{roll}}\,\hat{\mathbf{k}}_{\perp}
        \right) \\
   &\approx \mathbf{Q}
      + 2k\sin\theta_0\left(
          (q_{\mathrm{rock}}-\delta q)\,
              \hat{\mathbf{q}}_{\mathrm{rock}}
          + q_{\mathrm{strain}}\,\hat{\mathbf{Q}}
          + q_{\mathrm{roll}}\,\hat{\mathbf{k}}_{\perp}
        \right),
   \end{aligned}

where

.. math::
   :label: gaussian-rocking-offset

   \delta q
   = \frac{
       (\mathbf{Q}-\mathbf{G}_0)\cdot\hat{\mathbf{q}}_{\mathrm{rock}}
     }{2k\sin\theta_0}.

This offset measures the reflection's misalignment and is equal to the
rocking angle to first order, with the sign convention above. The incident
coordinates :math:`(\varepsilon,\zeta_{\parallel},\zeta_{\perp})` and the
reciprocal-space coordinates
:math:`(q_{\mathrm{rock}},q_{\mathrm{strain}},q_{\mathrm{roll}})` separate
bandwidth, collimation, misorientation, and strain effects.

Outgoing-beam coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~

The nominal scattered wavevector is

.. math::
   :label: gaussian-nominal-outgoing-wavevector

   \mathbf{k}_h
   = \mathbf{k}_0+\mathbf{Q}
   = k\left(
       \cos(2\theta_0)\,\hat{\mathbf{k}}_0
       + \sin(2\theta_0)\,\hat{\mathbf{k}}_{\parallel}
     \right).

Define the radial direction perpendicular to :math:`\mathbf{k}_h` by

.. math::

   \hat{\mathbf{k}}_{\mathrm{rad}}
   = \cos(2\theta_0)\,\hat{\mathbf{k}}_{\parallel}
     - \sin(2\theta_0)\,\hat{\mathbf{k}}_0.

The outgoing wavevector can then be written as

.. math::
   :label: gaussian-outgoing-wavevector

   \mathbf{p}
   = \mathbf{k}_h
     + k\left(
         \varepsilon'\,\hat{\mathbf{k}}_h
         + \psi_{\mathrm{rad}}\,\hat{\mathbf{k}}_{\mathrm{rad}}
         + \psi_{\mathrm{azim}}\,\hat{\mathbf{k}}_{\perp}
       \right).

Here :math:`\psi_{\mathrm{rad}}` and :math:`\psi_{\mathrm{azim}}` are small
transverse angular deviations from the nominal outgoing direction.

Linearized conservation equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To first order,

.. math::

   \lvert\mathbf{k}\rvert\approx k(1+\varepsilon),
   \qquad
   \lvert\mathbf{p}\rvert\approx k(1+\varepsilon').

Energy conservation therefore sets :math:`\varepsilon'=\varepsilon`.
Momentum conservation gives :math:`\mathbf{q}=\mathbf{p}-\mathbf{k}` and,
after substitution,

.. math::
   :label: gaussian-linearized-momentum

   \begin{aligned}
   &2\sin\theta_0\Bigl[
       (q_{\mathrm{rock}}-\delta q)\,
           \hat{\mathbf{q}}_{\mathrm{rock}}
       + q_{\mathrm{strain}}\,\hat{\mathbf{Q}}
       + q_{\mathrm{roll}}\,\hat{\mathbf{k}}_{\perp}
   \Bigr] \\
   &\qquad = 2\sin\theta_0\,\varepsilon\,\hat{\mathbf{Q}}
       + \psi_{\mathrm{rad}}\,\hat{\mathbf{k}}_{\mathrm{rad}} \\
   &\qquad\quad
       + \psi_{\mathrm{azim}}\,\hat{\mathbf{k}}_{\perp}
       - \zeta_{\parallel}\,\hat{\mathbf{k}}_{\parallel}
       - \zeta_{\perp}\,\hat{\mathbf{k}}_{\perp}.
   \end{aligned}

.. note::

   The incident-divergence terms have minus signs because the incident
   wavevector is subtracted in :math:`\mathbf{q}=\mathbf{p}-\mathbf{k}`.
   This corrects the corresponding signs in the original workflow. It does
   not affect the monochromatic, collimated specialization below, where
   both terms vanish.

Monochromatic, collimated specialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current model assumes

.. math::

   \varepsilon=\zeta_{\parallel}=\zeta_{\perp}=0,
   \qquad q_{\mathrm{strain}}=0.

The last condition excludes strain-distribution broadening; it does not
require the uniform subgrain strain to vanish. The conservation equation
reduces to

.. math::

   \begin{aligned}
   &2\sin\theta_0\left[
       (q_{\mathrm{rock}}-\delta q)\,
           \hat{\mathbf{q}}_{\mathrm{rock}}
       + q_{\mathrm{roll}}\,\hat{\mathbf{k}}_{\perp}
   \right] \\
   &\qquad
   = \psi_{\mathrm{rad}}\,\hat{\mathbf{k}}_{\mathrm{rad}}
     + \psi_{\mathrm{azim}}\,\hat{\mathbf{k}}_{\perp}.
   \end{aligned}

For a nondegenerate Bragg geometry, the three coordinate equations are

.. math::
   :label: gaussian-scattering-constraints

   q_{\mathrm{rock}}=\delta q,
   \qquad
   \psi_{\mathrm{rad}}=0,
   \qquad
   \psi_{\mathrm{azim}}=2\sin\theta_0\,q_{\mathrm{roll}}.

Thus, the scattered angular distribution samples the RSM along a line
parallel to :math:`\hat{\mathbf{k}}_{\perp}`, offset by :math:`\delta q`
from its centre. For a Gaussian RSM, this gives a one-dimensional Gaussian
angular distribution. The finite real-space extent of the illuminated
subgrain is subsequently projected onto the detector to produce a
two-dimensional spot.

Anisotropic Gaussian orientation distributions
----------------------------------------------

Tangent-space approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider a narrow orientation distribution centred on an orientation
:math:`g_0`. Approximate orientation space by its left tangent space at
:math:`g_0`, so that a small rotation vector :math:`\mathbf{r}` acts in
laboratory coordinates:

.. math::
   :label: gaussian-tangent-rotation

   g\approx R_{\mathbf{r}}g_0
   \approx
   \begin{bmatrix}
       1    & -r_z & r_y \\
       r_z  & 1    & -r_x \\
       -r_y & r_x  & 1
   \end{bmatrix}g_0.

The Gaussian orientation density used in the workflow is

.. math::
   :label: gaussian-orientation-density

   f(\mathbf{r})
   = \frac{2}{\sqrt{\pi}}\sqrt{\det\mathbf{T}}\,
     \exp\!\left(-\mathbf{r}^{\mathsf{T}}\mathbf{T}\mathbf{r}\right),

where :math:`\mathbf{T}` is a symmetric positive-definite concentration
tensor.

.. note::

   The prefactor above retains the workflow's normalization convention.
   With ordinary Euclidean tangent-space measure,
   :math:`\int f(\mathbf{r})\,\mathrm{d}^{3}\mathbf{r}=2\pi`, not one.
   It should therefore not be interpreted as a unit-normalized Euclidean
   probability density without rescaling.

   For an exponential of the form
   :math:`\exp(-\mathbf{r}^{\mathsf{T}}\mathbf{T}\mathbf{r})`, the
   corresponding normalized Gaussian has covariance
   :math:`\tfrac{1}{2}\mathbf{T}^{-1}`. The implementation directly inverts
   ``misorientation_tensor`` to obtain :math:`\mathbf{T}`. Consequently,
   the square roots of the input tensor's eigenvalues are Gaussian width
   parameters, not standard deviations. For an input eigenvalue
   :math:`a^2`, the standard deviation is :math:`a/\sqrt{2}`.
   The real-space Gaussian uses the same exponential convention for
   ``shape_tensor``. This distinction matters because the class attribute
   descriptions currently use the word "covariance" for these inputs.

Pole density as a line integral
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Miller indices :math:`(h,k,\ell)`, define the unit lattice direction in
crystal coordinates by

.. math::

   \mathbf{h}
   = \frac{
       \mathbf{B}_0[h,k,\ell]^{\mathsf{T}}
     }{
       \left\lvert\mathbf{B}_0[h,k,\ell]^{\mathsf{T}}\right\rvert
     },
   \qquad
   \mathbf{p}=g_0\mathbf{h}.

Here :math:`\mathbf{B}_0` is the reciprocal-lattice basis matrix. In this
section, :math:`\mathbf{p}` denotes the mean lattice direction, rather than
the outgoing wavevector used in the scattering section. It is parallel to
:math:`\mathbf{G}_0`. Let :math:`\mathbf{y}` be a unit direction in
laboratory coordinates, parallel to a sampled reciprocal-space vector
:math:`\mathbf{q}`; it is not a Cartesian coordinate-axis label.

The pole density, also called the pair-correlation function in texture
analysis, describes the density of lattice directions along
:math:`\mathbf{y}`. Its calculation normally involves an integral over a
circle in :math:`\mathrm{SO}(3)`. Within the tangent-space approximation,
replace this by an infinite line integral, parameterized as

.. math::
   :label: gaussian-orientation-fibre

   \mathbf{r}(\lambda)
   = \frac{\mathbf{p}\times\mathbf{y}}{\mathbf{p}\cdot\mathbf{y}}
     + \lambda\mathbf{p}
   = \mathbf{r}_0+\lambda\mathbf{p}.

The parameter :math:`\lambda` here is a line-integration parameter, not the
X-ray wavelength. Substitution into the Gaussian density and completion of
the square give, with
:math:`d=\mathbf{p}^{\mathsf{T}}\mathbf{T}\mathbf{p}`,

.. math::
   :label: gaussian-pole-density-integral

   \begin{aligned}
   A(\mathbf{y},\mathbf{p};f)
   &= \int_{-\infty}^{\infty}
        f\!\left(\mathbf{r}_0+\lambda\mathbf{p}\right)
        \,\mathrm{d}\lambda \\
   &= \frac{2\sqrt{\det\mathbf{T}}}
           {\sqrt{d}}
      \exp\!\left[
          -\mathbf{r}_0^{\mathsf{T}}\mathbf{T}\mathbf{r}_0
          + \frac{
              (\mathbf{r}_0^{\mathsf{T}}\mathbf{T}\mathbf{p})^2
            }{d}
      \right].
   \end{aligned}

Projected concentration tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For directions close to the pole, make the further approximation
:math:`\mathbf{p}\cdot\mathbf{y}\approx1`. Define

.. math::

   \mathbf{M}
   = \mathbf{T}
     - \frac{
         \mathbf{T}\mathbf{p}\mathbf{p}^{\mathsf{T}}\mathbf{T}
       }{d}.

Let :math:`[\mathbf{p}]_{\times}` denote the cross-product matrix,
:math:`[\mathbf{p}]_{\times}\mathbf{y}=\mathbf{p}\times\mathbf{y}`. Then

.. math::
   :label: gaussian-projected-concentration

   \mathbf{T}_{\mathbf{p}}
   = [\mathbf{p}]_{\times}^{\mathsf{T}}
       \mathbf{M}
       [\mathbf{p}]_{\times},

and the pole density becomes

.. math::
   :label: gaussian-pole-density

   A(\mathbf{y},\mathbf{p};f)
   \approx \frac{2\sqrt{\det\mathbf{T}}}{\sqrt{d}}\,
     \exp\!\left(
         -\mathbf{y}^{\mathsf{T}}\mathbf{T}_{\mathbf{p}}\mathbf{y}
     \right).

Equivalently, using the Levi-Civita symbol :math:`\epsilon_{ijk}` and
summation over repeated indices,

.. math::

   [\mathbf{T}_{\mathbf{p}}]_{ij}
   = p_k\,\epsilon_{lki}
     \left(
       T_{lm}-\frac{T_{la}p_a p_b T_{bm}}{d}
     \right)
     \epsilon_{mnj}\,p_n.

This absorbs the cross products into the projected tensor and makes the
subsequent Gaussian expressions more compact.

Reciprocal-space map
~~~~~~~~~~~~~~~~~~~~

The pole density is defined for unit-vector arguments. With no
strain-distribution broadening, extend it to a reciprocal-space map supported
on the local plane :math:`q_{\mathrm{strain}}=0`.

Introduce the two-column basis matrix and coordinate vector

.. math::

   \mathbf{U}
   = \begin{bmatrix}
       \hat{\mathbf{q}}_{\mathrm{rock}} & \hat{\mathbf{k}}_{\perp}
     \end{bmatrix},
   \qquad
   \mathbf{u}
   = \begin{bmatrix}
       q_{\mathrm{rock}} \\
       q_{\mathrm{roll}}
     \end{bmatrix}.

In the dimensionless reciprocal-space coordinates defined above, the
workflow's local RSM is

.. math::
   :label: gaussian-local-rsm

   f(\mathbf{q})
   \approx \delta(q_{\mathrm{strain}})\,
      \frac{2\sqrt{\det\mathbf{T}}}{\sqrt{d}}\,
      \exp\!\left[
          -\mathbf{u}^{\mathsf{T}}
           \left(\mathbf{U}^{\mathsf{T}}
                 \mathbf{T}_{\mathbf{p}}\mathbf{U}\right)
           \mathbf{u}
      \right].

The Dirac delta expresses the absence of a distribution of lattice spacings
within the subgrain. Sampling this Gaussian at
:math:`q_{\mathrm{rock}}=\delta q`, as required by
:eq:`gaussian-scattering-constraints`, gives the scattered angular
distribution for the reflection.

Illustrative tests
------------------

Comparison with a tetrahedral crystal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The workflow compares a one-degree rotation of a single quartz crystal
shaped as a symmetric tetrahedron with a Gaussian approximation of the same
crystal. The crystal is much larger than the detector pixels, and the largest
scattering angles exceed 60 degrees, making radial spot elongation from the
projection geometry visible.

Seven Gaussians approximate the tetrahedron: one isotropic Gaussian at its
centre and six prolate Gaussians along its edges. The Gaussian model uses a
small orientation spread.

The comparison shows similar peak positions and shapes. The Gaussian model
also produces additional weak peaks because its rotational integration uses
a Gaussian time window, whereas the tetrahedral model uses a top-hat time
window.

.. figure:: ../_static/single_crystal_quartz.png
   :alt: Quartz diffraction patterns from tetrahedral and Gaussian models.
   :align: center
   :width: 100%

   Comparison of the tetrahedral crystal model with a seven-Gaussian
   approximation during a one-degree sample rotation.

High-resolution rocking curve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A second test resolves a rocking curve and checks the shape of the resulting
three-dimensional reciprocal-space map. The workflow figure compares the
simulated profiles with the expected contributions from grain size and
misorientation.

.. figure:: ../_static/testing_rocking_curves.png
   :alt: Reciprocal-space map and profiles from a high-resolution rocking scan.
   :align: center
   :width: 100%

   High-resolution rocking-curve test showing the reciprocal-space map and
   the contributions of grain size and misorientation to the simulated
   profiles.

References
----------

.. [Kerbl2023] Kerbl, B., Kopanas, G., Leimkühler, T. and Drettakis, G.
   (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering.
   *ACM Transactions on Graphics*, **42**\ (4).
   `doi:10.1145/3592433 <https://doi.org/10.1145/3592433>`_.

.. [Brehm2023] Brehm, W., White, T. and Chapman, H. N. (2023).
   Crystal diffraction prediction and partiality estimation using Gaussian
   basis functions. *Acta Crystallographica Section A: Foundations and
   Advances*, **79**, 145–162.
   `doi:10.1107/S2053273323000682
   <https://doi.org/10.1107/S2053273323000682>`_.

.. [Poulsen2017] Poulsen, H. F., Jakobsen, A. C., Simons, H., Ahl, S. R.,
   Cook, P. K. and Detlefs, C. (2017). X-ray diffraction microscopy based on
   refractive optics. *Journal of Applied Crystallography*, **50**, 1441–1456.
   `doi:10.1107/S1600576717011037
   <https://doi.org/10.1107/S1600576717011037>`_.

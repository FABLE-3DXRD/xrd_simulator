.. image:: https://github.com/FABLE-3DXRD/xrd_simulator/blob/main/docs/source/images/logo.png?raw=true

.. image:: https://img.shields.io/pypi/pyversions/xrd-simulator.svg?
	:target: https://pypi.org/project/xrd-simulator/

.. image:: https://github.com/FABLE-3DXRD/xrd_simulator/actions/workflows/python-package-run-tests-linux-py38.yml/badge.svg?
	:target: https://github.com/FABLE-3DXRD/xrd_simulator/actions/workflows/python-package-run-tests-linux-py38.yml

.. image:: https://github.com/FABLE-3DXRD/xrd_simulator/actions/workflows/pages/pages-build-deployment/badge.svg?
	:target: https://github.com/FABLE-3DXRD/xrd_simulator/actions/workflows/pages/pages-build-deployment/

.. image:: https://badge.fury.io/py/xrd-simulator.svg?
	:target: https://pypi.org/project/xrd-simulator/

.. image:: https://anaconda.org/conda-forge/vsc-install/badges/platforms.svg?
	:target: https://anaconda.org/conda-forge/xrd_simulator/

.. image:: https://anaconda.org/conda-forge/xrd_simulator/badges/latest_release_relative_date.svg?
	:target: https://anaconda.org/conda-forge/xrd_simulator/

===================================================================================================
Simulate X-ray Diffraction from Polycrystals in 3D.
===================================================================================================
.. image:: https://img.shields.io/badge/stability-alpha-f4d03f.svg?
	:target: https://github.com/FABLE-3DXRD/xrd_simulator/


The **X**-**R** ay **D** iffraction **SIMULATOR** package defines polycrystals as a mesh of tetrahedral single crystals
and simulates diffraction as collected by a 2D discretized detector array while the sample is rocked
around an arbitrary rotation axis. The full journal paper associated to the release of this code can be found here:

*xrd_simulator: 3D X-ray diffraction simulation software supporting 3D polycrystalline microstructure morphology descriptions
Henningsson, A. & Hall, S. A. (2023). J. Appl. Cryst. 56, 282-292.*
`https://doi.org/10.1107/S1600576722011001`_

``xrd_simulator`` was originally developed with the hope to answer questions about measurement optimization in
scanning x-ray diffraction experiments. However, ``xrd_simulator`` can simulate a wide range of experimental
diffraction setups. The essential idea is that the sample and beam topology can be arbitrarily specified,
and their interaction simulated as the sample is rocked. This means that standard "non-powder" experiments
such as `scanning-3dxrd`_ and full-field `3dxrd`_ (or HEDM if you like) can be simulated as well as more advanced
measurement sequences such as helical scans for instance. It is also possible to simulate `powder like`_
scenarios using orientation density functions as input.

===================================================================================================
Introduction
===================================================================================================
Before reading all the boring documentation (`which is hosted here`_) let's dive into some end to end
examples to get us started on a good flavour.

The ``xrd_simulator`` is built around four python objects which reflect a diffraction experiment:

   * A **beam** of x-rays (using the ``xrd_simulator.beam`` module)
   * A 2D area **detector** (using the ``xrd_simulator.detector`` module)
   * A 3D **polycrystal** sample (using the ``xrd_simulator.polycrystal`` module)
   * A rigid body sample **motion** (using the ``xrd_simulator.motion`` module)

Once these objects are defined it is possible to let the **detector** collect scattering of the **polycrystal**
as the sample undergoes the prescribed rigid body **motion** while being illuminated by the xray **beam**.

Let's go ahead and build ourselves some x-rays:

   .. code:: python

      import numpy as np
      from xrd_simulator.beam import Beam
      # The beam of xrays is represented as a convex polyhedron
      # We specify the vertices in a numpy array.
      beam_vertices = np.array([
          [-1e6, -500., -500.],
          [-1e6, 500., -500.],
          [-1e6, 500., 500.],
          [-1e6, -500., 500.],
          [1e6, -500., -500.],
          [1e6, 500., -500.],
          [1e6, 500., 500.],
          [1e6, -500., 500.]])

      beam = Beam(
          beam_vertices,
          xray_propagation_direction=np.array([1., 0., 0.]),
          wavelength=0.28523,
          polarization_vector=np.array([0., 1., 0.]))

We will also need to define a detector:

   .. code:: python

      from xrd_simulator.detector import Detector
      # The detector plane is defined by it's corner coordinates det_corner_0,det_corner_1,det_corner_2
      detector = Detector(pixel_size_z=75.0,
                          pixel_size_y=55.0,
                          det_corner_0=np.array([142938.3, -38400., -38400.]),
                          det_corner_1=np.array([142938.3, 38400., -38400.]),
                          det_corner_2=np.array([142938.3, -38400., 38400.]))

Next we go ahead and produce a sample, to do this we need to first define a mesh that
describes the topology of the sample, in this example we make the sample shaped as a ball:

   .. code:: python

      from xrd_simulator.mesh import TetraMesh
      # xrd_simulator supports several ways to generate a mesh, here we
      # generate meshed solid sphere using a level set.
      mesh = TetraMesh.generate_mesh_from_levelset(
          level_set=lambda x: np.linalg.norm(x) - 768.0,
          bounding_radius=769.0,
          max_cell_circumradius=450.)

Every element in the sample is composed of some material, or "phase", we define the present phases
in a list of ``xrd_simulator.phase.Phase`` objects, in this example only a single phase is present:

   .. code:: python

      from xrd_simulator.phase import Phase
      quartz = Phase(unit_cell=[4.926, 4.926, 5.4189, 90., 90., 120.],
                     sgname='P3221',  # (Quartz)
                     path_to_cif_file=None  # phases can be defined from crystalographic information files
                     )

The polycrystal sample can now be created. In this example the crystal elements have random orientations
and the strain is uniformly zero in the sample:

   .. code:: python

      from scipy.spatial.transform import Rotation as R
      from xrd_simulator.polycrystal import Polycrystal
      orientation = R.random(mesh.number_of_elements).as_matrix()
      polycrystal = Polycrystal(mesh,
                                orientation,
                                strain=np.zeros((3, 3)),
                                phases=quartz,
                                element_phase_map=None)

We may save the polycrystal to disc by using the builtin ``save()`` command as

   .. code:: python

      polycrystal.save('my_polycrystal', save_mesh_as_xdmf=True)

We can visualize the sample by loading the .xdmf file into your favorite 3D rendering program.
In `paraview`_ the sampled colored by one of its Euler angles looks like this:

.. image:: https://github.com/FABLE-3DXRD/xrd_simulator/blob/main/docs/source/images/example_polycrystal_readme.png?raw=true
   :align: center

We can now define some motion of the sample over which to integrate the diffraction signal:

   .. code:: python

      from xrd_simulator.motion import RigidBodyMotion
      motion = RigidBodyMotion(rotation_axis=np.array([0, 1/np.sqrt(2), -1/np.sqrt(2)]),
                               rotation_angle=np.radians(1.0),
                               translation=np.array([123, -153.3, 3.42]))

Now that we have an experimental setup we may collect diffraction by letting the beam and detector
interact with the sample:

   .. code:: python

      polycrystal.diffract(beam, detector, motion)
      diffraction_pattern = detector.render(frames_to_render=0,
                                              lorentz=False,
                                              polarization=False,
                                              structure_factor=False,
                                              method="project")

The resulting rendered detector frame will look something like the below. Note that the positions of the diffraction spots may vary as the crystal orientations were randomly generated!:

   .. code:: python

      import matplotlib.pyplot as plt
      fig,ax = plt.subplots(1,1)
      ax.imshow(diffraction_pattern, cmap='gray')
      plt.show()

.. image:: https://github.com/FABLE-3DXRD/xrd_simulator/blob/main/docs/source/images/diffraction_pattern.png?raw=true
   :align: center

To compute several frames simply change the motion and collect the diffraction again. The sample may be moved before
each computation using the same or another motion.

   .. code:: python

      polycrystal.transform(motion, time=1.0)
      polycrystal.diffract(beam, detector, motion)

Many more options for experimental setups and intensity rendering exist, have fun experimenting!
The above example code can be found as a `single .py file here.`_

======================================
Installation
======================================

Anaconda installation (Linux and Macos)
=============================================
``xrd_simulator`` is distributed on the `conda-forge channel`_ and the preferred way to install
the xrd_simulator package is via `Anaconda`_::

   conda create -n xrd_simulator
   conda activate xrd_simulator
   conda install -c conda-forge xrd_simulator

This is meant to work across OS-systems and requires an `Anaconda`_ installation.

(The conda-forge feedstock of ``xrd_simulator`` `can be found here.`_)

Anaconda installation (Windows)
======================================
To install with anaconda on windows you must make sure that external dependencies of `pygalmesh`_ are preinstalled
on your system. Documentation on installing these package `can be found elsewhere.`_

Pip Installation
======================================
Pip installation is possible, however, external dependencies of `pygalmesh`_ must the be preinstalled
on your system. Installation of these will be OS dependent and documentation
`can be found elsewhere.`_::

   pip install xrd-simulator

Source installation
===============================
Naturally one may also install from the sources::

   git clone https://github.com/FABLE-3DXRD/xrd_simulator.git
   cd xrd_simulator
   python setup.py install

This will then again require the `pygalmesh`_ dependencies to be resolved beforehand.

Credits
===============================
``xrd_simulator`` makes good use of xfab and pygalmesh. The source code of these repos can be found here:

* `https://github.com/FABLE-3DXRD/xfab`_
* `https://github.com/nschloe/pygalmesh`_

Citation
===============================
If you feel that ``xrd_simulator`` was helpful in your research we would love for you to cite us.

*xrd_simulator: 3D X-ray diffraction simulation software supporting 3D polycrystalline microstructure morphology descriptions
Henningsson, A. & Hall, S. A. (2023). J. Appl. Cryst. 56, 282-292.*
`https://doi.org/10.1107/S1600576722011001`_

.. _https://doi.org/10.1107/S1600576722011001: https://doi.org/10.1107/S1600576722011001

.. _https://github.com/FABLE-3DXRD/xfab: https://github.com/FABLE-3DXRD/xfab

.. _https://github.com/marmakoide/miniball: https://github.com/marmakoide/miniball

.. _Anaconda: https://www.anaconda.com/products/individual

.. _pygalmesh: https://github.com/nschloe/pygalmesh

.. _https://github.com/nschloe/pygalmesh: https://github.com/nschloe/pygalmesh

.. _can be found elsewhere.: https://github.com/nschloe/pygalmesh#installation

.. _scanning-3dxrd: https://doi.org/10.1107/S1600576720001016

.. _3dxrd: https://en.wikipedia.org/wiki/3DXRD

.. _powder like: https://en.wikipedia.org/wiki/Powder_diffraction

.. _which is hosted here: https://FABLE-3DXRD.github.io/xrd_simulator/

.. _which is hosted here: https://FABLE-3DXRD.github.io/xrd_simulator/

.. _single .py file here.: https://github.com/FABLE-3DXRD/xrd_simulator/blob/main/docs/source/examples/readme_tutorial.py

.. _paraview: https://www.paraview.org/

.. _can be found here.: https://github.com/conda-forge/xrd_simulator-feedstock

.. _conda-forge channel: https://anaconda.org/conda-forge/xrd_simulator

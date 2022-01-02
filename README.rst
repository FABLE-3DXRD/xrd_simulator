=====================================================================
Welcome to the xrd_simulator
=====================================================================
This is a package for simulating X-ray Difraction from polycrystals

* Defines polycrystals as a mesh of tetrahedral single crystals.

* Simulates diffraction based on structure factors and a discretized detector array.

======================================
Installation
======================================

Prerequisites
======================================
xrd_simulator depends on pygalmesh which in turn depends in the external CGAL and EIGEN packages.
It is therefore neccessary, prior to installing xrd_simulator, to either install these packages
manually, as documented here, or to install pygalmesh using conda as::

   conda install -c conda-forge pygalmesh

the later should work identically across OS systems.

Pip installation
======================================
Once pygalmesh is available xrd_simulator can be installed as::

   pip install xrd-simulator

Source installation
===============================
Naturally one can also install from the sources::

   git clone https://github.com/AxelHenningsson/xrd_simulator.git
   cd xrd_simulator
   python setup.py install

Anaconda installation
===============================
Unfortunately xrd_simulator is not yet available for pure conda installation. This is mainly due to
xfab and miniball not being available as conda packages.
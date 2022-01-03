=====================================================================
Welcome to the xrd_simulator
=====================================================================
This is a package for simulating X-ray Difraction from polycrystals

* Defines polycrystals as a mesh of tetrahedral single crystals.

* Simulates diffraction based on structure factors and a discretized detector array.

======================================
Installation
======================================

Anaconda installation
===============================
The preffered way to install the xrd_simulator package is via anaconda::

   conda install -c axiomel xrd_simulator

This is meant work across OS-systems and requires no prerequisites except, of course,
that of `Anaconda`_ itself.

Pip Installation
======================================
Pip installation is possible, however, external dependecies of `pygalmesh`_ must the be preinstalled
on your system. Installation of these will be OS dependent and documentaiton 
`can be found elsewhere.`_::

   pip install xrd-simulator

Source installation
===============================
Naturally one may also install from the sources::

   git clone https://github.com/AxelHenningsson/xrd_simulator.git
   cd xrd_simulator
   python setup.py install

This will then again require the `pygalmesh`_ dependecies to be resolved beforehand.

.. _Anaconda: https://www.anaconda.com/products/individual

.. _pygalmesh: https://github.com/nschloe/pygalmesh

.. _can be found elsewhere.: https://github.com/nschloe/pygalmesh

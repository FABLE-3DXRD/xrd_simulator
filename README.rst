=====================================================================
Welcome to the xrd_simulator
=====================================================================
A package for simulating X-ray Diffraction from polycrystals.

``xrd_simulator`` defines polycrystals as a mesh of tetrahedral single crystals and simulates
diffraction as collected by a 2D discretized detector array while the sample is rocked
around an arbitrary rotation axis.

``xrd_simulator`` was originally developed with the hope to answer questions about measurement optimization in
scanning x-ray diffraction experiments. However, ``xrd_simulator`` can simulate a wide range of experimental
diffraction setups. The essential idea is that the sample and beam topology can be arbitrarily specified,
and their interaction simulated as the sample is rocked. This means that standard "non-powder" experiments
such as `scanning-3dxrd`_ and full-field `3dxrd`_ (or HEDM if you like) can be simulated as well as more advanced
measurement sequences such as helical scans for instance. It is also possible to simulate `powder like`_
scenarios using orientation density functions as input.


======================================
Installation
======================================

Anaconda installation
===============================
The preferred way to install the xrd_simulator package is via anaconda::

   conda install -c conda-forge -c axiomel xrd_simulator

This is meant work across OS-systems and requires no prerequisites except, of course,
that of `Anaconda`_ itself.

.. note::
   Although xrd_simulator is distributed at the private ``axiomel`` conda channel some packages
   need be fetched from the more standard conda-forge channel. This is usually not default
   when installing anaconda, hence the need to add ``-c conda-forge`` in the above. If conda-forge
   is already on your channels it is possible to install as: ``conda install -c axiomel xrd_simulator``.


Pip Installation
======================================
Pip installation is possible, however, external dependencies of `pygalmesh`_ must the be preinstalled
on your system. Installation of these will be OS dependent and documentation
`can be found elsewhere.`_::

   pip install xrd-simulator

Source installation
===============================
Naturally one may also install from the sources::

   git clone https://github.com/AxelHenningsson/xrd_simulator.git
   cd xrd_simulator
   python setup.py install

This will then again require the `pygalmesh`_ dependencies to be resolved beforehand.

.. _Anaconda: https://www.anaconda.com/products/individual

.. _pygalmesh: https://github.com/nschloe/pygalmesh

.. _can be found elsewhere.: https://github.com/nschloe/pygalmesh#installation

.. _scanning-3dxrd: https://doi.org/10.1107/S1600576720001016

.. _3dxrd: https://en.wikipedia.org/wiki/3DXRD

.. _powder like: https://en.wikipedia.org/wiki/Powder_diffraction
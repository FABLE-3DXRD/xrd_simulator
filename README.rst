.. image:: docs/source/images/logo.png

.. image:: https://img.shields.io/pypi/pyversions/xrd-simulator.svg?
	:target: https://pypi.org/project/xrd-simulator/

.. image:: https://github.com/AxelHenningsson/xrd_simulator/actions/workflows/python-package-conda-linux-py38.yml/badge.svg?
	:target: https://github.com/AxelHenningsson/xrd_simulator/actions/workflows/

.. image:: https://github.com/AxelHenningsson/xrd_simulator/actions/workflows/python-package-conda-macos-py38.yml/badge.svg?
	:target: https://github.com/AxelHenningsson/xrd_simulator/actions/workflows/

.. image:: https://github.com/AxelHenningsson/xrd_simulator/actions/workflows/pages/pages-build-deployment/badge.svg?
	:target: https://github.com/AxelHenningsson/xrd_simulator/actions/workflows/pages/pages-build-deployment/

.. image:: https://badge.fury.io/py/xrd-simulator.svg?
	:target: https://pypi.org/project/xrd-simulator/

.. image:: https://anaconda.org/axiomel/xrd_simulator/badges/installer/conda.svg?
	:target: https://anaconda.org/axiomel/xrd_simulator/

.. image:: https://anaconda.org/axiomel/xrd_simulator/badges/platforms.svg?
	:target: https://anaconda.org/axiomel/xrd_simulator/

.. image:: https://anaconda.org/axiomel/xrd_simulator/badges/latest_release_relative_date.svg?
	:target: https://anaconda.org/axiomel/xrd_simulator/

===================================================================================================
Simulate X-ray Diffraction from Polycrystals.
===================================================================================================

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
   ``xrd_simulator`` works on python versions =>3.8<3.9. Make sure your conda environment has the right
   python version before installation. For instance, running ``conda install python=3.8`` before 
   installation should ensure correct behavior.

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
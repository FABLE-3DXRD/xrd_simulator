.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. include:: ../../README.rst

======================================
Documentation
======================================

beam
======================================

.. automodule:: xrd_simulator.beam
    :members:
    :show-inheritance:

detector
======================================

.. automodule:: xrd_simulator.detector
    :members:
    :show-inheritance:

motion
======================================

.. automodule:: xrd_simulator.motion
    :members:
    :show-inheritance:

mesh
======================================

.. automodule:: xrd_simulator.mesh
    :members:
    :show-inheritance:

phase
======================================

.. automodule:: xrd_simulator.phase
    :members:
    :show-inheritance:

polycrystal
======================================
The polycrystal is your sample. Once created it supports diffraction computations for a
provided diffraction geometry which is specified via a :ref:`beam`, sample :ref:`motion` and :ref:`detector`.

.. automodule:: xrd_simulator.polycrystal
    :members:
    :show-inheritance:

scatterer
======================================

.. automodule:: xrd_simulator.scatterer
    :members:
    :show-inheritance:

laue
======================================

.. automodule:: xrd_simulator.laue
    :members:
    :show-inheritance:

utils
======================================

.. automodule:: xrd_simulator.utils
    :members:
    :show-inheritance:

templates
======================================
The ``templates`` module allows for fast creation of a few select sample types and diffraction geometries without having to 
worry about any of the "under the hood" scripting.

.. automodule:: xrd_simulator.templates
    :members:
    :show-inheritance:

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
    :inherited-members:

detector
======================================

.. automodule:: xrd_simulator.detector
    :members:
    :inherited-members:

motion
======================================

.. automodule:: xrd_simulator.motion
    :members:
    :inherited-members:

mesh
======================================

.. automodule:: xrd_simulator.mesh
    :members:
    :inherited-members:

phase
======================================

.. automodule:: xrd_simulator.phase
    :members:
    :inherited-members:

polycrystal
======================================
The polycrystal is your sample. Once created it supports diffraction computations for a
provided diffraction geometry which is specified via a :ref:`beam`, sample :ref:`motion` and :ref:`detector`.

.. automodule:: xrd_simulator.polycrystal
    :members:
    :inherited-members:

scatterer
======================================

.. automodule:: xrd_simulator.scatterer
    :members:
    :inherited-members:

laue
======================================

.. automodule:: xrd_simulator.laue
    :members:
    :inherited-members:

utils
======================================

.. automodule:: xrd_simulator.utils
    :members:
    :inherited-members:

templates
======================================
The ``templates`` module allows for fast creation of a few select sample types and diffraction geometries without having to 
worry about any of the "under the hood" scripting.

.. automodule:: xrd_simulator.templates
    :members:
    :inherited-members:

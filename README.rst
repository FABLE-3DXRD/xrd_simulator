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
Since xrd_simulator depends on pygalmesh it is neccessary to first install
the CGAL and EIGEN packages. On Linux::

Linux
===============================
On Linux::

   sudo apt install libcgal-dev libeigen3-dev

MacOS
===============================
On MacOS with homebrew::

   brew install cgal eigen

Windows
===============================
On Windows...::

  ...

Pip installation
======================================
Once CGAL and EIGEN is installed xrd_simulator is available via the pypi channel::

   pip install xrd_simulator

Anaconda installation
===============================
xrd_simulator is not yet available as a conda package.

Source installation
===============================
you may also build form the source. In that case replace the pip instal step above by::

   git clone https://github.com/AxelHenningsson/xrd_simulator.git

next, go to the just downloaded directory::

   cd xrd_simulator

and install::

   python setup.py install


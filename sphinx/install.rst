.. $Id$
.. pyformex documentation --- install

.. include:: defines.inc
.. include:: ../website/src/links.inc

.. _cha:install:

*********************
Installing |pyformex|
*********************

.. warning:: This document is still under construction! 

.. topic:: Abstract

   This document explains the different ways for obtaining a running 
   |pyformex| installation. You will learn how to obtain |pyformex|, how
   to install it, and how to get it running.


.. _sec:introduction:

Introduction
============

There are several ways to get a running installation of |pyformex|, and you may
choose the most appropriate for you, depending on your needs, your current 
infrastructure and your computer knowledge. We will describe them in detail in
this document, and advice you on which method might be the best in your case. 

But first we give an overview of the most important pros and cons of the 
different install methods.

:ref:`sec:official-release`
---------------------------

+-------------------------------+---------------------------------------------+
| PROS                          |   CONS                                      |
+===============================+=============================================+
|  - Stable                     |  - Linux required                           |
|  - Well supported             |  - Root access required                     |
|  - Easy install procedure     |  - Installation of prerequisites required   |
|  - Site-wide install          |  - May be missing latest features           |
+-------------------------------+---------------------------------------------+


:ref:`sec:development-version`
------------------------------

+---------------------+--------------------------------------------+
| PROS                |   CONS                                     |
+=====================+============================================+
| - Latest features   | - Linux required                           |
| - No                | - No install procedure                     |
|   root              | - (Usually) single user install            |
|   access            | - Manual installation of prerequisites     |
|   required          |   (and root access) may be required        |
|                     | - Less stable                              |
+---------------------+--------------------------------------------+
 

:ref:`sec:bumpix-live-linux`
----------------------------

+--------------------------------------+------------------------------+
| PROS                                 |                          CONS|
+======================================+==============================+
| - No Linux required                  | - Missing latest features    |
| - No root access required            | - Difficult to upgrade       |
| - No installation required           | - Somewhat slower loading    |
| - Stable version                     |                              |
| - Easily portable                    |                              |
+--------------------------------------+------------------------------+


.. _sec:official-release:

Installing an official release
==============================

|pyformex| is software under development, and many users run it directly from
the latest development sources. This holds a certain risk however, because the development version may at times become unstable or incompatible with previous versions and thus break your applications.
At regular times we therefore create official releases, which provide a more stable and better documented and supported version, together with an esy install procedure. 

If you can meet the requirements for using an officially packed release, this is the recommended way to install |pyformex|. All the software packages needed to run |pyformex| can be obtained for free.

To install an official |pyformex| release, you need a working Linux system, root (administrator) privileges to the system, and you need to make sure that the prerequisite packages are installed on the system. 

If you need to install a new Linux system from scratch, and have the choice to
pick any distribution, we highly recommend `Debian GNU/Linux`_ or derivatives.
This is because most of the |pyformex| development is done on Debian systems, 
and we will give you `precise install instructions`_ for this system.
Also, the Debian software repositories are amongst the most comprehensive to be found on the Internet. 

Most popular Linux distributions provide appropriately packed recent versions
of these prerequisites, so that you can install them easily from the pacakge manager of your system. In case a package or version is not available for your system, you can always install it from source. We provide the websites where
you can find the source packages.

 
.. _sec:prerequisites:

Prerequisites
-------------

In order to install an official release package of |pyformex|, you need to have the following installed (and working) on your computer:

**Python** (http://www.python.org)
   Version 2.4 or higher (2.5 is recommended). Nearly all Linux distributions
   come with Python installed, so this should not be no major obstacle. 

**NumPy** (http://www.numpy.org)
   Version 1.0-rc1 or higher. NumPy is the package used for efficient
   numerical array operations in Python and is essential for |pyformex|.

**Qt4** (http://www.trolltech.com/products/qt)
   The widget toolkit on which the |pyformex| Graphical User Interface (GUI) was built.

**PyQt4** (http://www.riverbankcomputing.co.uk/pyqt/index.php) 
   The Python bindings for Qt4. 

**PyOpenGL** (http://pyopengl.sourceforge.net/)
   Python bindings for OpenGL, used for drawing and manipulating the
   3D-structures.

If you only want to use the Formex data model and transformation methods and
do not need the GUI, then NumPy is all you need. This could e.g. suffice for a
non-interactive machine that only does the numerical processing. The install
procedure however does not provide this option yet, so you will have to do the
install by hand.
Currently we recommend to install the whole package including the GUI.
Most probably you will want to visualize your structures and for that you need the GUI anyway.

Additionally, we recommend you to also install **the GNU C-compiler and the Python and OpenGL header files**. The install procedure needs these to compile the |pyformex| acceleration library. While |pyformex| can run without the library (Python versions will be substituted
for all functions in the library), using the library will dramatically speed up some low level operations such as drawing, especially when working with large structures . 


.. _`precise install instructions`:

Installing prerequisites on `Debian GNU/Linux`_
-----------------------------------------------
Debian users can install the package python-numpy.The extra packages 
python-numpy-ext and python-scipy give saom added functionality, 
but are not required for the basic operation of |pyformex|. 
For Debian users this  comes in the packages python-qt4.
Debian users should install the packages python-qt4 and python-qt4-gl.
For debian users this is in the package python-opengl.

On a Debian based system, you should install tha packages python-dev, python-qt4-dev and
libgl1-mesa-dev.

.. _sec:downloading:

Downloading
-----------

The official releases of can be downloaded from the  website. As of the writing
of this manual, the latest release is .  is currently distributed in the form of
a .tar.gz (tarball) archive. See :ref:`sec:installation-linux` for how to
proceed further.

Alternatively you can download the tarball releases from our local FTP server.
The server may be slower, but occasionally you can find there an interim release
or release candidate not (yet) available on the official server.

Finally, you can also get the latest development code from the SVN repository on
the website. If you have Subversionhttp://subversion.tigris.org/ installed on
your system, you can just dosvn checkout svn://svn.berlios.de/pyformex/trunk
pyformex and the whole current tree will be copied to a subdirectory
``pyformex`` on your current path.

*Unless you want to help with the development or you absolutely need some of the
latest features or bugfixes, the tarball releases are what you want to go for.*


.. _sec:installation-linux:

Installation on Linux platforms
-------------------------------

Once you have downloaded the tarball, unpack it with tar xvzf pyformex-
version.tar.gz Then go to the created pyformex directory: cd pyformex-version
and do (with root privileges)  python setup.py install --prefix=/usr/local This
will install under /usr/local/. You can change the prefix to install in some
other place.

The installation procedure installs everything into a single directory, and
creates a symlink to the executable in /usr/local/bin. You can use the command
pyformex --whereami to find out where is installed.

Finally, a installation can usually be removed by giving the command pyformex
--remove and answering 'yes' to the question. You may want to do this before
installing a new version, especially if you install a new release of an already
existing version.


.. _sec:development-version:

Installing a development version
================================



.. _sec:bumpix-live-linux:

Using a BuMPix Live Linux system
================================

If you do not have access to a running Linux system, or if the above 
installation methods fail for some unknown reason (you can ask for help on the 
|pyformex| `Forums`_), you can still run |pyformex| by using a 
`Bumpix Live Linux` system. `Bumpix Live` is a full featured Linux system 
including |pyformex| that can be run from a single removable medium 
such as a CD or a USB key.

All you need in this case is some proper hardware: the system boots and runs 
from the removable medium and leaves any software installed on the hard disk
of the computer untouched.

You can also easily take this media with you wherever you go, plug it in an
available computer, and start or continue your |pyformex| work. Some users
even prefer this way to run |pyformex| for that single reason.

The Live system is also an excellent way to test and see what |pyformex| can
do for you, without having to install it. Or to demonstrate |pyformex| to
your friends or colleagues. 




  

.. _sec:running-on-windows:

Running |pyformex| on Windows
-----------------------------

There is no installation procedure yet. All the pre-requisite packages are
available for Windows, so in theory it is possible to run on Windows. We know of
some users who are running succesfully using the --nogui option, i.e. without
the Graphical User Interface (GUI).   A few things may need to be changed for
running the GUI on Windows. We might eventually have a look at this in the
future, but it certainly is not our primary concern. Still, any feedback on
(successful or not successful) installation attempts on Windows is welcome.

   For Windows, binary packages are available on the Sourceforge
   download page http://www.numpy.org/.

.. End

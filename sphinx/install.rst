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

.. note:: **pyFormex on non-Linux systems**

  |pyformex| is being developed on Linux systems, and most users run it on a
  Linux platform. The :ref:`sec:pyformex-nonlinux` section holds information
  on how pyFormex can be run on other platforms.

Let's first give you an overview of the most important pros and cons of the 
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


:ref:`sec:alpha-release`
------------------------

+-------------------------------+---------------------------------------------+
| PROS                          |   CONS                                      |
+===============================+=============================================+
|  - Easy install               |  - Linux required                           |
|    procedure                  |  - Root access required                     |
|  - Site-wide                  |  - Installation of prerequisites required   |
|    install                    |  - Latests features                         |
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

+--------------------------------------+-----------------------------------+
| PROS                                 |                            CONS   |
+======================================+===================================+
| - No Linux required                  | - Missing latest features         |
| - No root access required            | - Difficult to upgrade            |
| - No installation required           | - Somewhat slower loading         |
| - Stable version                     |                                   |
| - Easily portable                    |                                   |
+--------------------------------------+-----------------------------------+


To sum it up:

- Unless you want to help with the development, or you absolutely
  need some of the latest features or bugfixes, or you just can not
  meet the requirements, the latest :ref:`sec:official-release` is
  what you want to go for. It gives you the highest degree of
  stability and support and comes packed in an archive, with an install
  procedure included.

- If you need some recent feature of |pyformex| that is not yet in an official
  release, you may be lucky to find it in the latest :ref:`sec:alpha-release`.

- If the install procedures do not work for you, or you need the
  absolutely latest development code, you can run |pyformex| directly
  from the anonymously checked out :ref:`sec:development-version`.

- Finally, if you do not have enough permissions to install the prerequisites,
  or if you do not have a Linux system in the first place, or if you just want
  to try out |pyformex|, or if you want a portable system that can you take
  with you and run anywhere, choose for the :ref:`sec:bumpix-live-linux` on
  USB stick.

.. _sec:official-release:

Official release
================

|pyformex| is software under development, and many users run it directly from
the latest development sources. This holds a certain risk however, because the development version may at times become unstable or incompatible with previous versions and thus break your applications.
At regular times we therefore create official releases, which provide a more stable and better documented and supported version, together with an easy install procedure. 

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
...............................................
Debian users should just have to install the packages ``python-numpy`` and  ``python-qt4-gl``. The latter will install ``python-qt4`` and ``python-qt4-gl`` as dependencies. Also, for compiling the accelearation library, you should install 
``python-dev``, ``python-qt4-dev`` and ``libgl1-mesa-dev``.

Other optional packages that might be useful are ``admesh``, ``python-scipy``,
``python-numpy-ext``, ``units``.


.. _sec:downloading:

Download pyFormex
-----------------

Official |pyformex| releases can be downloaded from this website:
`Official releases`_. As of the writing of this manual, the latest
release is 0.8. 

|pyformex| is currently distributed in the form of a .tar.gz (tarball) archive. See :ref:`sec:installation-linux` for how to proceed further with the downloaded file.

.. _sec:installation-linux:

Install pyFormex
----------------

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


Install additional software
---------------------------
pyFormex uses a large number of external software packages to enhance its
functionality. Some of these packages are so essential, that they were listed as requirements. Others however are merely optional packages: interesting for those users who need them, but not essential for everybody.
The user has the choice to install these extras or not.

Some external packages however do not come in an easy to install package, ot
the available packaged formats do not collaborate well with |pyformex|.
Therefore, we have created a set of dedicated install script to ease the
installation of these external packages.


.. _sec:alpha-release:

Alpha release
=============

Official release are only created a couple of times per year, because
they require a lot of testing.  |pyformex| is however developing fast,
and as soon as an official release has been made, new features are
already being included in the source repository. Sometimes, you may be
in need of such a feature, that will only become officially available
within several months.  Between successive official releases, we
create interim alpha releases. They are less well tested (meaning they
may contain more bugs) and supported than the official ones, but they
may just work well for you.

These alpha releases can be downloaded from the developer `FTP site`_
or from our `local FTP server`_. The latter may be slower, but
occasionally you may find there old releases or release candidates not
available on the official server.

They install just like the :ref:`sec:official-release`s. 


.. _sec:development-version:

Development version
===================

Finally, you can also get the latest development code from the SVN repository on
the website. If you have Subversionhttp://subversion.tigris.org/ installed on
your system, you can just dosvn checkout svn://svn.berlios.de/pyformex/trunk
pyformex and the whole current tree will be copied to a subdirectory
``pyformex`` on your current path.



.. _sec:bumpix-live-linux:

BuMPix Live Linux system
========================

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




  

.. _sec:pyformex-nonlinux:

Running |pyformex| on non-Linux systems
=======================================

pyFormex is being developed on Linux platforms, and most of its users run
pyFormex on a Linux system. Because of that, there is no installation
procedure to run pyFormex natively on other systems.

Currently, the easiest way to run pyFormex on a non-Linux system is by
using the :ref:`sec:bumpix-live-linux`. We use this frequently with
large groups of students in classes having only Windows PCs. We also
have some professional users who could no install Linux due to
corporate regulations, that are working this way.

Another possibility is to run a virtual Linux instance on the
platform. There is currently quite good virtualization software
available for nearly any platform.

However, as all the essential underlying packages on which pyFormex is
built are available for many other platforms (including Windows, Mac),
it is (in theory) possible to get pyFormex to run natively on
non-Linux platforms.  There has already been a successful attempt with
a rather old version, but with recent versions nobody has (yet) taken
the bother to try it.

There may be a few things that have to be changed to successfully run
pyFormex on other platforms (especially on Windows), but it should all
be rather obvious.  If you have some programming experience on your
platform, and you want to give it a try, please do so. We certainly
are willing to help where we can.  for running the GUI on Windows. And
we are certainly interested in feedback about your attempt, whether
successful or not.




.. End

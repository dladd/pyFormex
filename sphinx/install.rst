.. $Id$

..
  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
  pyFormex is a tool for generating, manipulating and transforming 3D
  geometrical models by sequences of mathematical operations.
  Home page: http://pyformex.org
  Project page:  http://savannah.nongnu.org/projects/pyformex/
  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
  Distributed under the GNU General Public License version 3 or later.


  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see http://www.gnu.org/licenses/.



.. include:: defines.inc
.. include:: links.inc

.. _cha:install:

*******************
Installing pyFormex
*******************

.. topic:: Abstract

   This document explains the different ways for obtaining a running
   pyFormex installation. You will learn how to obtain pyFormex, how
   to install it, and how to get it running.


.. _sec:choose_installation:

Choose installation type
========================

There are several ways to get a running installation of pyFormex, and you may
choose the most appropriate method for you, depending on your needs, your current
infrastructure and your computer knowledge. We will describe them in detail in
this document, and advice you on which method might be the best in your case.

.. note:: **pyFormex on non-Linux systems**

  pyFormex is being developed on GNU/Linux systems, and most users run it on a
  GNU/Linux platform. The :ref:`sec:pyformex-nonlinux` section holds information
  on how pyFormex can be run on other platforms.

Let's first give you an overview of the most important pros and cons of the
different install methods.

:ref:`sec:debian-packages`:

.. The following fixes problems with generating the PDF docs
.. tabularcolumns:: |p{5cm}|p{5cm}|

+-------------------------------+-------------------------------------------+
| PROS                          |   CONS                                    |
+===============================+===========================================+
|  - Stable                     |  - Debian GNU/Linux required [#debian]_   |
|  - Well supported             |  - Root access required                   |
|  - Easy install procedure     |  - May be missing latest features         |
|  - Automatic installation of  |                                           |
|    all dependencies           |                                           |
|  - Easy update procecure      |                                           |
|  - Easy removal procecure     |                                           |
|  - Site-wide install          |                                           |
+-------------------------------+-------------------------------------------+

.. [#debian] Installing the Debian packages may also work on Debian derivatives
   like Ubuntu and Mint.

:ref:`sec:official-release`:

.. tabularcolumns:: |p{5cm}|p{5cm}|

+-------------------------------+-------------------------------------------+
| PROS                          |   CONS                                    |
+===============================+===========================================+
|  - Stable                     |  - GNU/Linux required                     |
|  - Well supported             |  - Root access required                   |
|  - Easy install procedure     |  - Installation of dependencies required  |
|  - Site-wide install          |  - May be missing latest features         |
+-------------------------------+-------------------------------------------+

:ref:`sec:alpha-release`:

.. tabularcolumns:: |p{5cm}|p{5cm}|

+------------------------------+--------------------------------------------+
| PROS                         |   CONS                                     |
+==============================+============================================+
|  - Easy install              |  - GNU/Linux required                      |
|    procedure                 |  - Root access required                    |
|  - Site-wide                 |  - Installation of dependencies required   |
|    install                   |  - Latests features                        |
+------------------------------+--------------------------------------------+


:ref:`sec:development-version`:

.. tabularcolumns:: |p{5cm}|p{5cm}|

+---------------------+--------------------------------------------+
| PROS                |   CONS                                     |
+=====================+============================================+
| - Latest features   | - GNU/Linux required                       |
| - No root           | - Requires development tools               |
|   access required   | - (Usually) single user install            |
| - No installation   | - Manual installation of dependencies      |
|   required          |   (and root access) may be required        |
|                     | - Less stable                              |
+---------------------+--------------------------------------------+


:ref:`sec:bumpix-live-linux`:

.. tabularcolumns:: |p{5cm}|p{5cm}|

+-------------------------------------+----------------------------------+
| PROS                                |                            CONS  |
+=====================================+==================================+
| - No GNU/Linux required             | - Missing latest features        |
| - No root access required           | - Somewhat slower loading        |
| - No installation required          |                                  |
| - Stable version                    |                                  |
| - Easily portable                   |                                  |
| - Upgradeable by installing         |                                  |
|   development version               |                                  |
+-------------------------------------+----------------------------------+


To sum it up:

- Unless you want to help with the development, or you absolutely
  need some of the latest features or bugfixes, or you just can not
  meet the requirements, the latest :ref:`sec:debian-packages` or
  :ref:`sec:official-release` source tarballs are
  what you want to go for. They give you the highest degree of
  stability and support and come packed in an archive, with an easy install
  procedure provided by your distributions package manager or included in
  the source tarball.

- If you need some recent feature of pyFormex that is not yet in an official
  release, you may be lucky to find it in some :ref:`sec:alpha-release`.

- If the install procedures do not work for you, or you need the
  absolutely latest development code, you can run pyFormex directly
  from the anonymously checked out :ref:`sec:development-version`.

- Finally, if you do not have enough permissions to install the dependencies,
  or if you do not have a GNU/Linux system in the first place, or if you just
  want  to try out pyFormex without having to install anything,
  or if you want a portable system that can you take
  with you and run anywhere, choose for the :ref:`sec:bumpix-live-linux` on
  USB stick.

.. _sec:debian-packages:

Debian packages
===============

If you are running Debian GNU/Linux, or have the opportunity to install
it, then (by far) the most easy install method is to use the packages
in the official Debian repositories. Currently pyFormex packages are available for Debian sid and wheezy releases. Be sure to also install the precompiled acceleration libraries::

  apt-get install pyformex pyformex-lib

This single command will install pyFormex and all its dependencies. Some extra
functionalities may be installable from a separate package::

  apt-get install pyformex-extra

If you need a more recent version of pyFormex then the one available in the official repositories, you may try your luck with our `local package repository`_.
It contains debian format packages of intermediate releases and test packages for the official releases.
To access our package repository from your normal package manager, add the following lines to your `/etc/apt/sources.list`::

  deb http://bumps.ugent.be/repos/debian/ sid main
  deb-src http://bumps.ugent.be/repos/debian/ sid main

You can add the key used to sign the packages to the apt keyring with the following command::

  wget -O - http://bumps.ugent.be/repos/pyformex-pubkey.gpg | apt-key add -

Then do ``apt-get update``. Now you can use the above commands to install
the latest alpha release.



.. _sec:official-release:

Official release
================

pyFormex is software under development, and many users run it directly from
the latest development sources. This holds a certain risk however, because the development version may at times become unstable or incompatible with previous versions and thus break your applications.
At regular times we therefore create official releases, which provide a more stable and better documented and supported version, together with an easy install procedure.

If you can meet the requirements for using an officially packed release, and you can not use the :ref:`sec:debian-packages`, this is the recommended way to install pyFormex. All the software packages needed to compile and run pyFormex can be obtained for free.

To install an official pyFormex release, you need a working GNU/Linux system, root (administrator) privileges to the system, and you need to make sure that the dependencies listed below are installed first on the system. Furthermore, you need the usual GNU development tools (gcc, make).

If you need to install a new GNU/Linux system from scratch, and have the choice to pick any distribution, we highly recommend `Debian GNU/Linux`_ or derivatives.
This is because most of the pyFormex development is done on Debian systems,
and we will give you `precise install instructions`_ for this system.
Also, the Debian software repositories are amongst the most comprehensive to be found on the Internet. Furthermore, as of pyFormex version 0.8.6, we provide official :ref:`sec:debian-packages`, making installation really a no-brainer.

Most popular GNU/Linux distributions provide appropriately packed recent versions
of the dependencies, so that you can install them easily from the pacakge manager of your system. In case a package or version is not available for your system, you can always install it from source. We provide the websites where
you can find the source packages.


.. _sec:dependencies:

Dependencies
-------------

In order to install an official release package of pyFormex, you need to have the following installed (and working) on your computer:

**Python** (http://www.python.org)
   Version 2.5 or higher (2.6 or 2.7 is recommended). Nearly all GNU/Linux distributions come with Python installed, so this should be no major obstacle.

**NumPy** (http://www.numpy.org)
   Version 1.0 or higher. NumPy is the package used for efficient
   numerical array operations in Python and is essential for pyFormex.

**Qt4** (http://www.trolltech.com/products/qt)
   The widget toolkit on which the pyFormex Graphical User Interface (GUI) was built.

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

Additionally, we recommend you to also install the Python and OpenGL header files. The install procedure needs these to compile the pyFormex acceleration library. While pyFormex can run without the library (Python versions will be substituted for all functions in the library), using the library will dramatically speed up some low level operations such as drawing, especially when working with large structures .


.. _`precise install instructions`:

Installing dependencies on `Debian GNU/Linux`
..............................................
Debian users should just have to install the packages ``python-numpy``, ``python-opengl`` and  ``python-qt4-gl``. The latter will install ``python-qt4`` as dependency. Also, for compiling the acceleration library, you should install
``python-dev`` and ``libglu1-mesa-dev``. This command will do it all::

  apt-get install python-numpy python-opengl python-qt4-gl python-dev libglu1-mesa-dev

Other optional packages that might be useful are ``admesh``, ``python-scipy``,
``python-numpy-ext``, ``units``.


.. _sec:downloading:

Download pyFormex
-----------------

Official pyFormex releases can be downloaded from this website:
`Releases`_. As of the writing of this manual, the latest
release is |latest|.

pyFormex is currently distributed in the form of a .tar.gz (tarball) archive. See :ref:`sec:installation-linux` for how to proceed further with the downloaded file.

.. _sec:installation-linux:

Install pyFormex
----------------

Once you have downloaded the tarball, unpack it with the command ::

   tar xvzf pyformex-VERSION.tar.gz

where you replace ``VERSION`` with the correct version from the downloaded file.
Then go to the created pyformex directory ::

   cd pyformex-VERSION

and execute the following command with root privileges::

   python setup.py install --prefix=/usr/local

This will install pyFormex under ``/usr/local/``.
You can change the prefix to install pyFormex in some other place.

The installation procedure installs everything into a single
directory, and creates a symlink to the executable in
``/usr/local/bin``. You can use the command ::

   pyformex --whereami

to find out where pyFormex is installed.

Finally, a pyFormex tarball installation can usually be removed by giving the command ::

   pyformex --remove

and answering 'yes' to the question.  You may
want to do this before installing a new version, especially if you
install a new release of an already existing version.


.. _subsec:development-version:

Install additional software
---------------------------

pyFormex uses a large number of external software packages to enhance
its functionality. Some of these packages are so essential, that they
were listed as requirements. Others however are merely optional
packages: interesting for those users who need them, but not essential
for everybody.  The user has the choice to install these extras or
not.

Some external packages however do not come in an easy to install package, or
the available packaged formats do not collaborate well with pyFormex.
Therefore, we have created a set of dedicated install script to ease the
installation of these external packages. Currently, there is an
install procedure for the following packages:

.. warning:: We provide these installation procedures for your
   convenience, but take no responsibility for them working correctly.

**gl2ps**
   This package allows to save the OpenGL rendering to a file in
   vector format. Currently supported are ``eps``, ``pdf`` and
   ``svg``. Our install procedure provides the necessary Python
   interface and installs the gl2ps library at the same time.

**gts**
   This package (Gnu Triangluted Surfaces) implements a library of
   powerful functions for operating on triangulated surface models.
   It also delivers some example programs built with the library.
   The pyFormex ``surface`` plugin uses these for many of its
   functions. Debian users should install the packages ``libgts-0.7.5``,
   ``libgts-bin`` and ``libgts-dev`` as dependencies.


**tetgen**
   This package provides a high quality tetrahedral mesher. pyFormex
   has some import and export functions for the specific ``tetgen`` file formats.
   Since ``tetgen`` is only distributed in source form, we provide this
   install procedure to help with the compile/install.

**calpy**
   Calpy is an experimental package that provides efficient Finite Element
   simulations through a Python interface. It does this by calling
   into a library of compiled Fortran routines. There is currently no
   distribution to the general public yet, but this install procedure
   grabs the source from our local FTP server, compiles the library
   and creates the Python interface. pyFormex comes with some examples
   that use Calpy as a simulatiopn tool.

To install any of these packages, proceed as follows. Go to the
directory where you unpacked the pyFormex distribution:
``cd pyformex-version``. Then go to the ``pyformex/external``
subdirectory, where you will find a subdirectory for each of the
above packages. Go into the directory of the package you wish to
install and execute the following commands (install may require
root privileges)::

  make
  make install

In some case there is no ``Makefile`` provided but an install script instead.
Then you can just do::

  ./install.sh all

All these procedures will install under ``/usr/local``. If you wish to
change this, you will have to change the ``Makefile`` or install procedure.
The install procedures can also be used to perform only part of the
installation process. Thus, ``./install.sh get unpack`` will
only download and unpack that package. See the README files and the
install procedures themselves for more info.



.. _sec:alpha-release:

Alpha release
=============

Official releases are only created a couple of times per year, because
they require a lot of testing.  pyFormex is however developing fast,
and as soon as an official release has been made, new features are
already being included in the source repository. Sometimes, you may be
in need of such a feature, that will only become officially available
within several months.  Between successive official releases, we
create interim alpha releases. They are less well tested (meaning they
may contain more bugs) and supported than the official ones, but they
may just work well for you.

These alpha releases can be downloaded from the developer `FTP site`_
or from our `local FTP server`_. The latter may be slower, but
you may find there some old releases or release candidates that are not
available on the official server.
They install just like the :ref:`sec:official-release`.

Again, as a Debian user, you may be extra lucky: we usually create Debian
:ref:`sec:debian-packages` from these alpha releases and make them available on our `local package repository`_.


.. _sec:development-version:

Development version
===================

If the install procedures for the packaged releases do not work for
you, or if you want to have the absolutely latest features and bug fixes,
then you can run pyFormex directly from the development sources.
Obviously, the pyFormex developers use this method, but there are also
several normal users who prefer this, because it allows for easy
updating to the latest version.

.. note: The pyFormex project has recently migrated from Subversion to git as
   its version control system. The Subversion repository is still accessible,
   but does not contain the latest revisions. See below for checking out the
   source from the Subversion repository


To run pyFormex from the development sources you need to have the same
dependencies installed as for the :ref:`sec:official-release`.
Furthermore, you need the `git`_ revision control system.
You can check whether you have it by trying the command ``git``.
If you do not have the command, you should first install it.
Debian and Ubuntu users can just do ``apt-get install git``.

Now you can anonymously check out the latest pyFormex version
from the `Source code`_ repository at the `Project page`_, using the
command::

  git clone git://git.savannah.nongnu.org/pyformex.git

This will create a directory ``pyformex`` with the full source.

.. note: If you already have a directory or file named ``pyformex`` in your
   current path, the git command will not overwrite it.

Now you can directly run pyFormex from the created ``pyformex`` directory::

  cd pyformex
  pyformex/pyformex

The first time you run the command, it will start with compiling the pyFormex
acceleration libraries. When that has finished, the pyFormex GUI will start,
running directly from your checked out source.
The next time you run the command, the library will not be
recompiled, unless some updates have been made to the files, making the
already compiled versions out of date.

You can make the ``pyformex/pyformex`` command
executable from anywhere by creating a symlink under one of the
directories in your ``PATH`` environment variable.
Many GNU/Linux distributions add ``/home/USER/bin`` to the user's path.
Thus the following command is suitable in most cases::

  ln -sfn BASEDIR/pyformex/pyformex /home/USER/bin

where ``BASEDIR`` is the full path to the directory where you checked out the
source.

The pyFormex repository contains a lot of files that are only needed and
interesting for the pyFormex developers. As a normal user you may want to
remove this extra overhead in your copy. To do so, run the `sparse_checkout`
script from the checkout directory::

  sh sparse_checkout

You can update your pyFormex installation at any time to the latest
version by issuing the command ::

  git pull

in your ``BASEDIR`` directory. You can even roll back to any older revision of
pyFormex. Just remember that after updating your sources, the compiled
libraries could be out of sync with your new sources. Normally pyFormex
will rebuild the libraries the next time you start it.
If you ever want to rebuild the libraries without starting the ``pyformex``
program, you can use the command ``make lib`` from inside ``BASEDIR``.


Using the older Subversion repository
-------------------------------------

To run pyFormex from the development sources you need to have the same
dependencies installed as for the
:ref:`sec:official-release`. Furthermore, you need the
`Subversion`_ revision control system. You can check whether you have it
by trying the command ``svn help``. If you do not have the command,
you should install Subversion first. Debian and Ubuntu users can just
do ``apt-get install subversion``.

Now you can anonymously check out the latest pyFormex version
from the `SVN Source code`_ repository at the `Project page`_.
If you are not a pyFormex developer, the suggested commands for this checkout
are::

  svn co svn://svn.savannah.nongnu.org/pyformex/trunk --depth files MYDIR
  svn update --depth infinity MYDIR/pyformex

In these commands you should replace ``MYDIR`` with the path name of a
directory where you have write permission. Many users choose
``~/pyformex`` as the checkout directory, but this is not
required. You can even check out different versions under
different path names. If you leave out the ``MYDIR`` from the above command,
a new directory ``trunk`` will be created in the current path.

Instead of the above two commands, you could also use the following single
command to check out the whole trunk, but that would download a lot of extra
files which are only useful for pyFormex developers, not for normal users ::

  svn co svn://svn.savannah.nongnu.org/pyformex/trunk MYDIR

Now change into the created ``MYDIR`` directory, where you can execute
the command ``pyformex/pyformex`` and proceed as explained above for a
checkout of the git repository.


.. _sec:bumpix-live-linux:

BuMPix Live GNU/Linux system
============================

If you do not have access to a running GNU/Linux system, or if the above
installation methods fail for some unknown reason (remember, you can
ask for help on the pyFormex `Support tracker`_), you can still run pyFormex by
using a `Bumpix Live GNU/Linux`_ system. `Bumpix Live` is a full featured
Debian GNU/Linux system including pyFormex that can be run from a single
removable medium such as a CD or a USB key.
Installation instructions can be found in :doc:`bumpix`.

Alternatively,

* if you do not succeed in properly writing the image to a USB key, or
* if you just want an easy solution without any install troubles, or
* if you want to financially support the further development of pyFormex, or
* if you need a large number of pyFormex USB installations,

you may be happy to know that we can provide ready-made BuMPix USB
sticks with the ``pyformex.org`` logo at a cost hardly exceeding that
of production and distribution.
If you think this is the right choice for you, just `email us`_ for a quotation.

Further guidelines for using the BuMPix system can be found in :doc:`bumpix`.


.. _sec:pyformex-nonlinux:

Running pyFormex on non-Linux systems
=====================================

pyFormex is being developed on GNU/Linux platforms, and most of its users run
pyFormex on a GNU/Linux system. Because of that, there is no installation
procedure to run pyFormex natively on other systems.

Currently, the easiest way to run pyFormex on a non-Linux system is by
using the :ref:`sec:bumpix-live-linux`. We use this frequently with
large groups of students in classes having only Windows PCs. We also
have some professional users who could no install GNU/Linux due to
corporate regulations, that are working this way.

Another possibility is to run a virtual GNU/Linux instance on the
platform. There is currently quite good virtualization software
available for nearly any platform.

However, as all the essential underlying packages on which pyFormex is
built are available for many other platforms (including Windows, Mac),
it is (in theory) possible to get pyFormex to run natively on
non-Linux platforms.  There has already been a successful attempt with
a rather old version, but with recent versions nobody has (yet) taken
the bother to try it.

.. note:: **pyFormex on Windows**
   Lately there have been some successful attempts to get the basic
   functionality of pyFormex running on Windows. Thomas Praet has
   compiled `this document
   <ftp://bumps.ugent.be/pub/pyformex/Install_pyFormex_on_Windows.html>`_
   on how to proceed.
   Submit a request on the `Support tracker`_ if you need any help.


There may be a few things that have to be changed to successfully run
pyFormex on other platforms (especially on Windows), but it should all
be rather obvious.  If you have some programming experience on your
platform, and you want to give it a try, please do so. We certainly
are willing to help where we can. And
we are certainly interested in feedback about your attempt, whether
successful or not.




.. End

.. $Id$
.. pyformex documentation --- install

.. include:: defines.inc
.. include:: ../website/src/links.inc

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

  pyFormex is being developed on Linux systems, and most users run it on a
  Linux platform. The :ref:`sec:pyformex-nonlinux` section holds information
  on how pyFormex can be run on other platforms.

Let's first give you an overview of the most important pros and cons of the 
different install methods. 

:ref:`sec:official-release`:

.. The following fixes problems with generating the PDF docs
.. tabularcolumns:: |p{5cm}|p{5cm}|

+-------------------------------+-------------------------------------------+
| PROS                          |   CONS                                    |
+===============================+===========================================+
|  - Stable                     |  - Linux required                         |
|  - Well supported             |  - Root access required                   |
|  - Easy install procedure     |  - Installation of prerequisites required |
|  - Site-wide install          |  - May be missing latest features         |
+-------------------------------+-------------------------------------------+

:ref:`sec:alpha-release`:

.. tabularcolumns:: |p{5cm}|p{5cm}|

+------------------------------+--------------------------------------------+
| PROS                         |   CONS                                     |
+==============================+============================================+
|  - Easy install              |  - Linux required                          |
|    procedure                 |  - Root access required                    |
|  - Site-wide                 |  - Installation of prerequisites required  |
|    install                   |  - Latests features                        |
+------------------------------+--------------------------------------------+


:ref:`sec:development-version`:

.. tabularcolumns:: |p{5cm}|p{5cm}|

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
 

:ref:`sec:bumpix-live-linux`:

.. tabularcolumns:: |p{5cm}|p{5cm}|

+-------------------------------------+----------------------------------+
| PROS                                |                            CONS  |
+=====================================+==================================+
| - No Linux required                 | - Missing latest features        |
| - No root access required           | - Difficult to upgrade           |
| - No installation required          | - Somewhat slower loading        |
| - Stable version                    |                                  |
| - Easily portable                   |                                  |
+-------------------------------------+----------------------------------+


To sum it up:

- Unless you want to help with the development, or you absolutely
  need some of the latest features or bugfixes, or you just can not
  meet the requirements, the latest :ref:`sec:official-release` is
  what you want to go for. It gives you the highest degree of
  stability and support and comes packed in an archive, with an install
  procedure included.

- If you need some recent feature of pyFormex that is not yet in an official
  release, you may be lucky to find it in the latest :ref:`sec:alpha-release`.

- If the install procedures do not work for you, or you need the
  absolutely latest development code, you can run pyFormex directly
  from the anonymously checked out :ref:`sec:development-version`.

- Finally, if you do not have enough permissions to install the prerequisites,
  or if you do not have a Linux system in the first place, or if you just want
  to try out pyFormex without having to install anything,
  or if you want a portable system that can you take
  with you and run anywhere, choose for the :ref:`sec:bumpix-live-linux` on
  USB stick.

.. _sec:official-release:

Official release
================

pyFormex is software under development, and many users run it directly from
the latest development sources. This holds a certain risk however, because the development version may at times become unstable or incompatible with previous versions and thus break your applications.
At regular times we therefore create official releases, which provide a more stable and better documented and supported version, together with an easy install procedure. 

If you can meet the requirements for using an officially packed release, this is the recommended way to install pyFormex. All the software packages needed to run pyFormex can be obtained for free.

To install an official pyFormex release, you need a working Linux system, root (administrator) privileges to the system, and you need to make sure that the prerequisite packages are installed on the system. 

If you need to install a new Linux system from scratch, and have the choice to
pick any distribution, we highly recommend `Debian GNU/Linux`_ or derivatives.
This is because most of the pyFormex development is done on Debian systems, 
and we will give you `precise install instructions`_ for this system.
Also, the Debian software repositories are amongst the most comprehensive to be found on the Internet. 

Most popular Linux distributions provide appropriately packed recent versions
of these prerequisites, so that you can install them easily from the pacakge manager of your system. In case a package or version is not available for your system, you can always install it from source. We provide the websites where
you can find the source packages.

 
.. _sec:prerequisites:

Prerequisites
-------------

In order to install an official release package of pyFormex, you need to have the following installed (and working) on your computer:

**Python** (http://www.python.org)
   Version 2.4 or higher (2.5 is recommended). Nearly all Linux distributions
   come with Python installed, so this should not be no major obstacle. 

**NumPy** (http://www.numpy.org)
   Version 1.0-rc1 or higher. NumPy is the package used for efficient
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

Additionally, we recommend you to also install the GNU C-compiler and the Python and OpenGL header files. The install procedure needs these to compile the pyFormex acceleration library. While pyFormex can run without the library (Python versions will be substituted
for all functions in the library), using the library will dramatically speed up some low level operations such as drawing, especially when working with large structures . 


.. _`precise install instructions`:

Installing prerequisites on `Debian GNU/Linux`_
...............................................
Debian users should just have to install the packages ``python-numpy`` and  ``python-qt4-gl``. The latter will install ``python-qt4`` and ``python-qt4-gl`` as dependencies. Also, for compiling the acceleration library, you should install 
``python-dev``, ``python-qt4-dev`` and ``libgl1-mesa-dev``.

Other optional packages that might be useful are ``admesh``, ``python-scipy``,
``python-numpy-ext``, ``units``.


.. _sec:downloading:

Download pyFormex
-----------------

Official pyFormex releases can be downloaded from this website:
`Official releases`_. As of the writing of this manual, the latest
release is 0.8. 

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

Finally, a pyFormex installation can usually be removed by giving the command ::

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
   functions. Because the examples programs are usually not installed
   from distribution specific binary packages, and pyFormex uses
   customized names for them, we advise you to use our install
   procedure.

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
install and execute the following command (with root privileges):
``./PACKAGENAME.install all``

All these procedures will install under ``/usr/local``. If you wish to
change this, you will have to change the install procedure. 
The install procedures can also be sued to perform only part of the
installation process. Thus, ``./PACKAGENAME.install get unpack`` will
only download and unpack that package. See the README files and the
install procedures themselves for more info.



.. _sec:alpha-release:

Alpha release
=============

Official release are only created a couple of times per year, because
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


.. _sec:development-version:

Development version
===================

If the install procedures for the packaged releases do not work for
you, or if you want to have the absolutely latest features and bug fixes, 
then you can run pyFormex directly from the development sources.
Obviously, the pyFormex developers use this method, but there are also
several normal users who prefer this, because it allows for easy
updating to the latest version.

To run pyFormex from the development sources you need to have the same
prerequisites installed as for the
:ref:`sec:official-release`. Furthermore, you need the
`Subversion`_ revision control system. You can check whether you have it
by trying the command ``svn help``. If you do not have the command,
you should install Subversion first. Debian and Ubuntu users can just
do ``apt-get install subversion``.

Now you can anonymously check out the latest pyFormex source
from the `SVN repository`_ at the `development`_ site. You can do this
with the command ``svn checkout svn://svn.berlios.de/pyformex/trunk/pyformex
MYDIR/pyformex``, where ``MYDIR`` should be replaced with a path name of your
choice, where you have write permission. 
Most users choose ``pyformex`` as base directory, but this is not
required. You can even check out different versions under
different path names.

Now change into the created ``MYDIR/pyformex`` directory and execute
the command ``./pyformex``. The latest pyFormex version should
startup. The acceleration library will however not be available yet.
To create the library, goto to the ``lib`` subdirectory and execute
the command ``./configure;make``. 

You can make the ``pyformex`` command in your checked out tree
executable from anywhere by creating a symlink under one of the
directories in you ``PATH`` setting. An example command to achieve
this: ``ln -sfn MYDIR/pyformex/pyformex /home/user/bin``

You can update this pyFormex installation at any time to the latest
version by issuing the command ``svn update`` in your ``MYDIR/pyformex``
directory. You can even roll back to any older revision of
pyFormex. Just remember that after updating your sources, the compiled
library could be out of sync with your new sources. Rebuild the
library just like specified above.



.. _sec:bumpix-live-linux:

BuMPix Live Linux system
========================

What is BuMPix
--------------

If you do not have access to a running Linux system, or if the above
installation methods fail for some unknown reason (remember, you can
ask for help on the pyFormex `Forums`_), you can still run pyFormex by
using a `Bumpix Live Linux`_ system. `Bumpix Live` is a full featured
Linux system including pyFormex that can be run from a single
removable medium such as a CD or a USB key.  BuMPix is still an
experimental project, but new versions are already produced at regular
intervals. While these are primarily intended for our students, the
install images are made available for download on the `Bumpix Live Linux`_ FTP
server, so that anyone can use them.  


All you need to use the `Bumpix Live Linux`_ is some proper PC
hardware: the system boots and runs from the removable medium and
leaves everything that is installed on the hard disk of the computer
untouched.


Because the size of the image (since version 0.4) exceeds that of a CD, 
we no longer produce CD-images (.iso) by default, but some older
images remain avaliable on the server. New (reduced) CD
images will only be created on request.
On the other hand, USB-sticks of 2GB and larger have become very
affordable and most computers nowadays can boot from a USB stick.
USB sticks are also far more easy to work with than CD's: you can
create a persistent partition where you can save your changes, while a CD can not be changed.

You can easily take your USB stick with you wherever you go, plug it into any
available computer, and start or continue your previous pyFormex work.
Some users even prefer this way to run pyFormex for that single reason.
The Live system is also an excellent way to test and see what pyFormex can
do for you, without having to install it. Or to demonstrate pyFormex to
your friends or colleagues. 

Download BuMPix
---------------
The numbering scheme of the
BuMPix images is independent from the pyFormex numbering. Just pick
the `latest BuMPix image`_ to get the most recent pyFormex available
on USB stick. After you downloaded the .img file, write it to a USB
stick as an image, not as file! Below, you find instructions on how to do
this on a Linux system or on a Windows platform.

.. warning:: Make sure you've got the device designation correct, or
   you might end up overwriting your whole hard disk! 

Also, be aware that the
USB stick will no longer be usable to store your files under Windows.

Create the BuMPix USB stick under Linux
---------------------------------------

If you have an existing Linux system available, you can write the 
downloaded image to the USB-stick using the command::

  dd if=bumpix-VERSION.img of=USBDEV

where ``bumpix-VERSION.img`` is the downloaded file and USBDEV
is the device corresponding to your USB key. This should be
``/dev/sda`` or ``/dev/sdb`` or, generally, ``/dev/sd?`` where ``?``
is a single character from ``a-z``. The value you should use depends
on your hardware. You can find out the correct value by giving the command ``dmesg``
after you have plugged in the USB key. You will see messages mentioning the
correct ``[sd?]`` device.

The ``dd`` command above will overwrite everything on the specified device,
so copy your files off the stick before you start, and make sure you've got the device designation correct.


Create the BuMPix USB stick under Windows
-----------------------------------------

If you have no Linux machine available to create the USB key, there
are ways to do this under Windows as well. 
We recommend to use `dd for Windows`_. You can then proceed as follows. 

* Download `dd for Windows`_ to a folder, say ``C:\\download\ddWrite``.

* Download the `latest BuMPix image`_ to the same folder.

* Mount the target USB stick and look for the number of the mounted
  USB. This can be done with the command ``c:\\download\ddWrite dd --list``.
  Look at the description (Removable media) and the size to make sure
  you've got the correct harddisk designation (e.g. ``harddisk1``).

* Write the image to the USB stick with the command, substituting the
  harddisk designation found above::

   dd if=c:\download\ddwrite\bumpix-0.4-b1.img of=\\?\device\harddisk1\partition0 bs=1M --progress

The ``dd`` command above will overwrite everything on the specified device,
so copy your files off the stick before you start, and make sure you've got the device designation correct.


Buy a USB stick with BuMPix 
--------------------------- 

Alternatively,

* if you do not succeed in properly writing the image to a USB key, or
* if you just want an easy solution without any install troubles, or
* if you want to financially support the further development of pyFormex, or
* if you need a large number of pyFormex USB installations,

you may be happy to know that we can provide ready-made BuMPix USB
sticks with the ``pyformex.org`` logo at a cost hardly exceeding that
of production and distribution.
If you think this is the right choice for you, just `email us`_ for a quotation.

.. image:: ../website/images/pyformex_ufd.jpg
   :align: center
   :alt: pyFormex USB stick with BuMPix Live Linux


Boot your BuMPix system
-----------------------
Once the image has been written, reboot your computer from the USB
stick. You may have to change your BIOS settings or use the boot menu
to do that. On success, you will have a full Linux system running,
containing pyFormex ready to use. There is even a start button in the 
toolbar at the bottom.


.. _sec:pyformex-nonlinux:

Running pyFormex on non-Linux systems
=====================================

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

.. note:: **pyFormex on Windows** 
   Lately there have been some successful attempts to get the basic 
   functionality of pyFormex running on Windows. Thomas Praet has 
   compiled `this document
   <ftp://ftp.berlios.de/pub/pyformex/Install_pyFormex_on_Windows.html>`_ on how to proceed.
   And a thread for discussing the installation on Windows has been
   opened in the `forums. <http://developer.berlios.de/forum/forum.php?thread_id=37162&forum_id=8348>`_
 

There may be a few things that have to be changed to successfully run
pyFormex on other platforms (especially on Windows), but it should all
be rather obvious.  If you have some programming experience on your
platform, and you want to give it a try, please do so. We certainly
are willing to help where we can. And
we are certainly interested in feedback about your attempt, whether
successful or not.




.. End

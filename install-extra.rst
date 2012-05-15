.. $Id$   *- rst -*-

.. |date| date::

==============================
Installation of extra software
==============================
:Date: |date|
:Author: benedict.verhegghe@ugent.be

Introduction
------------

We are currently changing some installation procedures for external
software packages (in the ``pyformex/extra`` directory).
The goal is to

- maximally use available packages from Linux distributions,
- allow some packages to be built and included in the Debian packages,
- allow users of an SVN version to build the packages by just clicking on
  a button in the pyFormex GUI.


Current status
--------------

'gts', 'postabq' and 'dxfparser' have been converted. They can be installed
from the *Help->Install Externals* menu. Of course you need root privileges.

gts
...

There are packages for 'gts' available, including a 'bin' package containing
some, but not all of the example gts commands that we are using. These packages
are built from more recent sources. The new installation script only builds
the missing executables (*gtsrefine, gtscoarsen, gtssmooth, gtsset*) and a new
one that we added: *gtsinside*. These programs (and no other files) will be installed under ``/usr/local/bin``.

Requirements: libgts-dev, libglib2.0-dev, libgts-bin

To remove the old local gts installation under /usr/local, you have to::

  rm -f /usr/local/bin/gts*
  rm -f /usr/local/bin/stl2gts
  rm -f /usr/local/include/gts*
  rm -f /usr/local/lib/libgts*
  rm -rf /usr/local/lib/python2.?/dist-packages/gts


dxfparser
.........

The executable will now be installed as ``pyformex-dxfparser`` under ``/usr/local/bin``.

Requirements: libdxflib-dev

To remove the old local dxfparser installation, do::

  rm -f /usr/local/bin/dxfparser


postabq
.......

The executable will now be installed as ``pyformex-postabq`` under ``/usr/local/bin``.

Requirements: none

To remove the old local postabq installation, do::

  rm -f /usr/local/bin/postabq

On the BuMPer cluster, the command remains `postabq` until the installation will be upgraded. We will then install a symlink to keep the old name.


.. End


.. This may look like plain text, but really is -*- rst -*-
  
..
  This file is part of the pyFormex project.
  pyFormex is a tool for generating, manipulating and transforming 3D
  geometrical models by sequences of mathematical operations.
  Home page: http://pyformex.org
  Project page:  https://savannah.nongnu.org/projects/pyformex/
  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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
  
  

.. include:: links.inc

====================================
pyFormex User Meeting 8 (2009-05-11)
====================================


Place and Date
--------------
These are the minutes of the pyFormex User Meeting of Monday May 11, 2009, at the Institute Biomedical Technology (IBiTech), Ghent University, Belgium.


Participants
------------
The following pyFormex developers, users and enthusiasts were present.

- Benedict Verhegghe, chairman
- Matthieu De Beule
- Peter Mortier
- Sofie Van Cauter
- Gianluca De Santis
- Tomas Praet, secretary


Apologies
---------
- None


Minutes of the previous meeting
-------------------------------
- The minutes of the previous meeting were not yet available. They will be approved later and put on the `pyFormex User Meeting page`_.


Agenda and discussion
---------------------

* The new pyFormex USB stick was demonstrated by Benedict. The most
  important new feature is that the USB stick can now be booted as
  persistent. This allows for storage of settings and documents
  directly on the bootable memory stick, without losing them at the
  next reboot. The other modes are still available at the boot screen
  (e.g. for when you're giving a presentation and do not want to
  compromise your files), but the persistent mode is set as
  default. Making a bootable USB stick that's persistent requires two
  partitions, hence the instructions on the web site `Debian Live
  Project`_ are no longer sufficient for this purpose.

  One of the greatest advantages is the possibility of running
  pyFormex directly from the checked-out pyFormex sources, which
  allows you to upgrade at any time to the latest version (or even to
  go back to any older version).

* Running pyFormex from SVN sources.
  First, you have to create a checked-out version. Check in your
  browser that you are connected. If not, the command command::

    sudo dhclient eth1

  may help.

  
  Check out the the pyFormex version with the commands::

    cd
    sudo svn checkout svn://svn.berlios.de/pyformex/trunk pyformex-svn

  This will create a directory ``pyformex-svn`` with the latest pyFormex
  version in your home directory. The command
  ``./pyformex-svn/pyformex/pyformex`` will start that version.
  You may create a link in your ``bin`` directory to make it the
  default ``pyformex``::

    ln -s ../trunk/pyformex/pyformex bin (creates a link)

  Log out and in again to activate the link.

  The version checked out with the command above is an anopnymous
  checkout: you can change it, but you can not check in those changes
  into the official pyFormex repository. 

  If you are a developer, you can retrieve a developer version with::

    svn checkout \
      svn+ssh://developername@svn.berlios.de/svnroot/repos/pyformex/trunk\
      pyformex-svn


  Once you have created your local pyFormex checkout, upgrading
  becomes very easy. Simply go to the ``pyformex-svn`` directory and
  use the command::
  
     svn up


* BuMPix netboot server: in the future, booting from an image on the web will be made possible.

* New functionalities in pyFormex

  - Rectangle zooming: a new button is added in the gui to allow the user to select the region on which he wants to zoom by dragging a rectangle on the canvas. After the zoom took place you will again retrieve the normal mouse key bindings, so if you want to do more rectangle zooming, you will have to push the button again before each zoom. This might change in future developments. Inspired from the rectangle zooming, the other zooming modes retrieved an update as well.
  - Projects are now stored in a new compressed format. Older formats
    can still be read, but the old format can not be written anymore.
  - A project can record a script as its 'autoFile', meaning that this
    cript will be executed whne the project is opened. You can set
    this script fromt the ``File -> Set current script as AutoFile``
    option. In future, the autoscript might be stored inside the project.



* Installing external software (gts, gl2ps, tetgen, calpy, admesh, units)

  All the interesting external tools that are not as easily installed with the normal procedures are combined in a separate folder: ``.../pyformex/external``

  Installation files are available here, as well as detailed installation instructions. Currently, the available installation files are

  - Calpy
  - Gts
  - Pygl2ps (specially adapted to pyFormex; used to export images)
  - Tetgen

  Installing these programs normally requires root permissions.


* Two dimensional text rendering was formally done using the GLUT library, but in the newest pyFormex versions this was switched to the QT library. Should you still want to make use of the old text rendering features, you can include gluttext.py in your scripts. The new system for writing 2D text on the canvas allows the usage of tons of different fonts, special characters and foreign languages.

* 3D font manipulations using FTGL

* Up till now pyFormex has been using rather old OpenGL functions. With the focus of OpenGL shifting to game development in stead of CAD, more and more important pyFormex functions will disappear on the middle long term. Suggested solutions are:

* Some consideration was given to the changed policy of OpenGL with
  regards to legacy features. The conclusion is that no immediate
  action is needed, but we will probably start adapting the pyFormex
  rendering engine after version 1.0 is reached. 

* Gianluca gave a nice demonstration of his use of pyFormex for the user-friendly creation of hexahedral meshes of arterial bifurcations.

Varia
-----
- None

Date of the next meeting
------------------------
The next meeting will be held at IBiTech somewhere in June 2009.

.. The following directive makes sure the targets are included in footnotes.

.. target-notes::


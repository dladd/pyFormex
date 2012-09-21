.. $Id$  -*- rst -*-
  
..
  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
  pyFormex is a tool for generating, manipulating and transforming 3D
  geometrical models by sequences of mathematical operations.
  Home page: http://pyformex.org
  Project page:  http://savannah.nongnu.org/projects/pyformex/
  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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
  
  

.. pyFormex documentation reference manual master file

.. include:: defines.inc
.. include:: links.inc



.. _cha:reference:

##########################
 pyFormex reference manual
##########################

.. topic:: Abstract

   This is the reference manual for pyFormex |release|.
   It describes most of the classes and functions
   defined in the pyFormex modules. It was built automatically from
   the pyFormex sources and is therefore the ultimate reference
   document if you want to look up the precise arguments (and their meaning) 
   of any class constructor or function in pyFormex. The :ref:`genindex`
   and :ref:`modindex` may be helpful in navigating through this
   document.


This reference manual describes the classes in functions defined in
most of the pyFormex modules. It was built automatically from the docstrings in
the pyFormex sources. The pyFormex modules are placed in three paths:

- ``pyformex`` contains the core functionality, with most of the
  geometrical transformations, the pyFormex scripting language and utilities,
- ``pyformex/gui`` contains all the modules that form the interactive
  graphical user interface,
- ``pyformex/plugins`` contains extensions that are not considered to
  be essential parts of pyFormex. They usually provide additional
  functionality for specific applications. 

Some of the modules are loaded automatically when pyFormex is
started. Currently this is the case with the modules
:mod:`coords`, :mod:`formex`, :mod:`arraytools`, :mod:`script` and, if the GUI is used, :mod:`draw` and :mod:`colors`.
All the public definitions in these modules are available to pyFormex
scripts without explicitly importing them. Also available is the complete 
:mod:`numpy` namespace, because it is imported by :mod:`arraytools`.

The definitions in the other modules can only be accessed using the
normal Python ``import`` statements.

.. _sec:autoloaded-modules:

Autoloaded modules
==================

The definitions in these modules are always available to your scripts, without
the need to explicitely import them.

.. toctree::
   :maxdepth: 2
   :numbered: 0

   ref/coords
   ref/formex

   ref/arraytools
   ref/script
   ref/draw
   ref/colors


.. _sec:core-modules:

Other pyFormex core modules
===========================

Together with the autoloaded modules, the following modules located under the
main pyformex path are considered to belong to the pyformex core functionality. 

.. toctree::
   :maxdepth: 2
   :numbered: 6

   ref/geometry
   ref/connectivity
   ref/adjacency
   ref/elements
   ref/mesh
   ref/simple
   ref/project
   ref/utils
   ref/geomtools
   ref/fileread


.. _sec:gui-modules:

pyFormex GUI modules
====================

These modules are located under pyformex/gui.

.. toctree::
   :maxdepth: 1
   :numbered: 13

   ref/widgets
   ref/menu
   ref/colorscale
   ref/actors
   ref/decors
   ref/marks
   ref/gluttext
   ref/canvas
   ref/viewport
   ref/camera
   ref/image
   ref/imagearray
   ref/imageViewer
   ref/scriptMenu
   ref/toolbar


.. _sec:plugins-modules:

pyFormex plugins
================

Plugin modules extend the basic pyFormex functions to variety of 
specific applications. Apart from being located under the pyformex/plugins
path, these modules are in no way different from other pyFormex modules. 

.. toctree::
   :maxdepth: 1
   :numbered: 28

   ref/calpy_itf
   ref/cameratools
   ref/curve
   ref/datareader
   ref/draw2d
   ref/dxf
   ref/export
   ref/f2flu
   ref/mesh_ext
   ref/trisurface
   ref/nurbs
   ref/isopar
   ref/section2d
   ref/inertia
   ref/units
   ref/properties
   ref/fe
   ref/fe_abq
   ref/fe_post
   ref/postproc
   ref/flavia
   ref/lima
   ref/turtle
   ref/tetgen
   ref/tools
   ref/objects
   ref/plot2d
   ref/pyformex_gts

.. _sec:menu-modules:

pyFormex plugin menus
=====================

Plugin menus are optionally loadable menus for the pyFormex GUI, providing
specialized interactive functionality. Because these are still under heavy
development, they are currently not documented. Look to the source code or
try them out in the GUI. They can be loaded from the File menu option and 
switched on permanently from the Settings menu.

Currently avaliable:

- geometry_menu
- formex_menu
- surface_menu
- tools_menu
- draw2d_menu
- nurbs_menu
- dxf_tools
- jobs menu
- postproc_menu

.. _sec:tools-modules:

pyFormex tools
==============

The main pyformex path contains a number of modules that are not
considered to be part of the pyFormex core, but are rather tools that
were used in the implementation of other modules, but can also be useful
elsewhere. 

.. toctree::
   :maxdepth: 1
   :numbered: 52

   ref/olist
   ref/mydict
   ref/odict
   ref/track
   ref/collection
   ref/config
   ref/flatkeydb
   ref/sendmail
   ref/timer

.. End


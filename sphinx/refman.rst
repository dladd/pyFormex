.. $Id$  -*- rst -*-

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



.. _sec:autoloaded_modules:

**Autoloaded modules**

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


**Other pyFormex core modules**

Together with the autoloaded modules, the following modules located under the
main pyformex path are considered to belong to the pyformex core functionality. 

.. toctree::
   :maxdepth: 2
   :numbered: 6

   ref/geometry
   ref/connectivity
   ref/simple
   ref/project
   ref/utils
   ref/elements


**pyFormex GUI modules**

These modules are located under pyformex/gui.

.. toctree::
   :maxdepth: 1
   :numbered: 12

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
   ref/imagecolor
   ref/imageViewer
   ref/scriptMenu
   ref/toolbar


**pyFormex plugins**

Plugin modules extend the basic pyFormex functions to variety of 
specific applications. Apart from being located under the pyformex/plugins
path, these modules are in no way different from other pyFormex modules. 

.. toctree::
   :maxdepth: 1
   :numbered: 28

   ref/curve
   ref/mesh
   ref/surface
   ref/geomtools
   ref/isopar
   ref/section2d
   ref/inertia
   ref/units
   ref/datareader
   ref/properties
   ref/fe
   ref/fe_abq
   ref/fe_post
   ref/postproc
   ref/lima
   ref/turtle
   ref/dxf
   ref/export
   ref/tetgen
   ref/tools
   ref/objects
..   ref/formex_menu
..   ref/tools_menu
..   ref/surface_menu
..   ref/postproc_menu

   
**pyFormex tools**

The main pyformex path contains a number of modules that are not
considered to be part of the pyFormex core, but are rather tools that
were used in the implementation of other modules, but can also be useful
elsewhere. 

.. toctree::
   :maxdepth: 1
   :numbered: 49

   ref/olist
   ref/mydict
   ref/odict
   ref/collection
   ref/config
   ref/flatkeydb
   ref/sendmail
   ref/timer
   ref/misc

.. End


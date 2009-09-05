.. $Id$  -*- rst -*-

.. pyFormex documentation reference manual master file

.. include:: defines.inc
.. include:: links.inc


.. _cha:reference:

===========================
|pyformex| reference manual
===========================

This is the reference manual for |pyformex| |release|.

When pyformex is started, it always loads the modules
:mod:`coords`, :mod:`formex`, :mod:`arraytools`, :mod:`script` and (if the GUI is used) :mod:`draw` and :mod:`colors`.
All the public definitions in these modules are available to |pyformex|
scripts without explicitly importing them. Also available is the complete 
:mod:`numpy` namespace.

All the other modules need to be accessed using the normal Python import
statements.


Autoloaded modules
------------------
The definitions in these modules are always available to your scripts, without
the need to explicitely import them.

.. toctree::
   :maxdepth: 1

   ref/coords
   ref/formex
   ref/arraytools
   ref/script
   ref/draw
   ref/colors


Other |pyFormex| core modules
-----------------------------
Together with the autoloaded modules, the following modules located under the
main pyformex path are considered to belong to the pyformex core functionality. 

.. toctree::
   :maxdepth: 1

   ref/connectivity
   ref/simple
   ref/project
   ref/utils


|pyFormex| GUI modules
----------------------
These modules are located under pyformex/gui.

.. toctree::
   :maxdepth: 1

   ref/colorscale
   ref/image
   ref/widgets


|pyFormex| plugins
------------------
Plugin modules extend the basic |pyformex| functions to variety of 
specific applications. Apart from being located under the pyformex/plugins
path, these modules are in no way different from other |pyformex| modules. 

.. toctree::
   :maxdepth: 1

   ref/curve
   ref/fe
   ref/inertia
   ref/isopar
   ref/lima
   ref/mesh
   ref/properties
   ref/section2d
   ref/surface
   ref/turtle
   ref/units

   
|pyFormex| tools
----------------
The main pyformex path contains a number of modules that are not
considered to be part of the |pyformex| core, but are rather tools that
were used in the implementation of other modules, but can also be useful
elsewhere. 

.. toctree::
   :maxdepth: 1

   ref/olist
   ref/mydict
   ref/odict

.. End


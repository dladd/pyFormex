.. This may look like plain text, but really is -*- rst -*-

.. include:: links.inc

====================================
pyFormex User Meeting 9 (2010-01-19)
====================================


Place and Date
==============
These are the minutes of the pyFormex User Meeting of Tuesday January 19, 2010, at the Institute Biomedical Technology (IBiTech), Ghent University, Belgium.


Participants
============
The following pyFormex developers, users and enthusiasts were present.

- Benedict Verhegghe
- Matthieu De Beule
- Peter Mortier
- Sofie Van Cauter
- Gianluca De Santis, secretary
- Tomas Praet
- Koen Van Canneyt


Apologies
=========
- None


Minutes of the previous meeting
===============================
- The minutes of the previous meeting were approved and will be put on the `pyFormex User Meeting page`_.


Agenda and discussion
=====================

Documentation
------------
- The examples available in pyFormex should be incorporated in the
  documentation and explained step by step. An automatic generation of 
  a pictured documentation by running all the examples would help.
- The same examples should be available both in the pyFormex release
  and on the website. On the website, a gallery of screeshots without code
  needs to be added.
- To understand a function, the examples containing that functions
  should be selected by keyword.


Draw 2D: interactive drawing
-------------
- Drawing points, lines, circles and curves is now possible clicking on the screen.
  These objects become global variables.
- Displaying the coordinate on the canvas may be useful.
- Selecting coords and elements of an object by a closed PolyLine need to be included.
- Currently this is limited to a plane (arbitrarily oriented) but it can be extended
  to the 3D space by using 2 perpendicular viewports.

Different menus need integration
------------
- Currently, different menus (Formex, Surface, Mesh etc.) do not interact: the 
   variables are not shared among the menus. This need to be fixed in 
   order to select an object only once.

Model class and Meshes
------------
- A new Model class has been chosed to together different instances of the mesh class.
- In the mesh plugins, some functions have been developed to convert the mesh element type
  (quad4 to tri3, quad4 to quad8). This functions need to be extended in 3D (eg Hex to Tet).

Canvas and drawing
------------
- At the moment, Formices are stored as Actors to be drawn. Grouping
   actors into layers can make the draing more versatile: draw smooth one layer
   and flat an other one.

pyFormex and Windows
------------
- Thomas Praet is the pioneer of pyFormex on Windows platform. pyFormex on Windows run slower
   becuase the C libraries are missing and not all its functionalitis are running.

New Demos: spirals and sweep
------------
- A new example has been added, showing the creation of spirals (and helics) and
   the new sweep function. The sweep function works on any curve.

Quadratic Bezier Spline and Fonts
------------
- The pyFormex Lustrum (5 th anniversary) has been created using quadratic bezier splines.
  Quadratic Bezier Splines have been added to the pugins.curves.
  It can be the best way to generate versatile fonts.

From STL surfaces of a vessel bifurcation to Hex mesh
------------
- pyFormex potentials as mesher have been shown by  Benedict,
  who has taken some meshing tools from Gianluca and incorporated into
  a user-friendly menu. The result is terrific and will be partially distributed.
  

Varia
=====

- The pyFormex meeting starts being to long. Need for time schedule.


Date of the next meeting
========================
The next meeting will be held at IBiTech in Feb 2010.


.. The following directive makes sure the targets are included in footnotes.

.. target-notes::


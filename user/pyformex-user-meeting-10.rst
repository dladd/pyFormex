.. This may look like plain text, but really is -*- rst -*-

.. include:: links.inc

=====================================
pyFormex User Meeting 10 (2010-01-19)
=====================================


Place and Date
==============
These are the minutes of the pyFormex User Meeting of Tuesday January 19, 2010, at the Institute Biomedical Technology (IBiTech), Ghent University, Belgium.


Participants
============
The following pyFormex developers, users and enthusiasts were present.

- Matthieu De Beule
- Gianluca De Santis, secretary
- Peter Mortier
- Tomas Praet
- Koen Van Canneyt
- Sofie Van Cauter
- Benedict Verhegghe


Apologies
=========
- None


Minutes of the previous meeting
===============================
- The minutes of the previous meeting were approved and will be put on the `pyFormex User Meeting page`_.


Agenda and discussion
=====================

Documentation
-------------

- The examples available in pyFormex should be incorporated in the
  documentation and explained step by step. An automatic generation of 
  a pictured documentation by running all the examples would help.
- The same examples should be available both in the pyFormex release
  and on the website. On the website, a gallery of screeshots without code
  needs to be added.
- To help in learning the use of a function, the examples containing
  that function should be selectable from an index.
- Automated generation of these features may be implemented by
  specifications in the example's docstring.


Draw 2D: interactive drawing
----------------------------
- The new `draw2d` plugin provides some interactive drawing
  functionality: points, lines, circles and curves can be drawn by
  clicking with the mouse. The created objects become global
  variables.
- Drawing is done in 2D on an x-y-plane at any z-value.
- What further functionality could/should pyFormex provide:
  
  - Displaying the coordinate values.
  - Entering coordinates instead of clicking.
  - Selecting points of an existing object.
  - Editing/Deleting points 
  - Transformation of the drawing plane to any position and orientation
    in space.
  - Combining multiple viewports may even allow full 3D interactive drawing.

Different menus need integration
--------------------------------
- Currently, different plugin menus (Formex, Surface, Mesh etc.) do
  not interact (well): variables are not shared among the menus and
  different incompatible techniques are used. This needs to be fixed
  by creating a single scheme for creating/using/selecting/drawing objects. 
 

FE Models and Meshes
--------------------
- The Model class will be changed to use Mesh class objects as its
  constituant objects.
- The Mesh plugin provides an efficient and generic machine for mesh
  conversions. Currently implemented are a whole bunch of 2D element
  type conversions and subdivisions (e.g. `quad4` to `tri3`, `quad4`
  to `quad8`). This needs to be extended with 3D element conversions
  (e.g. `hex8` to `tet4`).

The pyFormex rendering engine
-----------------------------
- It is planned to create a layer between the geometry (Formex)
  objects and their rendering (Actor), to enable advanced and more
  flexible functionality:

  - Creating a stronger reference mechanism between actor and
  - Hide and show objects by groups.
  - Toggling decorations on and off.
  - Displaying bitmaps on the canvas.
  - Faster drawing mode switching.

pyFormex on Windows
-------------------
- Thomas Praet, a pioneer of pyFormex on Windows platform, compiled
  some instructions of how to get pyFormex running on Windows.
  pyForme runs slowerx on Windows because the compiled C-libraries are
  missing. Also, some of the functionality will not be available. Still, the
  basic parts are mostly running well. An announcement will be made on
  the website and a link to the install instructions.


Demos
-----
- New examples Spiral and Sweep illustrate the creation of spiral
  curves and sweeping a cross section along a curve.

- Mesh conversions were demonstrated using the Mesh menu.

- The pyFormex Lustrum (5th anniversary) example presents a '5' shape
  created with quadratic bezier splines.  Quadratic Bezier Splines have
  been added to the `curves` plugin. They are often used to represent
  character outlines in scalable fonts. One day, pyFormex may be able
  to draw text using any TrueType font available on your computer.
  And not just the original font, but all kinds of transformations of
  such fonts.

- pyFormex as special purpose mesher: from STL surfaces of a vessel 
  bifurcation to Hexaeder mesh. Benedict added a user-friendly menu
  to Gianluca's meshing tools, resulting in a splendid meshing tool.
  Parts of it will be included in future pyFormex versions.

- interactive drawing tools (currently under development,see above), applied
  to hexahedral meshing of bifurcations. Expect more from this in future.
  

Varia
=====


Date of the next meeting
========================
The next meeting will be held at IBiTech in March 2010.


.. The following directive makes sure the targets are included in footnotes.

.. target-notes::


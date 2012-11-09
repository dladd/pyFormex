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
.. include:: <isonum.txt>


.. _cha:examples:

************************
pyFormex example scripts
************************

.. warning:: This document still needs some cleanup! 

Sometimes you learn quicker from studying an example than from reading a 
tutorial or user guide. To help you we have created this collection of 
annotated examples. Beware that the script texts presented in this document
may differ slightly from the corresponding example coming with the pyFormex 
distribution.

.. _sec:creating-geometry:

WireStent
=========

To get acquainted with the modus operandi of pyFormex, 
the :file:`WireStent.py` script is studied step by step. The lines are numbered
for easy referencing, but are not part of the script itself.


.. literalinclude:: static/scripts/WireStent.py
   :language: python
   :linenos:

As all pyFormex scripts, it starts with a comments line holding the word
``pyformex`` (line 1). This is followed more comments lines specifying the
copyright and license notices. If you intend to distribute your scripts, you should give these certainly special consideration.

Next is a documentation string explaining
the purpose of the script (lines 25-30). 
The script then starts by importing all definitions from
other modules required to run the :file:`WireStent.py`
script (line 32).


Subsequently, the class ``DoubleHelixStent`` is defined which
allows the simple use of the geometrical model in other scripts for e.g.
parametric, optimization and finite element analyses of braided wire stents.
Consequently, the latter scripts do not have to contain the wire stent geometry
building and can be condensed and conveniently arranged. The definition of the
class starts with a documentation string, explaining
its aim and functioning (lines 34-60).


The constructor ``__init__`` of the ``DoubleHelixStent`` class requires 8
arguments (line 61):

* stent external diameter :math:`De` (mm).

* stent length :math:`L` (mm).

* wire diameter :math:`d` (mm).

* Number of wires in one spiral set, i.e. wires with the same orientation,
  :math:`nx` (-).

* Pitch angle :math:`\beta` (:math:`\deg`).

* Extra radial distance between the crossing wires :math:`ds` (mm). By default,
  :math:`ds` is [0.0]mm for crossing wires, corresponding with a centre line
  distance between two crossing wires of exactly :math:`d`.

* Number of elements in a strut, i.e. part of a wire between two crossings,
  :math:`nb` (-). As every base element is a straight line, multiple elements are
  required to approximate the curvature of the stent wires. The default value of 4
  elements in a strut is a good assumption.

* If ``connectors=True``, extra elements are created at the positions where
  there is physical contact between the crossing wires. These elements are
  required to enable contact between these wires in finite element analyses.

The virtual construction of the wire stent structure is defined by the following
sequence of four operations: (i) Creation of a nearly planar base module of two
crossing wires; (ii) Extending the base module with a mirrored and translated
copy; (iii) Replicating the extended base module in both directions of the base
plane; and (iv) Rolling the nearly planar grid into the cylindrical stent
structure, which is easily parametric adaptable.


Creating the base module
------------------------
(lines 63-71)

Depending on the specified arguments in the constructor, the mean stent diameter
:math:`D`, the average stent radius :math:`r`, the ``bump`` or curvature of the
wires :math:`dz`, the pitch :math:`p` and the number of base modules in the
axial direction :math:`ny` are calculated with the following script. As the wire
stent structure is obtained by braiding, the wires have an undulating course and
the ``bump dz`` corresponds to the amplitude of the wave. If no extra distance
:math:`ds` is specified, there will be exactly one wire diameter between the
centre lines of the crossing wires. The number of modules in the axial direction
:math:`ny` is an integer, therefore, the actual length of the stent model might
differ slightly from the specified, desired length :math:`L`. However, this
difference has a negligible impact on the numerical results.


Of now, all parameters to describe the stent geometry are specified and
available to start the construction of the wire stent. Initially a simple Formex
is created using the ``pattern()``\ -function: a straigth line segment of length
1 oriented along the X-axis (East or :math:`1`\ -direction). The ``replic()``\
-functionality replicates this line segment :math:`nb` times with step 1 in the
X-direction (:math:`0`\ -direction). Subsequently, these :math:`nb` line
segments form a new Formex which is given a one-dimensional ``bump`` with the
``bump1()``\ -function. The Formex undergoes a deformation in the Z-direction
(:math:`2`\ -direction), forced by the point ``[0,0,dz]``. The ``bump``
intensity is specified by the quadratic ``bump_z`` function and varies along the
X-axis (:math:`0`\ -axis). The creation of this single bumped strut, oriented
along the X-axis is summarized in the next script and depicted in figures
:ref:`fig:straight`, :ref:`fig:replicated` and :ref:`fig:bumped`,.


.. _`fig:straight`:

.. figure:: images/WireStentDemot2Step01.*
   :align: center
   :width: 300px
   :alt: straight line segment

   A straight line segment

.. _`fig:replicated`:

.. figure:: images/WireStentDemot2Step02.*
   :align: center
   :width: 300px
   :alt: line segment with replications

   The line segment with replications

.. _`fig:bumped`:

.. figure:: images/WireStentDemot2Step03.*
   :align: center
   :width: 300px
   :alt: bumped line segment

   A bumped line segment


The single bumped strut (``base``) is rescaled homothetically in the XY-plane to
size one with the ``scale()``\ -function. Subsequently, the ``shear()``\
-functionality generates a new ``NE`` Formex by skewing the ``base`` Formex in
the Y-direction (:math:`1`\ -direction) with a ``skew`` factor of :math:`1` in
the YX-plane. As a result, the Y-coordinates of the ``base`` Formex are altered
according to the following rule: :math:`y_2 = y_1 + skew \* x_1`. Similarly a
``SE`` Formex is generated by a ``shear()`` operation on a mirrored copy of the
``base`` Formex. The ``base`` copy, mirrored in the direction of the XY-plane
(perpendicular to the :math:`2`\ -axis), is obtained by the ``reflect()``
command. Both Formices are given a different property number by the
``setProp()``\ -function, visualised by the different color codes in Figure
:ref:`fig:unit_cell` This number can be used as an entry in a database, which holds some
sort of property. The Formex and the database are two seperate entities, only
linked by the property numbers. The ``rosette()``\ -function creates a unit cell
of crossing struts by :math:`2` rotational replications with an angular step of
[180]:math:`\deg` around the Z-axis (the original Formex is the first of the
:math:`2` replicas). If specified in the constructor, an additional Formex with
property :math:`2` connects the first points of the ``NE`` and ``SE`` Formices.

(lines 72-83)

.. _`fig:rescaled`:

.. figure:: images/WireStentDemot2Step04.*
   :align: center
   :width: 300px
   :alt: rescaled bumped line segment

   Rescaled bumped strut

.. _`fig:mirrored`:

.. figure:: images/WireStentDemot2Step07.*
   :align: center
   :width: 300px
   :alt: mirrored and skewed bumped strut

   Mirrored and skewed bumped strut

.. _`fig:unit_cell`:

.. figure:: images/WireStentDemot2Step09.*
   :align: center
   :width: 300px
   :alt: unit cell of wires and connectors

   Unit cell of crossing wires and connectors


Extending the base module
-------------------------

Subsequently, a mirrored copy of the base cell is generated (Figure 
:ref:`fig:mirrored_unit_cell`). Both Formices are
translated to their appropriate side by side position with the ``translate()``\
-option and form the complete extended base module with 4 by 4 dimensions as
depicted in Figure :ref:`fig:base_module`.
Furthermore, both Formices are defined as an
attribute of the ``DoubleHelixStent`` class by the ``self``\ -statement,
allowing their use after every ``DoubleHelixStent`` initialisation. Such further
use is impossible with local variables, such as for example the ``NE`` and
``SE`` Formices.

(lines 84-89)

.. _`fig:mirrored_unit_cell`:

.. figure:: images/WireStentDemot2Step10.*
   :align: center
   :width: 300px
   :alt: mirrored unit cell

   Mirrored unit cell

.. _`fig:base_module`:

.. figure:: images/WireStentDemot2Step11.*
   :align: center
   :width: 300px
   :alt: completed base module

   Completed base module



Full nearly planar pattern
--------------------------

The fully nearly planar pattern is obtained by copying the base module in two
directions and shown in Figure :ref:`fig:full_plane`. ``replic2()`` generates this
pattern with :math:`nx` and :math:`ny` replications with steps :math:`dx` and
:math:`dy` in respectively, the default X- and Y-direction.

(lines 90-93)

.. _`fig:full_plane`:

.. figure:: images/WireStentDemot2Step12.*
   :align: center
   :width: 300px
   :alt: Full planar topology

   Full planar topology


.. _`fig:full_plane_orthoview`:

.. figure:: images/WireStentDemot2Step13.*
   :align: center
   :width: 300px
   :alt: orthogonal view of the full planar topology

   Orthogonal view of the full planar topology


Cylindrical stent structure
---------------------------

Finally the full pattern is translated over the stent radius :math:`r` in
Z-direction and transformed to the cylindrical stent structure by a coordinate
transformation with the Z-coordinates as distance :math:`r`, the X-coordinates
as angle :math:`\theta` and the Y-coordinates as height :math:`z`. The
``scale()``\ -operator rescales the stent structure to the correct circumference
and length. The resulting stent geometry is depicted in Figure :ref:`fig:stent`.
(lines 94-96)

In addition to the stent initialization, the ``DoubleHelixStent`` class script
contains a function ``all()`` representing the complete stent Formex.
Consequently, the ``DoubleHelixStent`` class has four attributes: the Formices
``cell1``, ``cell2`` and ``all``; and the number :math:`ny`.
(lines 97-100)

.. _`fig:stent`:

.. figure:: images/WireStentDemot2Step16.*
   :align: center
   :width: 300px
   :alt: cylindrical stent

   Cylindrical stent

.. _`fig:stent_ortho`:

.. figure:: images/WireStentDemot2Step15.*
   :align: center
   :width: 300px
   :alt: orthogonal view of thecylindrical stent

   Orthogonal view of the cylindrical stent

Parametric stent geometry
-------------------------

An inherent feature of script-based modeling is the possibility of easily
generating lots of variations on the original geometry. This is a huge advantage
for parametric analyses and illustrated in figures 
:ref:`fig:stent_D16L40d22n6b25`: these wire
stents are all created with the same script, but with other values of the
parameters :math:`De`, :math:`nx` and :math:`\beta`. As the script for building
the wire stent geometry is defined as a the ``DoubleHelixStent`` class in the
(:file:`WireStent.py`) script, it can easily be imported for e.g. this purpose.

.. _`fig:stent_D16L40d22n6b25`:

.. figure:: images/WireStentD16L40d22n6b25.*
   :align: center
   :width: 300px
   :alt: Stent variant 

   Stent variant with :math:`De=16, nx=6, \beta=25`


.. _`fig:stent_D16L40d22n6b50`:

.. figure:: images/WireStentD16L40d22n6b50.*
   :align: center
   :width: 300px
   :alt: Stent variant 

   Stent variant with :math:`De=16, nx=6, \beta=50`


.. _`fig:stent_D16L40d22n10b25`:

.. figure:: images/WireStentD16L40d22n10b25.*
   :align: center
   :width: 300px
   :alt: Stent variant 

   Stent variant with :math:`De=16, nx=10, \beta=25`


.. _`fig:stent_D16L40d22n10b50`:

.. figure:: images/WireStentD16L40d22n10b50.*
   :align: center
   :width: 300px
   :alt: Stent variant 

   Stent variant with :math:`De=16, nx=10, \beta=50`


.. _`fig:stent_D32L40d22n6b25`:

.. figure:: images/WireStentD32L40d22n6b25.*
   :align: center
   :width: 300px
   :alt: Stent variant 

   Stent variant with :math:`De=32, nx=6, \beta=25`


.. _`fig:stent_D32L40d22n6b50`:

.. figure:: images/WireStentD32L40d22n6b50.*
   :align: center
   :width: 300px
   :alt: Stent variant 

   Stent variant with :math:`De=32, nx=6, \beta=50`


.. _`fig:stent_D32L40d22n10b25`:

.. figure:: images/WireStentD32L40d22n10b25.*
   :align: center
   :width: 300px
   :alt: Stent variant 

   Stent variant with :math:`De=32, nx=10, \beta=25`


.. _`fig:stent_D32L40d22n10b50`:

.. figure:: images/WireStentD32L40d22n10b50.*
   :align: center
   :width: 300px
   :alt: Stent variant 

   Stent variant with :math:`De=32, nx=10, \beta=50`

.. include:: static/scripts/WireStentParametricExample.py
   :literal:

Obviously, generating such parametric wire stent geometries with classical CAD
methodologies is feasible, though probably (very) time consuming. However, as
provides a multitude of features (such as parametric modeling, finite element
pre- and postprocessing, optimization strategies, etcetera) in one single
consistent environment, it appears to be the obvious way to go when studying
the mechanical behavior of braided wire stents.


.. _sec:operating-surf-mesh:

Operating on surface meshes
===========================

Besides being used for creating geometries, also offers interesting
possibilities for executing specialized operations on surface meshes, usually
STL type triangulated meshes originating from medical scan (CT) images. Some of
the algorithms developed were included in .


.. _sec:unroll-stent:

Unroll stent
------------

A stent is a medical device used to reopen narrowed arteries. The vast majority
of stents are balloon-expandable, which means that the metal structure is
deployed by inflating a balloon, located inside the stent. Figure
:ref:`fig:cypher-stent` shows an example of such a stent prior to expansion
(balloon not shown). The 3D surface is obtained by micro CT and consists of
triangles.

.. _`fig:cypher-stent`:

.. figure:: images/cypher-stent.*
   :align: center
   :alt: Cypher stent 

   Triangulated mesh of a Cypher\ |reg| stent

The structure of such a device can be quite complex and difficult to analyse.
The same functions offers for creating geometries can also be employed to
investigate triangulated meshes. A simple unroll operation of the stent gives a
much better overview of the complete geometrical structure and allows easier
analysis (see figure :ref:`fig:cypher-stent-unroll`).

``F = F.toCylindrical().scale([1.,2*radius*pi/360,1.])``

.. _`fig:cypher-stent-unroll`:

.. figure:: images/cypher-stent-unroll.*
   :align: center
   :alt: Cypher stent unrolled

   Result of the unroll operation

The unrolled geometry can then be used for further investigations. An important
property of such a stent is the circumference of a single stent cell. The
``clip()`` method can be used to isolate a single stent cell. In order to obtain
a line describing the stent cell, the function ``intersectionLinesWithPlane()``
has been used. The result can be seen in figures :ref:`fig:cypher-stent-cell-full`.


.. _`fig:cypher-stent-cell-full`:

.. figure:: images/stent-cell-full.*
   :align: center
   :alt: Intersection with plane

   Part of the intersection with a plane

Finally, one connected circumference of a stent cell is selected 
(figure :ref:`fig:cypher-stent-cell`) 
and the ``length()`` function returns its length, which
is 9.19 mm.

.. _`fig:cypher-stent-cell`:

.. figure:: images/stent-cell.*
   :align: center
   :alt: circumference of stent cell

   Circumference of a stent cell

.. End


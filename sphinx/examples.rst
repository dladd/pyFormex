.. $Id$
.. pyformex documentation --- examples

.. include:: defines.inc
.. include:: links.inc

.. _cha:examples:

************************
pyFormex example scripts
************************

.. warning:: This document is currently under revision! 

Sometimes you learn quicker from studying an example than from reading a 
tutorial or user guide. To help you we have created this collection of 
annotated examples. Beware that the script texts presented in this document
may differ slightly from the corresponding example coming with the |pyformex| 
distribution.

.. _sec:creating-geometry:

Creating geometry
=================

To get acquainted with the modus operandi of |pyformex|, 
the :file:`WireStent.py` script is studied step by step. The lines are numbered
for easy referencing, but are not part of the script itself.


.. literalinclude:: _static/scripts/WireStent.py
   :language: python
   :linenos:

As all |pyformex| scripts, it starts with a comments line holding the word
``pyformex``. 

To start, all required modules to run the :file:`WireStent.py`
script are imported (e.g. the ``math`` module to use the mathematical constant
:math:`\pi`). Subsequently, the class ``DoubleHelixStent`` is defined which
allows the simple use of the geometrical model in other scripts for e.g.
parametric, optimization and finite element analyses of braided wire stents.
Consequently, the latter scripts do not have to contain the wire stent geometry
building and can be condensed and conveniently arranged. The definition of the
class starts with a :math:`"""`\ documentation string\ :math:`"""`, explaining
its aim and functioning.


.. literalinclude:: _static/scripts/WireStent1.py
   :language: python
   :linenos:

The constructor ``__init__`` of the ``DoubleHelixStent`` class requires 8
arguments:

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


.. include:: _static/scripts/WireStent2.py
   :literal:

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
along the X-axis is summarized in the next script and depicted in Figure
:ref:`bumped`.


.. include:: _static/scripts/WireStent3.py
   :literal:

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
:ref:`base` This number can be used as an entry in a database, which holds some
sort of property. The Formex and the database are two seperate entities, only
linked by the property numbers. The ``rosette()``\ -function creates a unit cell
of crossing struts by :math:`2` rotational replications with an angular step of
[180]:math:`\deg` around the Z-axis (the original Formex is the first of the
:math:`2` replicas). If specified in the constructor, an additional Formex with
property :math:`2` connects the first points of the ``NE`` and ``SE`` Formices.

.. figure:: _static/images/WireStentDemot2Step01.*
   :align: center
   :alt: straight line segment

   A straight line segment

.. figure:: _static/images/WireStentDemot2Step02.*
   :align: center
   :alt: line segment with replications

   The line segment with replications

.. figure:: _static/images/WireStentDemot2Step03.*
   :align: center
   :alt: bumped line segment

   A bumped line segment



.. include:: _static/scripts/WireStent4.py
   :literal:

.. % \begin{figure} [ht]
.. % \centering
.. % \begin{makeimage}
.. % \end{makeimage}
.. % \begin{latexonly}
.. % \hspace{0.1cm}
.. % \begin{minipage} [c] [] [c]{3.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentDemot2Step04}
.. % \begin{center}
.. % \vspace{-3ex}
.. % (a)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \begin{minipage} [c] [] [c] {3.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentDemot2Step07}
.. % \begin{center}
.. % \vspace{-3ex}
.. % (b)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \begin{minipage} [c] [] [c] {3.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentDemot2Step09}
.. % \begin{center}
.. % \vspace{-3ex}
.. % (c)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \end{latexonly}
.. % \begin{htmlonly}
.. % \htmladdimg{../images/WireStentDemot2Step04.png}
.. % \htmladdimg{../images/WireStentDemot2Step07.png}
.. % \htmladdimg{../images/WireStentDemot2Step09.png}
.. % \end{htmlonly}
.. % \caption {Creation of unit cell of crossing and connected struts (c) from a rescaled (a) and mirrored, skewed (b) bumped strut.}
.. % \label{base}
.. % \end{figure}


Extending the base module
-------------------------

Subsequently, a mirrored copy of the base cell is generated. Both Formices are
translated to their appropriate side by side position with the ``translate()``\
-option and form the complete extended base module with 4 by 4 dimensions as
depicted in Figure :ref:`fig:base`. Furthermore, both Formices are defined as an
attribute of the ``DoubleHelixStent`` class by the ``self``\ -statement,
allowing their use after every ``DoubleHelixStent`` initialisation. Such further
use is impossible with local variables, such as for example the ``NE`` and
``SE`` Formices.

.. % 


.. include:: _static/scripts/WireStent5.py
   :literal:

.. % \begin{figure} [ht]
.. % \centering
.. % \begin{makeimage}
.. % \end{makeimage}
.. % \begin{latexonly}
.. % \hspace{0.1cm}
.. % \begin{minipage} [c] [] [c]{5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentDemot2Step10}
.. % \begin{center}
.. % \vspace{-3ex}
.. % (a)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \begin{minipage} [c] [] [c] {5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentDemot2Step11}
.. % \begin{center}
.. % \vspace{-3ex}
.. % (b)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \end{latexonly}
.. % \begin{htmlonly}
.. % \htmladdimg{../images/WireStentDemot2Step10.png}
.. % \htmladdimg{../images/WireStentDemot2Step11.png}
.. % \end{htmlonly}
.. % \caption {Creation of the complete extended base module (b) from the original and mirrored (a) unit cell.}
.. % \label{fig:base}
.. % \end{figure}


Full nearly planar pattern
--------------------------

The fully nearly planar pattern is obtained by copying the base module in two
directions and shown in Figure :ref:`plane`. ``replic2()`` generates this
pattern with :math:`nx` and :math:`ny` replications with steps :math:`dx` and
:math:`dy` in respectively, the default X- and Y-direction.

.. % 


.. include:: _static/scripts/WireStent6.py
   :literal:

.. % \begin{figure} [ht]
.. % \centering
.. % \begin{makeimage}
.. % \end{makeimage}
.. % \begin{latexonly}
.. % \hspace{0.1cm}
.. % \begin{minipage} [c] [] [c]{5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentDemot2Step12}
.. % \begin{center}
.. % \vspace{-3ex}
.. % %(a)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \begin{minipage} [c] [] [c] {5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentDemot2Step13}
.. % \begin{center}
.. % \vspace{-3ex}
.. % %(b)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \end{latexonly}
.. % \begin{htmlonly}
.. % \htmladdimg{../images/WireStentDemot2Step12.png}
.. % \htmladdimg{../images/WireStentDemot2Step13.png}
.. % \end{htmlonly}
.. % \caption {Creation of the fully nearly planar pattern.}
.. % \label{plane}
.. % \end{figure}


Cylindrical stent structure
---------------------------

Finally the full pattern is translated over the stent radius :math:`r` in
Z-direction and transformed to the cylindrical stent structure by a coordinate
transformation with the Z-coordinates as distance :math:`r`, the X-coordinates
as angle :math:`\theta` and the Y-coordinates as height :math:`z`. The
``scale()``\ -operator rescales the stent structure to the correct circumference
and length. The resulting stent geometry is depicted in Figure :ref:`stent`.

.. % 


.. include:: _static/scripts/WireStent7.py
   :literal:

In addition to the stent initialization, the ``DoubleHelixStent`` class script
contains a function ``all()`` representing the complete stent Formex.
Consequently, the ``DoubleHelixStent`` class has four attributes: the Formices
``cell1``, ``cell2`` and ``all``; and the number :math:`ny`.

.. % \begin{figure} [ht]
.. % \centering
.. % \begin{makeimage}
.. % \end{makeimage}
.. % \begin{latexonly}
.. % \hspace{0.1cm}
.. % \begin{minipage} [c] [] [c]{5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentDemot2Step16}
.. % \begin{center}
.. % \vspace{-3ex}
.. % (a)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \begin{minipage} [c] [] [c] {5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentDemot2Step15}
.. % \begin{center}
.. % \vspace{-3ex}
.. % (b)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \end{latexonly}
.. % \begin{htmlonly}
.. % \htmladdimg{../images/WireStentDemot2Step16.png}
.. % \htmladdimg{../images/WireStentDemot2Step15.png}
.. % \end{htmlonly}
.. % \caption {Creation of the cylindrical stent structure ((a) iso and (b) right view).}
.. % \label{stent}
.. % \end{figure}
.. % 


.. include:: _static/scripts/WireStent8.py
   :literal:

.. % 


Parametric stent geometry
-------------------------

An inherent feature of script-based modeling is the possibility of easily
generating lots of variations on the original geometry. This is a huge advantage
for parametric analyses and illustrated in Figure :ref:`param`: these wire
stents are all created with the same script, but with other values of the
parameters :math:`De`, :math:`nx` and :math:`\beta`. As the script for building
the wire stent geometry is defined as a the ``DoubleHelixStent`` class in the
(:file:`WireStent.py`) script, it can easily be imported for e.g. this purpose.

.. % 

.. % \begin{figure} [ht]
.. % \centering
.. % \begin{makeimage}
.. % \end{makeimage}
.. % \begin{latexonly}
.. % \hspace{0.1cm}
.. % \begin{minipage} [c] [] [c]{5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentD16L40d22n6b25}
.. % \begin{center}
.. % \vspace{-3ex}
.. % \code{DHS}(16,40,0.22,6,25)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \begin{minipage} [c] [] [c] {5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentD16L40d22n6b50}
.. % \begin{center}
.. % \vspace{-3ex}
.. % \code{DHS}(16,40,0.22,6,50)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \hspace{0.1cm}
.. % \begin{minipage} [c] [] [c]{5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentD16L40d22n10b25}
.. % \begin{center}
.. % \vspace{-3ex}
.. % \code{DHS}(16,40,0.22,10,25)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \begin{minipage} [c] [] [c] {5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentD16L40d22n10b50}
.. % \begin{center}
.. % \vspace{-3ex}
.. % \code{DHS}(16,40,0.22,10,50)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \hspace{0.1cm}
.. % \begin{minipage} [c] [] [c]{5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentD32L40d22n6b25}
.. % \begin{center}
.. % \vspace{-3ex}
.. % \code{DHS}(32,40,0.22,6,25)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \begin{minipage} [c] [] [c] {5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentD32L40d22n6b50}
.. % \begin{center}
.. % \vspace{-3ex}
.. % \code{DHS}(32,40,0.22,6,50)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \hspace{0.1cm}
.. % \begin{minipage} [c] [] [c]{5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentD32L40d22n10b25}
.. % \begin{center}
.. % \vspace{-3ex}
.. % \code{DHS}(32,40,0.22,10,25)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \begin{minipage} [c] [] [c] {5.5cm}
.. % \includegraphics [width =\textwidth] {images/WireStentD32L40d22n10b50}
.. % \begin{center}
.. % \vspace{-3ex}
.. % \code{DHS}(32,40,0.22,10,50)
.. % \vspace{1ex}
.. % \end{center}
.. % \end{minipage}
.. % \hspace{0.3cm}
.. % \end{latexonly}
.. % \begin{htmlonly}
.. % \htmladdimg{../images/WireStentD16L40d22n6b25.png}
.. % \htmladdimg{../images/WireStentD16L40d22n6b50.png}
.. % \htmladdimg{../images/WireStentD16L40d22n10b25.png}
.. % \htmladdimg{../images/WireStentD16L40d22n10b50.png}
.. % \htmladdimg{../images/WireStentD32L40d22n6b25.png}
.. % \htmladdimg{../images/WireStentD32L40d22n6b50.png}
.. % \htmladdimg{../images/WireStentD32L40d22n10b25.png}
.. % \htmladdimg{../images/WireStentD32L40d22n10b50.png}
.. % \end{htmlonly}
.. % \caption {Variations on the wire stent geometry using the DoubleHelixStent($De,L,d,nx,\beta$) (DHS() class.}
.. % \label{param}
.. % \end{figure}


.. include:: _static/scripts/WireStentParametricExample.py
   :literal:

Obviously, generating such parametric wire stent geometries with classical CAD
methodologies is feasible, though probably (very) time consuming. However, as
provides a multitude of features (such as parametric modeling, finite element
pre- and postprocessing, optimization strategies, etcetera) in one sinlge
consistent environment, it appearss to be the obvious way to go when studying
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

The structure of such a device can be quite complex and difficult to analyse.
The same functions offers for creating geometries can also be employed to
investigate triangulated meshes. A simple unroll operation of the stent gives a
much better overview of the complete geometrical structure and allows easier
analysis (see figure :ref:`fig:cypher-stent-unroll`).

.. % \begin{figure}[ht]
.. % \centering
.. % \begin{makeimage}
.. % \end{makeimage}
.. % \begin{latexonly}
.. % \includegraphics[width=8cm]{images/cypher-stent}
.. % \end{latexonly}
.. % \begin{htmlonly}
.. % \htmladdimg{../images/cypher-stent.png}
.. % \end{htmlonly}
.. % \caption{Triangulated mesh of a stent}
.. % \label{fig:cypher-stent}
.. % \end{figure}

``F = F.toCylindrical().scale([1.,2*radius*pi/360,1.])``

This unrolled geometry can then be used for further investigations. An important
property of such a stent is the circumference of a single stent cell. The
``clip()`` method can be used to isolate a single stent cell. In order to obtain
a line describing the stent cell, the function ``intersectionLinesWithPlane()``
has been used. The result can be seen in figure :ref:`fig:stent-cell`.

.. % \begin{figure}[ht]
.. % \centering
.. % \begin{makeimage}
.. % \end{makeimage}
.. % \begin{latexonly}
.. % \includegraphics[width=8cm]{images/cypher-stent-unroll}
.. % \end{latexonly}
.. % \begin{htmlonly}
.. % \htmladdimg{../images/cypher-stent-unroll.png}
.. % \end{htmlonly}
.. % \caption{Result of unroll operation}
.. % \label{fig:cypher-stent-unroll}
.. % \end{figure}

Finally, the ``length()`` function returns the circumference of the cell, which
is 9.19 mm.

.. % \begin{figure}[ht]
.. % \centering
.. % \begin{makeimage}
.. % \end{makeimage}
.. % \begin{latexonly}
.. % \includegraphics[width=6cm]{images/stent-cell-full}
.. % \includegraphics[width=6cm]{images/stent-cell}
.. % \end{latexonly}
.. % \begin{htmlonly}
.. % \htmladdimg{../images/stent-cell-full.png}
.. % \htmladdimg{../images/stent-cell.png}
.. % \end{htmlonly}
.. % \caption{Intersection of stent cell with plane and inner line of stent cell}
.. % \label{fig:stent-cell}
.. % \end{figure}

.. End


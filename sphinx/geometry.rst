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

.. _cha:geometry:

*******************************
Modeling Geometry with pyFormex
*******************************

.. warning:: This document still needs to be written! 

.. topic:: Abstract

  This chapter explains the different geometrical models in pyFormex,
  how and when to use them, how to convert between them, how to import
  and export them in various formats.

.. _sec:geom_intro:

Introduction
============

  *Everything is geometry*

In everyday life, geometry is ubiquitous. Just look around you: all the things you see, whether objects or living organisms or natural phenomena like clouds, they all have a shape or geometry. This holds for all concrete things, even if they are ungraspable, like a rainbow, or have no defined or fixed shape, like water. The latter evidently takes the shape of its container. Only abstract concepts do not have a geometry. Any material thing has though [#quantum]_, hence our claim: everything is geometry.

Since geometry is such an important aspect of everybody's life, one would expect that it would take an important place in education (base as well as higher). Yet we see that in the educational system of many developed countries, attention for geometry has vaned during the last decades. 
Important for craftsmen, technician, engineer, designer, artist

We will give some general ideas about geometry, but do not pretend to be a full
geometry course. Only concepts needed for or related to modleing with pyFormex. 

We could define the geometry of an object as the space it occupies. In our three-dimensional world, any object is also 3D. Some objects however have very small dimensions in one or more directions (e.g. a thin wire or a sheet of paper). It may be convenient then to model these only in one or two dimensions. [#4d]_


Concrete things also have a material. THIngs going wrong is mostly mechanical: geometry/materail

.. [#quantum] We obviously look here at matter in the way we observe it with our senses (visual, tactile) and not in a quantum-mechanics way.
.. [#4d] Mathematically we can also define geometry with higher dimensionality than 3, but this is of little practical use.


.. _sec:formex_model:

The **Formex** model
====================


.. _sec:mesh_model:

The **Mesh** model
==================


.. _sec:analytical_models:

Analytical models
=================


.. End


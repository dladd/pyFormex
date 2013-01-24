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

.. _cha:widgets:

*************
Using Widgets
*************

.. warning:: This document still needs to be written!

.. topic:: Abstract

  This chapter gives an overview of the specialized widgets in
  pyFormex and how to use them to quickly create a specialized
  graphical interface for you application.


The pyFormex Graphical User Interface (GUI) is built on the QT4 toolkit,
accessed from Python by PyQt4. Since the user has full access to all
underlying libraries, he can use any part from QT4 to construct the
most sophisticated user interface and change the pyFormex GUI like
he wants and sees fit. However, properly programming a user interface is
a difficult and tedious task, and many normal users do not have the knowledge
or time to do this. pyFormex provides a simplified framework to access the
QT4 tools in a way that complex and sophisticated user dialogs can be built
with a minimum effort. User dialogs are create automatically from a very
limited input. Specialized input widgets are included dependent on the
type of input asked from the user. And when this simplified framework falls
short for your needs, you can always access the QT4 functions directly to
add what you want.




.. _sec:input_askitems:

The askItems functions
======================

The :func:`askItems` function reduces the effort needed to create an interactive dialog asking input data from the user.


.. _sec:input_dialog:

The input dialog
================



.. _sec:user_menu:

The user menu
=============


.. _sec:other_widgets:

Other widgets
=============

.. End


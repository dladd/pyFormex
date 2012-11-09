..
  
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
  
  

=========
gtsinside
=========

-----------------------------------------------
Test whether points are inside a closed surface
-----------------------------------------------

:Author: Benedict Verhegghe <benedict.verhegghe@ugent.be>. This manual page was written for the Debian project (and may be used by others).
:Date:   2012-08-08
:Copyright: GPL v3 or higher
:Version: 0.1
:Manual section: 1
:Manual group: text and X11 processing

SYNOPSIS
========

gtsinside [OPTION ...] SURFACE POINTS

DESCRIPTION
===========

Test whether points are inside a closed surface.

SURFACE is the a file representing a closed manifold surface in GTS format.
POINTS is a text file where each line contains the three coordinates of a point, separated with blanks. This program uses the GTS library to test which of the points are inside the surface. It outputs a list of integer numbers to stdout.
The numbers represent the line numbers in POINTS corresponding to the points inside the surface.

Due to rounding errors in the floating point computations, this command may report spurious false positives or negatives. It is mainly primarily intended for being used through pyFormex. pyFormex will apply the test three times in different coordinate directions, and report the results that have occurred at least twice, making it far more robust. 

OPTIONS
=======

-v, --verbose        Print statistics about the surface.
-h, --help           Display this help and exit.


SEE ALSO
========

gtscheck


AUTHOR
======

gtsinside was written by Benedict Verhegghe <benedict.verhegghe@ugent.be>.
The GTS library was written by Stephane Popinet <popinet@users.sourceforge.net>.

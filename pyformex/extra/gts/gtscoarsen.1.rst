..
  
..
  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
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
  
  

==========
gtscoarsen
==========

---------------------
Coarsen a GTS Surface
---------------------

:Author: Benedict Verhegghe <benedict.verhegghe@ugent.be>. This manual page was written for the Debian project (and may be used by others).
:Date:   2012-08-08
:Copyright: GPL v3 or higher
:Version: 0.1
:Manual section: 1
:Manual group: text and X11 processing

SYNOPSIS
========

gtscoarsen [OPTION] < file.gts

DESCRIPTION
===========

Construct a coarsened version of the input GTS surface.

OPTIONS
=======

-n N, --number=N     Stop the coarsening process if the number of edges 
                     was to fall below N.
-c C, --cost=C       Stop the coarsening process if the cost of collapsing
                     an edge is larger than C.
-m, --midvertex      Use midvertex as replacement vertex instead of default
                     volume optimized point.
-l, --length         Use length^2 as cost function instead of default
                     optimized point cost.
-f F, --fold=F       Set maximum fold angle to F degrees (default 1 degree).
-w W, --vweight=W    Set weight used for volume optimization (default 0.5).
-b W, --bweight=W    Set weight used for boundary optimization (default 0.5).
-s W, --sweight=W    Set weight used for shape optimization (default 0.0).
-p, --progressive    Write progressive surface file.
-L, --log            Log the evolution of the cost.
-f VAL, --fold=VAL   Smooth only folds.
-v, --verbose        Print statistics about the surface.
-h, --help           Display this help and exit.


SEE ALSO
========

gtscheck


AUTHOR
======

The GTS library was written by Stephane Popinet <popinet@users.sourceforge.net>.
The gtscoarsen command is taken from the examples.

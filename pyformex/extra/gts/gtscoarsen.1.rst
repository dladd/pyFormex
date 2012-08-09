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

=========
gtsrefine
=========

--------------------
Refine a GTS Surface
--------------------

:Author: Benedict Verhegghe <benedict.verhegghe@ugent.be>. This manual page was written for the Debian project (and may be used by others).
:Date:   2012-08-08
:Copyright: GPL v3 or higher
:Version: 0.1
:Manual section: 1
:Manual group: text and X11 processing

SYNOPSIS
========

gtsrefine [OPTION] < file.gts

DESCRIPTION
===========

Construct a refined version of the input GTS surface.

OPTIONS
=======

-n N, --number=N     Stop the refining process if the number of edges
                     was to become greater than N.
-c C, --cost=C       Stop the refining process if the cost of refining
                     an edge is smaller than C.
-L, --log            Log the evolution of the cost.
-v, --verbose        Print statistics about the surface.
-h, --help           Display this help and exit.


SEE ALSO
========

gtscheck


AUTHOR
======

The GTS library was written by Stephane Popinet <popinet@users.sourceforge.net>.
The gtsrefine command is taken from the examples.

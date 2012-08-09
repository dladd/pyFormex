=========
gtssmooth
=========

--------------------
Smooth a GTS Surface
--------------------

:Author: Benedict Verhegghe <benedict.verhegghe@ugent.be>. This manual page was written for the Debian project (and may be used by others).
:Date:   2012-08-08
:Copyright: GPL v3 or higher
:Version: 0.1
:Manual section: 1
:Manual group: text and X11 processing

SYNOPSIS
========

gtssmooth [OPTION] LAMBDA NITER < file.gts > smooth.gts

DESCRIPTION
===========

Smooth a GTS file by applying NITER iterations of a Laplacian filter
of parameter LAMBDA.

OPTIONS
=======

--fold=VAL, -f VAL   Smooth only folds
--verbose, -v        Print statistics about the surface.
--help, -h           Display this help and exit.


SEE ALSO
========

gtscheck


AUTHOR
======

The GTS library was written by Stephane Popinet <popinet@users.sourceforge.net>.
The gtssmooth command is taken from the examples.

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

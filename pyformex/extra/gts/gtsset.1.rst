======
gtsset
======

---------------------------------------
Compute set operations between surfaces
---------------------------------------

:Author: Benedict Verhegghe <benedict.verhegghe@ugent.be>. This manual page was written for the Debian project (and may be used by others).
:Date:   2012-08-08
:Copyright: GPL v3 or higher
:Version: 0.1
:Manual section: 1
:Manual group: text and X11 processing

.. TODO: authors and author with name <email>

SYNOPSIS
========

  gtsset [OPTION] OPERATION FILE1 FILE2

DESCRIPTION
===========

Compute set operations between surfaces, where OPERATION is either.
union, inter, diff, all.

FILE1 and FILE2 are files with closed surface representations in GTS format.

The default operation is to write a new closed surface in GTS format to stdout.

OPTIONS
=======

--inter, -i      Output an OOGL (Geomview) representation of the curve
                 intersection of the surfaces
--self, -s       Check that the surfaces are not self-intersecting. 
                 If one of them is, the set of self-intersecting faces
                 is written (as a GtsSurface) on standard output.
--verbose, -v    Do not print statistics about the surface.
--help, -h       Display this help and exit.



SEE ALSO
========

gtscheck


AUTHOR
======

The gts library was written by Stephane Popinet <popinet@users.sourceforge.net>.
This gtsset command is taken from the examples.

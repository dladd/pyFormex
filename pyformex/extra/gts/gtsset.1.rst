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

SYNOPSIS
========

gtsset [OPTION] OPERATION surface1.gts surface2.gts

DESCRIPTION
===========

Compute boolean operations between surfaces. OPERATION is one of:
union, inter, diff, all.

surface1.gts and surface2.gts are files with closed surface representations
in GTS format.

The default operation is to write to stdout a new closed surface in GTS format.
For the 'all' operation however, four GTS surface files are written: 
s1out2.gts, s1in2.gts, s2out1.gts, s2in1.gts.

OPTIONS
=======

-i, --inter          Instead of outputting a surface, output an OOGL (Geomview)
    		     representation of the curve intersection of the surfaces.
-s, --self           Check that the surfaces are not self-intersecting. 
                     If one of them is, the set of self-intersecting faces
                     is written (as a GtsSurface) on standard output.
-v, --verbose        Print statistics about the surface.
-h, --help           Display this help and exit.


SEE ALSO
========

gtscheck


AUTHOR
======

The gts library was written by Stephane Popinet <popinet@users.sourceforge.net>.
This gtsset command is taken from the examples.

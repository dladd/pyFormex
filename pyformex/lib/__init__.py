#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""pyFormex C library module initialisation.

This tries to load the compiled libraries, and replaces those that failed
to load with the (slower) Python versions.
"""

__all__ = [ 'drawgl', 'misc', 'accelerated' ]

drawgl = misc = None
accelerated = []


def init_libs():
    global drawgl,misc
    
    import pyformex as GD

    # testing for not False makes other values than T/F (like None) pass
    if GD.options.uselib is not False:

        try:
            import misc
            accelerated .append(misc)
            GD.debug("Succesfully loaded the pyFormex compiled misc library")
            has_misc = True
        except ImportError:
            GD.debug("Error while loading the pyFormex compiled misc library")

        if GD.options.gui: 
            try:
                import drawgl
                accelerated .append(drawgl)
                GD.debug("Succesfully loaded the pyFormex compiled draw library")
                has_drawgl = True
            except ImportError:
                GD.debug("Error while loading the pyFormex compiled draw library")

    if misc is None:
        GD.debug("Using the (slower) Python misc functions")
        import pyformex.misc as misc

    if drawgl is None:
        GD.debug("Using the (slower) Python draw functions")
        import pyformex.gui.drawgl as drawgl


init_libs()

# End

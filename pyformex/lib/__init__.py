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

__all__ = [ 'drawgl', 'misc', 'has_drawgl', 'has_misc' ]

import pyformex as GD

if GD.options.uselib is None:
    GD.options.uselib = True

has_misc = has_drawgl = GD.options.uselib

if has_drawgl:
    try:
        import pyformex.lib.drawgl
        GD.debug("Succesfully loaded the pyFormex compiled draw library")
    except ImportError:
        GD.debug("Error while loading the pyFormex compiled draw library")
        GD.debug("Reverting to scripted versions")
        has_drawgl = False

if has_misc:
    try:
        import pyformex.lib.drawgl
        GD.debug("Succesfully loaded the pyFormex compiled misc library")
    except ImportError:
        GD.debug("Error while loading the pyFormex compiled misc library")
        GD.debug("Reverting to scripted versions")
        has_misc = False

        
if not has_drawgl:
    GD.debug("Using the (slower) Python draw functions")
    import pyformex.gui.drawgl

if not has_misc:
    GD.debug("Using the (slower) Python misc functions")
    import pyformex.misc

# End

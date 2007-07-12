#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.5 Release Tue Jul 10 13:43:12 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""Global data for pyFormex."""

# Set pyformex version
__version__ = "0.5-alpha"
Version = 'pyFormex version %s' % __version__
Copyright = 'Copyright (C) 2007 Benedict Verhegghe'
options = None
print_help = None

cfg = None
refcfg = None
preffile = None

app = None
gui = None
canvas = None
board = None
PF = {}  # globals that will be offered to scripts
image_formats_qt = []
image_formats_qtr = []
image_formats_gl2ps = []
image_formats_fromeps = []
multisave = False
canPlay = False
scriptName = None

external = {} # dict with detected external commands
calpy_version = None   # do we have calpy?


def message(s):
    print s

def debug(s):
    if hasattr(options,'debug'):
        if options.debug:
            print "DEBUG: %s" % str(s)



def savePreferences():
    """Save the preferences.

    The name of the preferences file was set in GD.preffile.
    If a local preferences file was read, it will be saved there.
    Otherwise, it will be saved as the user preferences, possibly
    creating that file.
    """
    f = preffile
    del cfg['__ref__']
    debug("!!!Saving config:\n%s" % cfg)

    try:
        fil = file(f,'w')
        fil.write("%s" % cfg)
        fil.close()
        res = "Saved"
    except:
        res = "Could not save"
    debug("%s preferences to file %s" % (res,f))

# End

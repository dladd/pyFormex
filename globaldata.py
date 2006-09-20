#!/usr/bin/env python
# $Id$
"""Global data for pyFormex."""
Version = "pyFormex 0.4-alpha"


cfg = None
refcfg = None
preffile = None

gui = None
canvas = None
board = None
help = None
PyFormex = {}  # globals that will be offered to scripts
image_formats_qt = []
image_formats_gl2ps = []
multisave = False
canPlay = False
scriptName = None

def debug(s):
    if options.debug:
        print s



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

#!/usr/bin/env python
# $Id$
"""Global data for pyFormex."""
Version = "pyFormex 0.3-alpha"

import config

cfg = config.Config()
gui = None
canvas = None
help = None
PyFormex = {}  # globals that will be offered to scripts
image_formats = []
prefsChanged = False
multisave = False
canPlay = False

def debug(s):
    if options.debug:
        print s

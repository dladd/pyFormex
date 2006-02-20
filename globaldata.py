#!/usr/bin/env python
# $Id$
"""Global data for pyFormex."""
Version = "pyFormex 0.3-alpha"

import myconfig

cfg = myconfig.Config()
gui = None
canvas = None
PyFormex = {}  # globals that will be offered to scripts
scriptName = None
image_formats = []

def debug(s):
    if options.debug:
        print s

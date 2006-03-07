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
scriptName = None
image_formats = []

def debug(s):
    if options.debug:
        print s

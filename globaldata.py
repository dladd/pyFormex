#!/usr/bin/env python
# $Id$
"""Global data for pyFormex."""
Version = "pyFormex 0.3-alpha"
##import __main__

##class SuperGlobal:

##    def __getattr__(self, name):
##        return __main__.__dict__.get(name, None)
        
##    def __setattr__(self, name, value):
##        __main__.__dict__[name] = value
        
##    def __delattr__(self, name):
##        if __main__.__dict__.has_key(name):
##            del  __main__.__dict__[name]

config = {}
gui = None
canvas = None
PyFormex = {}  # globals that will be offered to scripts

def debug(s):
    if options.debug:
        print s

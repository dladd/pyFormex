# Definition of some RGB colors
##
## This file is part of pyformex 0.1.2 Release Fri Jul  9 14:48:57 2004
## pyformex is a python implementation of Formex algebra
## (c) 2004 Benedict Verhegghe (email: benedict.verhegghe@ugent.be)
## Releases can be found at ftp://mecatrix.ugent.be/pub/pyformex/
## Distributed under the General Public License, see file COPYING for details
##

black   = (0.0,0.0,0.0)
red     = (1.0,0.0,0.0)
green   = (0.0,1.0,0.0)
blue    = (0.0,0.0,1.0)
cyan    = (0.0,1.0,1.0)
magenta = (1.0,0.0,1.0)
yellow  = (1.0,1.0,0.0)
white   = (1.0,1.0,1.0)

def grey(i):
    return (i,i,i)

lightgrey = grey(0.8)
mediumgrey = grey(0.7)
darkgrey = grey(0.5)

def RGBA(rgb,alpha=0.0):
    """Adds an alpha channel to an RGB color"""
    return rgb+(alpha,)

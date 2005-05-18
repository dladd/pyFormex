# Definition of some RGB colors
# $Id$
##
## This file is part of pyFormex 0.2.1 Release Fri Apr  8 23:30:39 2005
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where otherwise stated 
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

#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Mon Feb 26 08:57:40 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""Definition of some RGB colors and color convedrsion functions"""

from PyQt4 import QtGui

def GLColor(color):
    """Convert a color to an OpenGL RGB color.

    The output is a tuple of three RGB float values ranging from 0.0 to 1.0.
    The input can be any of the following:
    - a string specifying the Xwindow name of the color
    - a QColor
    - a tuple or list of 3 int values 0..255
    - a tuple or list of 3 float values 0.0..1.0
    Any other input may give unpredictable results.
    """
    if type(color) == str:
        color = QtGui.QColor(color)
    if isinstance(color,QtGui.QColor):
        color = (color.red(),color.green(),color.blue())
    try:
        #print type(color),list(color)
        #print "OK"
        color = list(color)
        #print color
        #print len(color)
        if len(color) == 3:
            if type(color[0]) == int:
                color = [ c/255. for c in color ]
            #print "color is now",color
            if type(color[0]) == float:
                return tuple(color)
            else:
                #print "type:",type(color[0])
                pass
        else:
            #print "len is not 3"
            pass
        raise
    except:
        raise RuntimeError,"GLColor: unexpected input type %s: %s" % (type(color),color)


def RGBA(rgb,alpha=1.0):
    """Adds an alpha channel to an RGB color"""
    return GLColor(rgb)+(alpha,)


black   = (0.0, 0.0, 0.0)
red     = (1.0, 0.0, 0.0)
green   = (0.0, 1.0, 0.0)
blue    = (0.0, 0.0, 1.0)
cyan    = (0.0, 1.0, 1.0)
magenta = (1.0, 0.0, 1.0)
yellow  = (1.0, 1.0, 0.0)
white   = (1.0, 1.0, 1.0)


def GREY(val,alpha=1.0):
    """Returns a grey OpenGL color of given intensity (0..1)"""
    return (val,val,val,1.0)

def grey(i):
    return (i,i,i)

lightgrey = grey(0.8)
mediumgrey = grey(0.7)
darkgrey = grey(0.5)

if __name__ == "__main__":
    print GLColor('red')
    print GLColor(red)
    print GLColor([200,200,255])
    print GLColor([1.,1.,1.])
    print GLColor(lightgrey)
    

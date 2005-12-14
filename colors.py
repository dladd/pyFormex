#!/usr/bin/env python
# Definition of some RGB colors
# $Id$

from qt import QColor

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
        color = QColor(color)
    if isinstance(color,QColor):
        color = (color.red(),color.green(),color.blue())
    if len(color) == 3:
        if type(color[0]) == int:
            color = [ c/255. for c in color ]
        if type(color[0]) == float:
            return tuple(color)
    raise RuntimeError,"GLColor: unexpected input %s" % color


def RGBA(rgb,alpha=1.0):
    """Adds an alpha channel to an RGB color"""
    return rgb+(alpha,)


black   = (0.0, 0.0, 0.0)
red     = (1.0, 0.0, 0.0)
green   = (0.0, 1.0, 0.0)
blue    = (0.0, 0.0, 1.0)
cyan    = (0.0, 1.0, 1.0)
magenta = (1.0, 0.0, 1.0)
yellow  = (1.0, 1.0, 0.0)
white   = (1.0, 1.0, 1.0)

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
    

# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""Definition of some RGB colors and color conversion functions

"""
from __future__ import print_function

from PyQt4 import QtCore,QtGui
import numpy

# BV: THIS DETECTION SHOULD GO ELSEWHERE
import pyformex as pf
try:
    QtGui.QColor.setAllowX11ColorNames(True)
    pf.X11 = True
except:
    print("WARNING: THIS IS NOT AN X11 WINDOW SYSTEM!")
    print("SOME THINGS MAY NOT WORK PROPERLY!")
    pf.X11 = False

def GLColor(color):
    """Convert a color to an OpenGL RGB color.

    The output is a tuple of three RGB float values ranging from 0.0 to 1.0.
    The input can be any of the following:
    - a QColor
    - a string specifying the Xwindow name of the color
    - a hex string '#RGB' with 1 to 4 hexadecimal digits per color 
    - a tuple or list of 3 integer values in the range 0..255
    - a tuple or list of 3 float values in the range 0.0..1.0
    Any other input may give unpredictable results.
    """
    col = color

    # as of Qt4.5, QtGui.Qcolor no longer raises an error if given
    # erroneous input. Therefore, we check it ourselves

    # str, QString or QtCore.Globalcolor: convert to QColor
    if ( type(col) is str or
         type(col) is QtCore.QString  or
         isinstance(col,QtCore.Qt.GlobalColor) ):
        try:
            col = QtGui.QColor(col)
        except:
            pass

    # QColor: convert to (r,g,b) tuple (0..255)
    if isinstance(col,QtGui.QColor):
        col = (col.red(),col.green(),col.blue())
        
    # Convert to a list and check length
    try:
        col = tuple(col)
        if len(col) == 3:
            if type(col[0]) == int:
                # convert int values to float
                col = [ c/255. for c in col ]
            col = map(float,col)
            # SUCCESS !
            return tuple(col)
    except:
        pass

    # No success: rais an error
    raise ValueError,"GLColor: unexpected input of type %s: %s" % (type(color),color)


def colorName(color):
    """Return a string designation for the color.

    color can be anything that is accepted by GLColor.
    In most cases
    If color can not be converted, None is returned.
    """
    try:
        return str(QtGui.QColor.fromRgbF(*(GLColor(color))).name())
    except:
        return None

def RGBcolor(color):
    """Return an RGB (0-255) tuple for an OpenGL color"""
    col = array(color)*255
    return col.round().astype(Int)

def WEBcolor(color):
    """Return an RGB hex string for an OpenGL color"""
    col = RGBcolor(color)
    return "#%02x%02x%02x" % tuple(col)


def createColorDict():
    for c in QtGui.QColor.colorNames():
        col = QtGui.QColor
        print("Color %s = %s" % (c,colorName(c)))


def closestColorName(color):
    """Return the closest color name."""
    pass


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
pyformex_pink = (1.0,0.2,0.4)


def GREY(val,alpha=1.0):
    """Returns a grey OpenGL color of given intensity (0..1)"""
    return (val,val,val,1.0)

def grey(i):
    return (i,i,i)

lightlightgrey = grey(0.9)
lightgrey = grey(0.8)
mediumgrey = grey(0.7)
darkgrey = grey(0.5)

if __name__ == "__main__":
    print(GLColor(QtGui.QColor('red')))
    print(GLColor(QtGui.QColor('indianred')))
    print(GLColor('red'))
    print(GLColor(red))
    print(GLColor([200,200,255]))
    print(GLColor([1.,1.,1.]))
    print(GLColor(lightgrey))
    print(GLColor('#ffddff'))
    
    print(colorName(red))
    print(colorName('red'))
    print(colorName('#ffddff'))
    print(colorName([1.,1.,1.]))
        
    createColorDict()

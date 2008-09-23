# $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Definition of some RGB colors and color convedrsion functions"""

from PyQt4 import QtGui

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

    # include this test because QtGui.Qcolor spits output if given
    # an erroneous string.
    if not ( type(col) == tuple or type(col) == list ):
        # Convert strings or QColors to a color tuple
        try:
            col = QtGui.QColor(col)
            col = (col.red(),col.green(),col.blue())
        except:
            pass

    # col should now be a 3-tuple
    if (type(col) == tuple or type(col) == list) and len(col) == 3:
        try:
            if type(col[0]) == int:
                # convert int values to float
                col = [ c/255. for c in col ]
            col = map(float,col)
            # SUCCESS !
            return tuple(col)
        except:
            pass

    raise ValueError,"GLColor: unexpected input type %s: %s" % (type(color),color)

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

def createColorDict():
    for c in QtGui.QColor.colorNames():
        col = QtGui.QColor
        print "Color %s = %s" % (c,colorName(c))


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


def GREY(val,alpha=1.0):
    """Returns a grey OpenGL color of given intensity (0..1)"""
    return (val,val,val,1.0)

def grey(i):
    return (i,i,i)

lightgrey = grey(0.8)
mediumgrey = grey(0.7)
darkgrey = grey(0.5)

if __name__ == "__main__":
    print GLColor(QtGui.QColor('red'))
    print GLColor('red')
    print GLColor(red)
    print GLColor([200,200,255])
    print GLColor([1.,1.,1.])
    print GLColor(lightgrey)
    print GLColor('#ffddff')
    
    print colorName(red)
    print colorName('red')
    print colorName('#ffddff')
    print colorName([1.,1.,1.])
        
    createColorDict()

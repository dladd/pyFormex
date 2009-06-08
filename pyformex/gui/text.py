# $Id$
##
##  This file is part of pyFormex 0.8 Release Mon Jun  8 11:56:55 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

"""Functions related to text and fonts"""

import pyformex
from PyQt4 import QtGui


def getFont(font=None,size=None):
    """Get the best fonts matching font name and size

    If nothing is specified, returns the default GUI font.
    """
    if font is None:
        font = pyformex.GUI.font()
    else:
        font = QtGui.QFont(font)
    if size is not None:
        font.setPointSize(size)
    return font


def fontHeight(font=None,size=None):
    """Return the height in pixels of the given font.

    This can be used to determine the canvas coordinates where the text is
    to be drawn.
    """
    font = getFont(font,size)
    fh = font.pixelSize()
    if fh < 0:
        fh = QtGui.QFontInfo(font).pixelSize()
    return fh

# End

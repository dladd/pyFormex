# $Id$

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

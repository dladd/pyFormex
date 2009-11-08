#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
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
"""Using bitmap images as colors.

This module contains functions to use bitmap images as colors on a
pyFormex geometry.
"""

from pyformex.arraytools import *
from PyQt4.QtGui import QImage
from imagearray import qimage2numpy

def image2glcolor(im,flip=True):
    """Convert a bitmap image to corresponding OpenGL colors.

    im is a QImage or any data from which a QImage can be initialized.
    The image RGB colors are converted to OpenGL colors.
    The return value is a (w,h,3) shaped array of values in the range
    0.0 to 1.0.
    By default the image is flipped upside-down because the vertical
    OpenGL axis points upwards, while bitmap images are stored downwards.
    """
    im = QImage(im)
    c,t = qimage2numpy(im)
    if flip:
        c = flipud(c)
    if t is None:
        color = dstack([c['r'],c['g'],c['b']]).reshape(-1,3)
        # print color.shape
        return color.astype(Float)/255.,t
    else:
        return c,t

# End

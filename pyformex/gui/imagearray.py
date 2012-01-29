# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Convert bitmap images into numpy arrays.

This module contains functions to convert bitmap images into numpy
arrays and vice versa.

This code was based on ideas found on the PyQwt mailing list.
"""

from PyQt4.QtGui import QImage, QColor
import numpy

def image2numpy(image,order='RGBA',expand=False,flip=True):
    """Transform an image to a Numpy array.

    Parameters:

    - image: a QImage or any data that can be converted to a QImage,
      e.g. the name of an image file.
    - order: string with a permutation of the characters 'RGBA', defining
      the order in which the colors are returned. Default is RGBA, so that
      result[...,0] gives the red component. Note however that QImage stores
      in ARGB order. 
    - expand: boolean: if True, an indexed image format will be converted
      to a full color array.
    - flip: boolean: if True, the image scanlines are flipped upside down.
      This is practical because image files are usually stored in top down
      order, while OpenGL uses an upwards positive direction, requiring a
      flip to show the image upright.

    Returns:

    - if expand is False: a tuple (colors,colortable) where the meaning of
      colors and colortable depends on the format of the QImage:

      - for Indexed8 format, colors is an array of integer indices into
        the colortable, which is a single list if colors (i.e. a
        float array with shape (ncolors,3 or 4).

        !!!Beware! This currently does not work! All images are returned in
        full color mode!

      - for RGB32, ARGB32 and ARGB32_Premultiplied formats, colors returns
        the real colors and colortable is None

    - if expand is True, always returns a single array with the full colors
    
    """
    if not isinstance(image,QImage):
        image = QImage(image)
        
    if image.format() == QImage.Format_Indexed8:
        # TODO: make the colortable method work
        # The colortable method does not work yet:
        # work around is to convert to full color
        image = image.convertToFormat(QImage.Format_ARGB32)
        
    if image.format() in (QImage.Format_ARGB32_Premultiplied,
                           QImage.Format_ARGB32,
                           QImage.Format_RGB32):
        h,w = image.height(),image.width()
        buf = image.bits().asstring(image.numBytes())
        ar = numpy.frombuffer(buf,dtype='ubyte',count=image.numBytes()).reshape(h,w,4)
        idx = [ 'BGRA'.index(c) for c in order ]
        ar = ar[...,idx]
        if flip:
            ar = numpy.flipud(ar)
        colortable = None

    elif image.format() == QImage.Format_Indexed8:
        ncolors = image.numColors()
        print("Number of colors: %s" % ncolors)
        colortable = image.colorTable()
        print(colortable)
        colortable = numpy.array(colortable)
        print(colortable.dtype)
        print(colortable.size)
        dtype = numpy.uint8
        buf = image.bits().asstring(image.numBytes())
        ar = numpy.frombuffer(buf, dtype)
        h,w = image.height(),image.width()
        if w*h != ar.size:
            print("!! Size of image (%s) does not match dimensions: %s x %s = %s" % (ar.size,w,h,w*h))
        ar = ar[:w*h]
        print(ar.shape)
        print(ar.dtype)
        print(ar.shape)
        print(ar)
    
        return ar.reshape(h,w),colortable
        
    else:
        raise ValueError("image2numpy only supports 32bit and 8bit images")

    if expand and colortable:
        ar = colortable[ar]
    if expand:
        return ar
    else:
        return ar,colortable
    

def numpy2qimage(array):
        if numpy.ndim(array) == 2:
                return gray2qimage(array)
        elif numpy.ndim(array) == 3:
                return rgb2qimage(array)
        raise ValueError("can only convert 2D or 3D arrays")


def gray2qimage(gray):
        """Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
        colormap.  The first dimension represents the vertical image axis."""
        if len(gray.shape) != 2:
                raise ValueError("gray2QImage can only convert 2D arrays")

        gray = numpy.require(gray, numpy.uint8, 'C')

        h, w = gray.shape

        result = QImage(gray.data, w, h, QImage.Format_Indexed8)
        result.ndarray = gray
        for i in range(256):
                result.setColor(i, QColor(i, i, i).rgb())
        return result


def rgb2qimage(rgb):
        """Convert the 3D numpy array `rgb` into a 32-bit QImage.  `rgb` must
        have three dimensions with the vertical, horizontal and RGB image axes."""
        if len(rgb.shape) != 3:
                raise ValueError("rgb2QImage can expects the first (or last) dimension to contain exactly three (R,G,B) channels")
        if rgb.shape[2] != 3:
                raise ValueError("rgb2QImage can only convert 3D arrays")

        h, w, channels = rgb.shape

        # Qt expects 32bit BGRA data for color images:
        bgra = numpy.empty((h, w, 4), numpy.uint8, 'C')
        bgra[...,0] = rgb[...,2]
        bgra[...,1] = rgb[...,1]
        bgra[...,2] = rgb[...,0]
        bgra[...,3].fill(255)

        result = QImage(bgra.data, w, h, QImage.Format_RGB32)
        result.ndarray = bgra
        return result



def image2glcolor(im):
    """Convert a bitmap image to corresponding OpenGL colors.

    im is a QImage or any data from which a QImage can be initialized.
    The image RGB colors are converted to OpenGL colors.
    The return value is a (w,h,3) shaped array of values in the range
    0.0 to 1.0.
    By default the image is flipped upside-down because the vertical
    OpenGL axis points upwards, while bitmap images are stored downwards.
    """
    c = image2numpy(im,order='RGB',flip=True,expand=True)
    c = c.reshape(-1,3)
    c = c / 255.
    return c, None


# Image to data using PIL

def imagefile2string(filename):
    import Image
    im = Image.open(filename)
    nx,ny = im.size[0],im.size[1]
    try:
        data = im.tostring("raw","RGBA",0,-1)
    except SystemError:
        data = im.tostring("raw","RGBX",0,-1)
    return nx,ny,data

# End

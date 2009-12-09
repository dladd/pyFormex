# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Tue Dec  8 12:25:08 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

"""2D text decorations using GLUT fonts"""

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

#from drawable import *


# Needed to initialize the fonts
GLUT.glutInit([])

import colors

### Some drawing functions ###############################################

# These are the available GLUT fonts.
GLUTFONTS = {
    '9x15' : GLUT.GLUT_BITMAP_9_BY_15,
    '8x13' : GLUT.GLUT_BITMAP_8_BY_13,
    'tr10' : GLUT.GLUT_BITMAP_TIMES_ROMAN_10,
    'tr24' : GLUT.GLUT_BITMAP_TIMES_ROMAN_24,
    'hv10' : GLUT.GLUT_BITMAP_HELVETICA_10,
    'hv12' : GLUT.GLUT_BITMAP_HELVETICA_12,
    'hv18' : GLUT.GLUT_BITMAP_HELVETICA_18,
}

GLUTFONTALIAS = {
    'fixed' : ('9x15','8x13'),
    'serif' : ('times','tr10','tr24'),
    'sans'  : ('helvetica','hv10','hv12','hv18'),
    }

def glutSelectFont(font=None,size=None):
    """Select one of the glut fonts using a font + size description.

    font is one of: 'fixed', 'serif', 'sans'
    size is an int that will be rounded to the nearest available size.

    The return value is a 4-character string representing one of the
    GLUT fonts.
    """
    #GD.debug("INPUT %s,%s" % (font,size))
    if size is None and font in GLUTFONTS:
        return font
    if font is None:
        font = 'sans'
    for k in GLUTFONTALIAS:
        if font in GLUTFONTALIAS[k]:
            font = k
            break
    if size is None or not size > 0:
        size = 14
    if font == 'fixed':
        selector = [ (0,'8x13'), (14,'9x15') ]
    elif font == 'serif':
        selector = [ (0,'tr10'), (16,'tr24') ]
    else:
        selector = [ (0,'hv10'), (12,'hv12'), (16,'hv18') ]
    sel = selector[0]
    for s in selector[1:]:
        if s[0] <= size:
            sel = s

    return sel[1]


def getFont(font,size):
    return glutFont(glutSelectFont(font,size))


def glutFont(font):
    """Return GLUT font designation for the named font.

    The recognized font names are:
    
    - fixed: '9x15', '8x13',
    - times-roman: 'tr10', 'tr24'
    - helvetica:   'hv10', 'hv12',  'hv18'
    
    If an unrecognized string is  given, the default is 'hv18'.
    """
    return GLUTFONTS.get(font,GLUTFONTS['hv18'])



def glutFontHeight(font):
    """Return the height of the named glut font.

    This supposes that the last two characters of the name
    hold the font height.
    """
    return int(font[-2:])


def glutRenderText(text,font):
    """Draw a text in given font at the current rasterpoint.

    font should be one  of the legal fonts returned by glutFont().
    If text is not a string, it will be formatted to a string
    before drawing.
    After drawing, the rasterpos will have been updated!
    """
    if type(font) == str:
        font = glutFont(font)
    for character in str(text):
        GLUT.glutBitmapCharacter(font, ord(character))


def glutBitmapLength(font, text):
    """ Compute the length in pixels of a text string in given font.

    We use our own function to calculate the length because the builtin
    has a bug.
    """
    if type(font) == str:
        font = glutFont(font)
    len = 0
    for c in text:
        len += GLUT.glutBitmapWidth(font, ord(c))
    return len


def glutDrawText(text,x,y,font='hv18',gravity=''):
    """Draw the given text at given 2D position in window.

    If adjust == 'center', the text will be horizontally centered on
    the insertion point.
    If adjust == 'right', the text will be right aligned on the point.
    Any other setting will align the text left.
    Default is to center.
    """
    # !!! Do not use GLUT.glutBitmapLength(font, text)
    width = glutBitmapLength(font, text)
    height = glutFontHeight(font)
    w2,h2 = 0.5*width,0.4*height
    x -= w2
    y -= h2
    if 'E' in gravity:
        x += w2
    elif 'W' in gravity:
        x -= w2
    if 'N' in gravity:
        y += h2
    elif 'S' in gravity:
        y -= h2
    GL.glRasterPos2f(float(x),float(y));
    #GD.debug("RENDERING WITH FONT %s" % font) 
    glutRenderText(text,font)


# End

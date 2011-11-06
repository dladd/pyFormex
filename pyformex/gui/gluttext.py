# $Id$
##
##  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
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

"""2D text decorations using GLUT fonts

This module provides the basic functions for using the GLUT library in the
rendering of text on an OpenGL canvas.
"""

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

    - font: 'fixed', 'serif' or 'sans'
    - size: an int that will be rounded to the nearest available size.

    The return value is a 4-character string representing one of the
    GLUT fonts.
    """
    #pf.debug("INPUT %s,%s" % (font,size))
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
    
    If an unrecognized string is given, the default is 'hv18'.
    """
    return GLUTFONTS.get(font,GLUTFONTS['hv18'])



def glutFontHeight(font):
    """Return the height of the named glut font.

    This supposes that the last two characters of the name
    hold the font height.
    """
    return int(font[-2:])


#
# BV: !! gravity does not work yet!
#
def glutRenderText(text,font,gravity=''):
    """Draw a text in given font at the current rasterpoint.

    font should be one  of the legal fonts returned by glutFont().
    If text is not a string, it will be formatted to a string
    before drawing.
    After drawing, the rasterpos will have been updated!
    """
    if type(font) == str:
        font = glutFont(font)
    if gravity:
        curpos = GL.glGetFloatv(GL.GL_CURRENT_RASTER_POSITION)
        #print curpos
        GL.glRasterPos2f(curpos[0]-20.,curpos[1])
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


def glutDrawText(text,x,y,font='hv18',gravity='',spacing=1.0):
    """Draw a text at given 2D position in window.

    - text: a simple string, a multiline string or a list of strings. If it is
      a string, it will be splitted on the occurrence of '\\n' characters.
    - x,y: insertion position on the canvas
    - gravity: a string that determines the adjusting of the text with
      respect to the insert position. It can be a combination of one of the
      characters 'N or 'S' to specify the vertical positon, and 'W' or 'E'
      for the horizontal. The default(empty) string will center the text.
      
    """
    if type(text) is str:
        text = text.split('\n')
    nlines = len(text)
    widths = [ glutBitmapLength(font, t) for t in text ]
    fontheight = glutFontHeight(font)
    spacing *= fontheight
    width = max(widths)
    height = spacing*nlines 

    x,y = float(x),float(y)
    if 'S' in gravity:
        yi = y
    elif 'N' in gravity:
        yi = y + height
    else:
        yi = y + height/2

    
    for t,w in zip(text,widths):
        if 'E' in gravity:
            xi = x
        elif 'W' in gravity:
            xi = x - w
        else:
            xi = x - w/2
        yi -= spacing
        GL.glRasterPos2f(float(xi),float(yi))
        glutRenderText(t,font)



# End

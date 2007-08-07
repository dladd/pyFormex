# canvas.py
# $Id$
##
## This file is part of pyFormex 0.5 Release Mon Jul 30 13:38:48 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""2D decorations for the OpenGL canvas."""

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

# Needed to initialize the fonts
#GLUT.glutInit([])

import colors
#import colorscale

### Some drawing functions ###############################################

GLUTFONTS = {
    '9x15' : GLUT.GLUT_BITMAP_9_BY_15,
    '8x13' : GLUT.GLUT_BITMAP_8_BY_13,
    'tr10' : GLUT.GLUT_BITMAP_TIMES_ROMAN_10,
    'tr24' : GLUT.GLUT_BITMAP_TIMES_ROMAN_24,
    'hv10' : GLUT.GLUT_BITMAP_HELVETICA_10,
    'hv12' : GLUT.GLUT_BITMAP_HELVETICA_12,
    'hv18' : GLUT.GLUT_BITMAP_HELVETICA_18,
}

def glutFont(font):
    """Return GLUT font designation for the named font.

    The recognized font names are:
      fixed: '9x15', '8x13',
      times-roman: 'tr10', 'tr24'
      helvetica:   'hv10', 'hv12',  'hv18'
    If an unrecognized string is  given, the default is 9x15.
    """
    return GLUTFONTS.get(font,GLUTFONTS['9x15'])

def glutFontHeight(font):
    """Return the height of the named glut font.

    This supposes that the last two characters of the name
    hold the font height.
    """
    return int(font[-2:])


def drawGlutText(text,font):
    """Draw a text in given font at the current rasterpoint.

    font should be one  of the legal fonts returned by glutFont().
    If text is not a string, it will be formatted to a string
    before drawing.
    After drawing, the rasterpos will have been updated!
    """
    for character in str(text):
        GLUT.glutBitmapCharacter(font, ord(character))


def drawLine(x1,y1,x2,y2):
    """Draw a straight line from (x1,y1) to (x2,y2) in canvas coordinates."""
    GL.glBegin(GL.GL_LINES)
    GL.glVertex2f(x1, y1)
    GL.glVertex2f(x2, y2)
    GL.glEnd()


def drawGrid(x1,y1,x2,y2,nx,ny):
    """Draw a rectangular grid of lines
        
    The rectangle has (x1,y1) and and (x2,y2) as opposite corners.
    There are (nx,ny) subdivisions along the (x,y)-axis. So the grid
    has (nx+1) * (ny+1) lines. nx=ny=1 draws a rectangle.
    nx=0 draws 1 vertical line (at x1). nx=-1 draws no vertical lines.
    ny=0 draws 1 horizontal line (at y1). ny=-1 draws no horizontal lines.
    """
    GL.glBegin(GL.GL_LINES)
    ix = range(nx+1)
    if nx==0:
        jx = [1]
        nx = 1
    else:
        jx = ix[::-1] 
    for i,j in zip(ix,jx):
        x = (i*x2+j*x1)/nx
        GL.glVertex2f(x, y1)
        GL.glVertex2f(x, y2)

    iy = range(ny+1)
    if ny==0:
        jy = [1]
        ny = 1
    else:
        jy = iy[::-1] 
    for i,j in zip(iy,jy):
        y = (i*y2+j*y1)/ny
        GL.glVertex2f(x1, y)
        GL.glVertex2f(x2, y)
    GL.glEnd()


##def drawRect(x1,y1,x2,y2):
##    drawGrid(x1,y1,x2,y2,1,1)


def myBitmapLength(font, text):
    """ Compute the length in pixels of a text string in given font.

    We use our own fucntion to calculate the length because the builtin
    has a bug.
    """
    len = 0
    for c in text:
        len += GLUT.glutBitmapWidth(font, ord(c))
    return len


def drawText2D(text, x,y, font='9x15', adjust='left'):
    """Draw the given text at given 2D position in window.

    If adjust == 'center', the text will be horizontally centered on
    the insertion point.
    If adjust == 'right', the text will be right aligned on the point.
    Any other setting will align the text left.
    Default is to center.
    """
    if type(font) == str:
        font = glutFont(font)
    #print "font = ",font
    if adjust != 'left':
        len1 = myBitmapLength(font, text)
##  UNCOMMENT THESE LINES TO SEE WHEN gluBitmapLength GOES WRONG !
##        len2 = GLUT.glutBitmapLength(font, text)
##        if len1 != len2:
##            print "incorrect glutBitmapLength",len1,len2
        if adjust == 'center':
            x -= len1/2
        elif adjust == 'right':
            x -= len1 
    GL.glRasterPos2f(float(x),float(y));
    drawGlutText(text,font)


def unProject(x,y,win):
    "Map the window coordinates (x,y) to object coordinates."""
    win.makeCurrent()
    y = win.h-y
    model = GL.glGetFloatv(GL_MODELVIEW_MATRIX)
    proj = GL.glGetFloatv(GL_PROJECTION_MATRIX)
    view = GL.glGetIntegerv(GL_VIEWPORT)
    #print "Modelview matrix:",model
    #print "Projection matrix:",proj
    #print "Viewport:",view
    objx, objy, objz = GLU.gluUnProject(x,y,0.0,model,proj,view)
    #print "Coordinates: ",x,y," map to ",objx,objy
    return (objx,objy)


### Decorations ###############################################

class Decoration(object):
    """A decoration is a 2-D drawing at canvas position x,y.

    All decoration have at least the following attributes:
      x,y : (int) window coordinates of the insertion point
      draw() : function that draws the decoration at (x,y).
               This should only use openGL function that are
               allowed in a display list.
    """

    def __init__(self,x,y):
        """Create a decoration at acnvas coordinates x,y"""
        self.x = int(x)
        self.y = int(y)

    def draw(self,mode='wireframe'):
        """Draw the decoration."""
        pass

        
class Text(Decoration):
    """A viewport decoration showing a text."""

    def __init__(self,text,x,y,font='9x15',adjust='left',color=colors.black):
        """Create a text actor"""
        Decoration.__init__(self,x,y)
        self.text = str(text)
        self.font = font
        self.adjust = adjust
        self.color = colors.GLColor(color)

    def draw(self,mode='wireframe'):
        """Draw the text."""
        GL.glColor3fv(self.color)
        drawText2D(self.text,self.x,self.y,self.font,self.adjust)


class ColorLegend(Decoration):
    """A viewport decoration showing a colorscale legend."""
    def __init__(self,colorlegend,x,y,w,h,font='9x15',dec=2):
        Decoration.__init__(self,x,y)
        self.cl = colorlegend
        self.w = int(w)
        self.h = int(h)
        self.xgap = 4  # hor. gap between coloras and labels 
        self.ygap = 4  # vert. gap between labels
        self.font = font
        self.dec = 2   # number of decimals

    def draw(self,mode='wireframe'):
        n = len(self.cl.colors)
        x1 = float(self.x)
        x2 = float(self.x+self.w)
        y0 = float(self.y)
        dy = float(self.h)/n
        # colors
        y1 = y0
        for i,c in enumerate(self.cl.colors):
            #print c
            y2 = y0 + (i+1)*dy
            GL.glColor3f(*c)
            GL.glRectf(x1,y1,x2,y2)   
            y1 = y2
        # values
        x1 = x2 + self.xgap
        fh = glutFontHeight(self.font)
        dh = fh + self.ygap # vert. distance between successive labels
        y0 -= dh/2
        GL.glColor3f(*colors.black)
        for i,v in enumerate(self.cl.limits):
            y2 = y0 + i*dy
            if y2 >= y1 or i == 0:
                drawText2D(("%%.%df" % self.dec) % v,x1,y2)   
                y1 = y2 + dh
            

class Grid(Decoration):
    """A 2D-grid on the canvas."""
    def __init__(self,x1,y1,x2,y2,nx=1,ny=1,color=None,linewidth=None):
        Decoration.__init__(self,x1,y1)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.nx = nx
        self.ny = ny
        if color is None:
            self.color = None
        else:
            self.color = colors.GLColor(color)
        if linewidth is None:
            self.linewidth = None
        else:
            self.linewidth = float(linewidth)

    def draw(self,mode='wireframe'):
        if self.color:
            GL.glColor3fv(self.color)
        if self.linewidth:
            GL.glLineWidth(self.linewidth)
        drawGrid(self.x1,self.y1,self.x2,self.y2,self.nx,self.ny)
            

class Line(Decoration):
    """A straight line on the canvas."""
    def __init__(self,x1,y1,x2,y2):
        Decoration.__init__(self,x1,y1)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self,mode='wireframe'):
        drawLine(self.x1,self.y1,self.x2,self.y2)

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
"""2D decorations for the OpenGL canvas."""

from OpenGL import GL
from PyQt4 import QtOpenGL

from drawable import *
from text import *
from marks import TextMark

import colors
import gluttext

### Some drawing functions ###############################################


def drawDot(x,y):
    """Draw a dot at canvas coordinates (x,y)."""
    GL.glBegin(GL.GL_POINTS)
    GL.glVertex2f(x,y)
    GL.glEnd()


def drawLine(x1,y1,x2,y2):
    """Draw a straight line from (x1,y1) to (x2,y2) in canvas coordinates."""
    GL.glBegin(GL.GL_LINES)
    GL.glVertex2f(x1,y1)
    GL.glVertex2f(x2,y2)
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


def drawRect(x1,y1,x2,y2):
    """Draw the circumference of a rectangle."""
    drawGrid(x1,y1,x2,y2,1,1)


def drawRectangle(x1,y1,x2,y2,color,texture=None):
    """Draw a single rectangular quad."""
    x = array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    c = resize(asarray(color),(4,3))
    ##drawPolygons(coord,None,'flat',color=color,texture=texture)
    if texture is not None:
        glTexture(texture)
        t = [[0.,0.],[1.,0.],[1.,1.],[0.,1.]]
    else:
        t = None
    GL.glBegin(GL.GL_QUADS)
    for i in range(4):
        GL.glColor3fv(c[i])
        if texture is not None:
            GL.glTexCoord2fv(t[i]) 
        GL.glVertex2fv(x[i])
    GL.glEnd()


### Decorations ###############################################

class Decoration(Drawable):
    """A decoration is a 2-D drawing at canvas position x,y.

    All decorations have at least the following attributes:
    
    - x,y : (int) window coordinates of the insertion point
    - drawGL() : function that draws the decoration at (x,y).
               This should only use openGL function that are
               allowed in a display list.
               
    """

    def __init__(self,x,y,**kargs):
        """Create a decoration at canvas coordinates x,y"""
        self.x = int(x)
        self.y = int(y)
        if 'nolight' not in kargs:
            kargs['nolight'] = True
        if 'ontop' not in kargs:
            kargs['ontop'] = True
        Drawable.__init__(self,**kargs)


# Marks database: a dict with mark name and a function to draw
# the mark. The 
_marks_ = {
    'dot':drawDot,
    }

class Mark(Decoration):
    """A mark at a fixed position on the canvas."""
    def __init__(self,x,y,mark='dot',color=None,linewidth=None,**kargs):
        Decoration.__init__(self,x,y,**kargs)
        self.x = x
        self.y = y
        if not mark in _marks_:
            raise ValueError,"Unknown mark: %s" % mark
        self.mark = mark
        self.color = saneColor(color)
        self.linewidth = saneLineWidth(linewidth)


    def drawGL(self,**kargs):
        if self.color is not None:
            GL.glColor3fv(self.color)
        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)
        _marks_[self.mark](self.x,self.y)


class Line(Decoration):
    """A straight line on the canvas."""
    def __init__(self,x1,y1,x2,y2,color=None,linewidth=None,**kargs):
        Decoration.__init__(self,x1,y1,**kargs)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = saneColor(color)
        self.linewidth = saneLineWidth(linewidth)


    def drawGL(self,**kargs):
        if self.color is not None:
            GL.glColor3fv(self.color)
        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)
        drawLine(self.x1,self.y1,self.x2,self.y2)

        
## ## class QText(Decoration):
## ##     """A viewport decoration showing a text."""

## ##     def __init__(self,text,x,y,adjust='left',font=None,size=None,color=None):
## ##         """Create a text actor"""
## ##         Decoration.__init__(self,x,y)
## ##         self.text = str(text)
## ##         self.adjust = adjust
## ##         self.font = getFont(font,size)
## ##         self.color = saneColor(color)

## ##     count = 0
## ##     # QT text color does not seem to work good with display lists,
## ##     # therefore we redefine draw(), not drawGL()
## ##     def draw(self,mode='wireframe',color=None):
## ##         """Draw the text."""
## ##         self.count += 1
## ## #        pf.canvas.makeCurrent()
## ##         if self.color is not None:
## ##             GL.glColor3fv(self.color)
## ##         pf.canvas.renderText(self.x,pf.canvas.height()-self.y,self.text,self.font)
## ## #        pf.canvas.swapBuffers() 
## ## #        pf.canvas.updateGL() 


class GlutText(Decoration):
    """A viewport decoration showing a text string.

    - text: a simple string, a multiline string or a list of strings. If it is
      a string, it will be splitted on the occurrence of '\\n' characters.
    - x,y: insertion position on the canvas
    - gravity: a string that determines the adjusting of the text with
      respect to the insert position. It can be a combination of one of the
      characters 'N or 'S' to specify the vertical positon, and 'W' or 'E'
      for the horizontal. The default(empty) string will center the text.
    
    """

    def __init__(self,text,x,y,font='9x15',size=None,gravity=None,color=None,zoom=None,**kargs):
        """Create a text actor"""
        Decoration.__init__(self,x,y,**kargs)
        self.text = str(text)
        self.font = gluttext.glutSelectFont(font,size)

        if gravity is None:
            gravity = 'E'
        self.gravity = gravity
        self.color = saneColor(color)
        self.zoom = zoom

    def drawGL(self,**kargs):
        """Draw the text."""
        ## if self.zoom:
        ##     pf.canvas.zoom_2D(self.zoom)
        if self.color is not None: 
            GL.glColor3fv(self.color)
        gluttext.glutDrawText(self.text,self.x,self.y,font=self.font,gravity=self.gravity)
        ## if self.zoom:
        ##     pf.canvas.zoom_2D()

Text = GlutText


class ColorLegend(Decoration):
    """A labeled colorscale legend.

    When showing the distribution of some variable over a domain by means
    of a color encoding, the viewer expects some labeled colorscale as a
    guide to decode the colors. The ColorLegend decoration provides
    such a color legend. This class only provides the visual details of
    the scale. The conversion of the numerical values to the matching colors
    is provided by the :class:`colorscale.ColorLegend` class.

    Parameters:

    - `colorlegend`: a :class:`colorscale.ColorLegend` instance providing
      conversion between numerical values and colors
    - `x,y,w,h`: four integers specifying the position and size of the
      color bar rectangle
    - `ngrid`: int: number of intervals for the grid lines to be shown.
      If > 0, grid lines are drawn around the color bar and between the
      ``ngrid`` intervals.
      If = 0, no grid lines are drawn.
      If < 0 (default), the value is set equal to the number of colors (as
      set in the ``colorlegend``) or to 0 if this number is higher than 50.
    - `linewidth`: float: width of the grid lines. If not specified, the
      current canvas line width is used.
    - `nlabel`: int: number of intervals for the labels to be shown.
      If > 0, labels will be displayed at `nlabel` interval borders, if
      possible. The number of labels displayed thus will be ``nlabel+1``,
      or less if the labels would otherwise be too close or overlapping.
      If 0, no labels are shown. 
      If < 0 (default), a default number of labels is shown.
    - `font`, `size`: font and size to be used for the labels
    - `dec`: int: number of decimals to be used in the labels
    - `scale`: int: exponent of 10 for the scaling factor of the label values.
      The displayed values will be equal to the real values multiplied with
      ``10**scale``.
    - `lefttext`: bool: if True, the labels will be drawn to the left of the
      color bar. The default is to draw the labels at the right.

    Some practical guidelines:

    - The number of colors is defined by the ``colorlegend`` argument.
    - Large numbers of colors result inb a quasi continuous color scheme.
    - With a high number of colors, grid lines disturb the image, so either
      use ``ngrid=0`` or ``ngrid=`` to only draw a border around the colors.
    - With a small number of colors, set ``ngrid = len(colorlegend.colors)``
      to add gridlines between each color.
      Without it, the individual colors in the color bar may seem to be not
      constant, due to an optical illusion. Adding the grid lines reduces
      this illusion.
    - When using both grid lines and labels, set both ``ngrid`` and ``nlabel``
      to the same number or make one a multiple of the other. Not doing so
      may result in a very confusing picture.
    - The best practices are to use either a low number of colors (<=20) and
      the default ``ngrid`` and ``nlabel``, or a high number of colors (>=200)
      and the default values or a low value for ``nlabel``.

    The `ColorScale` example script provides opportunity to experiment with
    different settings.
    """
    def __init__(self,colorlegend,x,y,w,h,ngrid=0,linewidth=None,nlabel=-1,font=None,size=None,dec=2,scale=0,lefttext=False,**kargs):
        """Initialize the ColorLegend."""
        Decoration.__init__(self,x,y,**kargs)
        self.cl = colorlegend
        self.w = int(w)
        self.h = int(h)
        self.ngrid = int(ngrid)
        if self.ngrid < 0:
            self.ngrid = len(self.cl.colors)
            if self.ngrid > 50:
                self.ngrid = 0
        self.linewidth = saneLineWidth(linewidth)
        self.nlabel = int(nlabel)
        if self.nlabel < 0:
            self.nlabel = len(self.cl.colors)
        self.font = gluttext.glutSelectFont(font,size)
        self.dec = dec   # number of decimals
        self.scale = 10 ** scale # scale all numbers with 10**scale
        self.lefttext = lefttext
        self.xgap = 4  # hor. gap between color bar and labels 
        self.ygap = 4  # (min) vert. gap between labels


    def drawGL(self,**kargs):
        self.decorations = []
        n = len(self.cl.colors)
        pf.debug("NUMBER OF COLORS: %s" % n)
        x1 = float(self.x)
        x2 = float(self.x+self.w)
        y0 = float(self.y)
        dy = float(self.h)/n
        
        # colors
        y1 = y0
        #GL.glLineWidth(1.5)
        for i,c in enumerate(self.cl.colors):
            y2 = y0 + (i+1)*dy
            GL.glColor3f(*c)
            GL.glRectf(x1,y1,x2,y2)   
            y1 = y2
            
        if self.nlabel > 0 or self.ngrid > 0:
            GL.glColor3f(*colors.black)

        # labels
        if self.nlabel > 0:
            fh = gluttext.glutFontHeight(self.font)
            pf.debug("FONT HEIGHT %s" % fh)
            # minimum label distance
            dh = fh + self.ygap
            maxlabel = float(self.h)/dh # maximum n umber of labels
            if self.nlabel <= maxlabel:
                dh = float(self.h)/self.nlabel # respect requested number
                
            if self.lefttext:
                x1 = x1 - self.xgap
                gravity = 'W'
            else:
                x1 = x2 + self.xgap
                gravity = 'E'

            # FOR 3-VALUE SCALES THIS SHOULD BE DONE IN TWO PARTS,
            # FROM THE CENTER OUTWARDS, AND THEN ADDING THE
            # MIN AND MAX VALUES
            for i,v in enumerate(self.cl.limits):
                y2 = y0 + i*dy
                if y2 >= y1 or i == 0 or i == len(self.cl.limits)-1:
                    if y2 >= self.y+self.h-dh/2 and i < len(self.cl.limits)-1:
                        continue
                    t = Text(("%%.%df" % self.dec) % (v*self.scale),x1,round(y2),font=self.font,gravity=gravity)
                    self.decorations.append(t)
                    t.drawGL(**kargs)
                    y1 = y2 + dh

        # grid: after values, to be on top
        if self.ngrid > 0:
            if self.linewidth is not None:
                GL.glLineWidth(self.linewidth)
            drawGrid(self.x,self.y,self.x+self.w,self.y+self.h,1,self.ngrid)


    def use_list(self):
        Decoration.use_list(self)
        for t in self.decorations:
            t.use_list()


class Rectangle(Decoration):
    """A 2D-rectangle on the canvas."""
    def __init__(self,x1,y1,x2,y2,color=None,texture=None,**kargs):
        Decoration.__init__(self,x1,y1,**kargs)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.setColor(color,ncolors=4)
        self.setTexture(texture)

    def drawGL(self,**kargs):
        drawRectangle(self.x1,self.y1,self.x2,self.y2,self.color,self.texture)


class Grid(Decoration):
    """A 2D-grid on the canvas."""
    def __init__(self,x1,y1,x2,y2,nx=1,ny=1,color=None,linewidth=None,**kargs):
        Decoration.__init__(self,x1,y1,**kargs)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.nx = nx
        self.ny = ny
        self.color = saneColor(color)
        self.linewidth = saneLineWidth(linewidth)

    def drawGL(self,**kargs):
        if self.color is not None:
            GL.glColor3fv(self.color)
        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)
        drawGrid(self.x1,self.y1,self.x2,self.y2,self.nx,self.ny)


class LineDrawing(Decoration):
    """A collection of straight lines on the canvas."""
    def __init__(self,data,color=None,linewidth=None,**kargs):
        """Initially a Line Drawing.

        data can be a 2-plex Formex or equivalent coordinate data.
        The z-coordinates of the Formex are unused.
        A (n,2,2) shaped array will do as well.
        """
        data = data.view()
        data = data.reshape((-1,2,data.shape[-1]))
        data = data[:,:,:2]
        self.data = data.astype(Float)
        x1,y1 = self.data[0,0]
        Decoration.__init__(self,x1,y1,**kargs)
        self.color = saneColor(color)
        self.linewidth = saneLineWidth(linewidth)
    

    def drawGL(self,**kargs):
        if self.color is not None:
            GL.glColor3fv(self.color)
        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)
        GL.glBegin(GL.GL_LINES)
        for e in self.data:
            GL.glVertex2fv(e[0])
            GL.glVertex2fv(e[1])
        GL.glEnd()


# Not really a decoration, though it could be made into one
# Can this be merged with actors.AxesActor ?
#

class Triade(Drawable):
    """An OpenGL actor representing a triade of global axes.
    
    - `pos`: position on the canvas: two characters, of which first sets
      horizontal position ('l', 'c' or 'r') and second sets vertical
      position ('b', 'c' or 't').

    - `size`: size in pixels of the zone displaying the triade.

    - `pat`: shape to be drawn in the coordinate planes. Default is a square.
      '16' givec a triangle. '' disables the planes.

    - `legend`: text symbols to plot at the end of the axes. A 3-character
      string or a tuple of 3 strings.
      
    """

    def __init__(self,pos='lb',siz=100,pat='3:012934',legend='xyz',color=[red,green,blue,cyan,magenta,yellow],**kargs):
        Drawable.__init__(self,**kargs)
        self.pos = pos
        self.siz = siz
        self.pat = pat
        self.legend = legend
        self.color = color

 
    def _draw_me(self):
        """Draw the triade components."""
        GL.glBegin(GL.GL_LINES)
        pts = Formex('1').coords.reshape(-1,3)
        GL.glColor3f(*black)
        for i in range(3):
            #GL.glColor(*self.color[i])
            for x in pts:
                GL.glVertex3f(*x)
            pts = pts.rollAxes(1)
        GL.glEnd()
        # Coord planes
        if self.pat:
            GL.glBegin(GL.GL_TRIANGLES)
            pts = Formex(self.pat)
            #pts += pts.reverse()
            pts = pts.scale(0.5).coords.reshape(-1,3)
            for i in range(3):
                pts = pts.rollAxes(1)
                GL.glColor3f(*self.color[i])
                for x in pts:
                    GL.glVertex3f(*x)
            GL.glEnd()
        # Coord axes denomination
        for i,x in enumerate(self.legend):
            p = unitVector(i)*1.1
            t = TextMark(p,x)
            t.drawGL()

 
    def _draw_relative(self):
        """Draw the triade in the origin and with the size of the 3D space."""
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glTranslatef (*self.pos) 
        GL.glScalef (self.size,self.size,self.size)
        self._draw_me()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()


    def _draw_absolute(self):
        """Draw the triade in the lower left corner."""
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        # Cancel the translations
        rot = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX)
        rot[3,0:3] = [0.,0.,0.]
        GL.glLoadMatrixf(rot)
        vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
        x,y,w,h = vp
        w0,h0 = self.siz,self.siz # we force aspect ratio 1
        if self.pos[0] == 'l':
            x0 = x
        elif self.pos[0] =='r':
            x0 = x + w-w0
        else:
            x0 = x + (w-w0)/2     
        if self.pos[1] == 'b':
            y0 = y
        elif self.pos[1] =='t':
            y0 = y + h-h0
        else:
            y0 = y + (h-h0)/2
        GL.glViewport(x0,y0,w0,h0)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        fovy = 45.
        fv = tand(fovy*0.5)
        fv *= 4.
        fh = fv
        # BEWARE: near/far should be larger than size, but not very large
        # or the depth sort will fail
        frustum = (-fh,fh,-fv,fv,-3.,100.)
        GL.glOrtho(*frustum)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable (GL.GL_BLEND)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK,GL.GL_FILL)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glClearDepth(1.0)
        GL.glDepthMask (GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self._draw_me()
        GL.glViewport(*vp)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()

    def draw(self,**kargs):
        self._draw_absolute()

# End

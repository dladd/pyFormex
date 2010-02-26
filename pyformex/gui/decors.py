# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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
    drawGrid(x1,y1,x2,y2,1,1)


def drawRectangle(x1,y1,x2,y2,color):
    color = resize(asarray(color),(4,3))
    coord = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    GL.glBegin(GL.GL_QUADS)
    for c,x in zip(color,coord):
        GL.glColor3fv(c)
        GL.glVertex2fv(x)
    GL.glEnd()


### Decorations ###############################################

class Decoration(Drawable):
    """A decoration is a 2-D drawing at canvas position x,y.

    All decoration have at least the following attributes:
      x,y : (int) window coordinates of the insertion point
      drawGL() : function that draws the decoration at (x,y).
               This should only use openGL function that are
               allowed in a display list.
    """

    def __init__(self,x,y):
        """Create a decoration at canvas coordinates x,y"""
        self.x = int(x)
        self.y = int(y)
        Drawable.__init__(self)


# Marks database: a dict with mark name and a function to draw
# the mark. The 
_marks_ = {
    'dot':drawDot,
    }

class Mark(Decoration):
    """A mark at a fixed position on the canvas."""
    def __init__(self,x,y,mark='dot',color=None,linewidth=None):
        Decoration.__init__(self,x,y)
        self.x = x
        self.y = y
        if not mark in _marks_:
            raise ValueError,"Unknown mark: %s" % mark
        self.mark = mark
        self.color = saneColor(color)
        self.linewidth = saneLineWidth(linewidth)


    def drawGL(self,mode='wireframe',color=None):
        if self.color is not None:
            GL.glColor3fv(self.color)
        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)
        _marks_[self.mark](self.x,self.y)


class Line(Decoration):
    """A straight line on the canvas."""
    def __init__(self,x1,y1,x2,y2,color=None,linewidth=None):
        Decoration.__init__(self,x1,y1)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = saneColor(color)
        self.linewidth = saneLineWidth(linewidth)


    def drawGL(self,mode='wireframe',color=None):
        if self.color is not None:
            GL.glColor3fv(self.color)
        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)
        drawLine(self.x1,self.y1,self.x2,self.y2)

        
## class QText(Decoration):
##     """A viewport decoration showing a text."""

##     def __init__(self,text,x,y,adjust='left',font=None,size=None,color=None):
##         """Create a text actor"""
##         Decoration.__init__(self,x,y)
##         self.text = str(text)
##         self.adjust = adjust
##         self.font = getFont(font,size)
##         self.color = saneColor(color)

##     count = 0
##     # QT text color does not seem to work good with display lists,
##     # therefore we redefine draw(), not drawGL()
##     def draw(self,mode='wireframe',color=None):
##         """Draw the text."""
##         self.count += 1
## #        GD.canvas.makeCurrent()
##         if self.color is not None:
##             GL.glColor3fv(self.color)
##         GD.canvas.renderText(self.x,GD.canvas.height()-self.y,self.text,self.font)
## #        GD.canvas.swapBuffers() 
## #        GD.canvas.updateGL() 


class GlutText(Decoration):
    """A viewport decoration showing a text string.

    - text: a simple string, a multiline string or a list of strings. If it is
      a string, it will be splitted on the occurrence of '\n' characters.
    - x,y: insertion position on the canvas
    - gravity: a string that determines the adjusting of the text with
      respect to the insert position. It can be a combination of one of the
      characters 'N or 'S' to specify the vertical positon, and 'W' or 'E'
      for the horizontal. The default(empty) string will center the text.
    """

    def __init__(self,text,x,y,font='9x15',size=None,gravity=None,color=None,zoom=None):
        """Create a text actor"""
        Decoration.__init__(self,x,y)
        self.text = str(text)
        self.font = gluttext.glutSelectFont(font,size)

        if gravity is None:
            gravity = 'E'
        self.gravity = gravity
        self.color = saneColor(color)
        self.zoom = zoom

    def drawGL(self,mode='wireframe',color=None):
        """Draw the text."""
        ## if self.zoom:
        ##     GD.canvas.zoom_2D(self.zoom)
        if self.color is not None: 
            GL.glColor3fv(self.color)
        gluttext.glutDrawText(self.text,self.x,self.y,font=self.font,gravity=self.gravity)
        ## if self.zoom:
        ##     GD.canvas.zoom_2D()

Text = GlutText


class ColorLegend(Decoration):
    """A viewport decoration showing a colorscale legend."""
    def __init__(self,colorlegend,x,y,w,h,font=None,size=None,dec=2,scale=0,grid=0,linewidth=None,lefttext=False):
        Decoration.__init__(self,x,y)
        self.cl = colorlegend
        self.w = int(w)
        self.h = int(h)
        self.xgap = 4  # hor. gap between colors and labels 
        self.ygap = 4  # vert. gap between labels
        #self.font = getFont(font,size)
        self.font = gluttext.glutSelectFont(font,size)
        self.dec = dec   # number of decimals
        self.scale = 10 ** scale # scale all numbers with 10**scale
        self.grid = abs(int(grid))
        self.linewidth = saneLineWidth(linewidth)
        self.lefttext = lefttext

    def drawGL(self,mode='wireframe',color=None):
        #from draw import drawText
        n = len(self.cl.colors)
        x1 = float(self.x)
        x2 = float(self.x+self.w)
        y0 = float(self.y)
        dy = float(self.h)/n
        # colors
        y1 = y0
        GD.debug("START COLORS AT %s" % y0)
        GL.glLineWidth(1.5)
        for i,c in enumerate(self.cl.colors):
            y2 = y0 + (i+1)*dy
            GL.glColor3f(*c)
            GL.glRectf(x1,y1,x2,y2)   
            y1 = y2
        # values
        if self.lefttext:
            x1 = x1 - self.xgap
            gravity = 'W'
        else:
            x1 = x2 + self.xgap
            gravity = 'E'
        fh = gluttext.glutFontHeight(self.font)
        GD.debug("FONT HEIGHT %s" % fh)
        dh = fh + self.ygap # vert. distance between successive labels
        #y0 -= 0.25*fh  # 0.5*fh seems more logic, but character pos is biased
        GD.debug("FIRST TEXT AT %s" % y0)
        GL.glColor3f(*colors.black)
        self.decorations = []
        for i,v in enumerate(self.cl.limits):
            y2 = y0 + i*dy
            GD.debug("next y = %s" % y2)
            if y2 >= y1 or i == 0:
                GD.debug("drawing at %s" % y2)
                t = Text(("%%.%df" % self.dec) % (v*self.scale),x1,y2,font=self.font,gravity=gravity)
                self.decorations.append(t)
                t.drawGL(mode,color)
                y1 = y2 + dh
        # grid: after values, to be on top
        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)
        if self.grid:
            drawGrid(self.x,self.y,self.x+self.w,self.y+self.h,1,self.grid)

    def use_list(self):
        Decoration.use_list(self)
        for t in self.decorations:
            t.use_list()
            ## if TA not in GD.canvas.decorations:
            ##     GD.canvas.addDecoration(TA)


class Rectangle(Decoration):
    """A 2D-rectangle on the canvas."""
    def __init__(self,x1,y1,x2,y2,color=None):
        Decoration.__init__(self,x1,y1)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.setColor(color,ncolors=4)

    def drawGL(self,mode='wireframe',color=None):
        drawRectangle(self.x1,self.y1,self.x2,self.y2,self.color)


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
        self.color = saneColor(color)
        self.linewidth = saneLineWidth(linewidth)

    def drawGL(self,mode='wireframe',color=None):
        if self.color is not None:
            GL.glColor3fv(self.color)
        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)
        drawGrid(self.x1,self.y1,self.x2,self.y2,self.nx,self.ny)


class LineDrawing(Decoration):
    """A collection of straight lines on the canvas."""
    def __init__(self,data,color=None,linewidth=None):
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
        Decoration.__init__(self,x1,y1)
        self.color = saneColor(color)
        self.linewidth = saneLineWidth(linewidth)
    

    def drawGL(self,mode=None,color=None):
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

class Triade(Drawable):
    """An OpenGL actor representing a triade of global axes.

    - `pos`: position on the canvas: two charaters, of which first sets
       horizontal position ('l', 'c' or 'r') and second sets vertical
       position ('b', 'c' or 't').
    - `size`: size in pixels of the zone displaying the triade.
    - `pat`: shape to be drawn in the coordinate planes. Default is a square.
      '16' givec a triangle. '' disables the planes.
    - `legend`: text symbols to plot at the end of the axes. A 3-character
      string or a tuple of 3 strings.
    """

    def __init__(self,pos='lb',siz=100,pat='12-34',legend='xyz',color=[red,green,blue,cyan,magenta,yellow]):
        Drawable.__init__(self)
        self.pos = pos
        self.siz = siz
        self.pat = pat
        self.legend = legend
        self.color = color

 
    def _draw_me(self):
        """Draw the triade components."""
        GL.glBegin(GL.GL_LINES)
        pts = Formex(pattern('1')).coords.reshape(-1,3)
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
            pts = Formex(mpattern(self.pat))
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

    def draw(self,mode='wireframe',color=None):
        self._draw_absolute()

# End

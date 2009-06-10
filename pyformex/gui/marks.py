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
"""OpenGL marks for annotating 3D actors."""

from OpenGL import GL,GLU
from colors import *
from formex import *
from drawable import *
from text import *
 
### Marks ###############################################

class Mark(Drawable):
    """An 2D drawing inserted at a 3D position of the scene.

    The minimum attributes and methods are:
      pos    : 3D point where the mark will be drawn
      draw() : function to draw the mark
    """
    
    def __init__(self,pos):
        self.pos = pos
        Drawable.__init__(self)


class AxesMark(Mark):
    """Two viewport axes drawn at a 3D position."""
    def __init__(self,pos,color=None):
        Mark.__init__(self,pos)
        self.color = saneColor(color)

    def drawGL(self,mode='wireframe',color=None):
        if self.color is not None:
            GL.glColor3fv(self.color)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
        GL.glRasterPos3fv(self.pos)
        a =  0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0x80
        b = 0x00,0x00,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,0x00
        bitmap = [b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,a,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b]
        GL.glBitmap(81,81,41,41,0,0,bitmap)


class TextMark(Mark):
    """A text drawn at a 3D position."""
    
    def __init__(self,pos,text,color=None,font=None,size=None):
        Mark.__init__(self,pos)
        self.text = text
        self.color = saneColor(color)
        self.font = gluttext.getFont(font,size)

    def drawGL(self,mode=None,color=None):
        if self.color is not None:
            GL.glColor3fv(self.color)
        GL.glRasterPos3fv(self.pos)

    def use_list(self):
        Mark.use_list(self)
        gluttext.glutRenderText(self.text,self.font)
        #x,y,z = self.pos
        #GD.canvas.renderText(x,y,z,self.text,self.font)


import gluttext
class MarkList(Mark):
    """A list of numbers drawn at 3D positions."""
    
    def __init__(self,pos,val,color=black,font='sans',size=18):
        """Create a number list.

        pos is an (N,3) array of positions.
        val is an (N,) array of marks to be plot at those positions.

        While intended to plot integer numbers, val can be any object
        that allows index operations for the required length N and allows
        its items to be formatted as a string.
        """
        if len(val) < len(pos):
            raise ValueError,"Not enough values for positions"
        Mark.__init__(self,pos)
        self.val = val
        self.color = saneColor(color)
        self.font = gluttext.glutSelectFont(font,size)
        #self.font = getFont(font,size)


    def draw(self,mode=None,color=None):
        if self.color is not None:
            GL.glColor3fv(self.color)
        for p,v in zip(self.pos,self.val):
            GL.glRasterPos3fv(p)
            gluttext.glutRenderText(str(v),self.font)
            #x,y,z = p
            #GD.canvas.renderText(x,y,z,str(v))


    def drawpick(self):
        """This functions mimicks the drawing of a number list for picking."""
        GL.glSelectBuffer(16+3*len(self.val))
        GL.glRenderMode(GL.GL_SELECT)
        GL.glInitNames() # init the name stack
        for p,v in zip(self.pos,self.val):
            GL.glPushName(v)
            GL.glRasterPos3fv(p)
            #drawGlutText(str(v),self.font)
            GL.glPopName()
        buf = GL.glRenderMode(GL.GL_RENDER)
        numbers =[]
        for r in buf:
            numbers += map(int,r[2])
        return numbers



# End

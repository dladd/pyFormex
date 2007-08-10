# marks.py
# $Id$
##
## This file is part of pyFormex 0.5 Release Fri Aug 10 12:04:07 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""OpenGL marks for annotating 3D actors."""

from OpenGL import GL,GLU,GLUT
from colors import *
from formex import *
from decors import glutFont,drawGlutText

 
### Marks ###############################################

class Mark(object):
    """An 2D drawing inserted at a 3D position of the scene.

    The minimum attributes and methods are:
      pos    : 3D point where the mark will be drawn
      draw() : function to draw the mark
    """
    
    def __init__(self,pos):
        self.pos = pos

    def draw(self,mode='wireframe'):
        pass


class TextMark(Mark):
    """A text drawn at a 3D position."""
    
    def __init__(self,pos,text,font='9x15'):
        Mark.__init__(self,pos)
        self.text = text
        self.font = glutFont(font)

    def draw(self,mode='wireframe'):
        GL.glColor3f(0.0,0.0,0.0)
        GL.glRasterPos3fv(self.pos)
        drawGlutText(self.text,self.font)

        
class MarkList(Mark):
    """A list of numbers drawn at 3D positions."""
    
    def __init__(self,pos,val,font='9x15'):
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
        self.font = glutFont(font)

    def draw(self,mode='wireframe'):
        GL.glColor3f(0.0,0.0,0.0)
        for p,v in zip(self.pos,self.val):
            GL.glRasterPos3fv(p)
            drawGlutText(str(v),self.font)


    def drawpick(self):
        GL.glSelectBuffer(16+3*len(self.val))
        GL.glRenderMode(GL.GL_SELECT)
        GL.glInitNames() # init the name stack
        for p,v in zip(self.pos,self.val):
            GL.glPushName(v)
            GL.glRasterPos3fv(p)
            drawGlutText(str(v),self.font)
            GL.glPopName()
        buf = GL.glRenderMode(GL.GL_RENDER)
        numbers =[]
        for r in buf:
            print r[2]
            numbers += map(int,r[2])
        return numbers
        

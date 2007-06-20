# marks.py
# $Id$
"""OpenGL marks for annotating 3D actors."""

from OpenGL import GL,GLU,GLUT
from colors import *
from formex import *
from decors import glutFont 

 
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
        for character in self.text:
            GLUT.glutBitmapCharacter(self.font, ord(character));
        GL.glFlush()

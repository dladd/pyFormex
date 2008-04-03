# $Id$
##
## This file is part of pyFormex 0.6 Release Sun Sep 30 14:33:15 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Basic OpenGL drawing functions.

The functions in this module should be exact emulations of the
functions in the compiled library.
"""

from OpenGL import GL,GLU
from numpy import *

def glColor(color,alpha=1.0):
    """Set the OpenGL color, possibly with transparency.

    color is a tuple of 3 real values.
    alpha is a single real value.
    All values are between 0.0 and 1.0
    """
    if alpha == 1.0:
        GL.glColor3fv(color)
    else:
        GL.glColor4fv(append(color,alpha)) 


def drawLines(x,c=None):
    """Draw a collection of lines.

    x : float (nlines,2,3) : coordinates.
    c : float (nlines,3) or (nlines,2,3) : color(s)
    If two colors per line are given, and rendering mode is flat,
    the second color will be used.
    """
    GL.glBegin(GL.GL_LINES)
    if c is None:
        for xi in x.reshape((-1,3)):
            GL.glVertex3fv(xi)
    elif c.ndim == 2:
        for xi,ci in zip(x,c):
            GL.glColor3fv(ci)
            GL.glVertex3fv(xi[0])
            GL.glVertex3fv(xi[1])
    elif c.ndim == 3:
        for xi,ci in zip(x.reshape((-1,3)),c.reshape((-1,3))):
            GL.glColor3fv(ci[0])
            GL.glVertex3fv(xi[0])
    GL.glEnd()


def drawPolygons(x,n=None,c=None,alpha=1.0):
    """Draw a collection of polygons.

    x : float (nel,nplex,3) : coordinates.
    n : float (nel,3) : normals.
    c : float (nel,3) or (nel,nplex,3) : color(s)
    If nplex colors per triangle are given, and shading mode is flat,
    the last color will be used.
    """
    x = x.astype(float32)
    nplex = x.shape[1]
    #print x.shape
    #print x.dtype
    #print x.flags
    if n is not None:
        n = n.astype(float32)
        #print "NORMAL",n.shape
        #print n.dtype
        #print n.flags
    if c is not None:
        c = c.astype(float32)
        #print "COLOR",c.shape

    if nplex < 3:
        return
    elif nplex == 3:
        GL.glBegin(GL.GL_TRIANGLES)
    elif nplex == 4:
        GL.glBegin(GL.GL_QUADS)
    else:
        GL.glBegin(GL.GL_POLYGON)

    if c is None:
        if n is None:
            for xi in x.reshape((-1,3)):
                GL.glVertex3fv(xi)
        elif n.ndim == 2:
            for xi,ni in zip(x,n):
                GL.glNormal3fv(ni)
                for j in range(nplex):
                    GL.glVertex3fv(xi[j])
        elif n.ndim == 3:
            for i in range(x.shape[0]):
                for xij,nij in zip(x[i],n[i]):
                    GL.glNormal3fv(nij)
                    GL.glVertex3fv(xij)
                
    elif c.ndim == 2:
        if n is None:
            for xi,ci in zip(x,c):
                GL.glColor3fv(ci)
                for j in range(nplex):
                    GL.glVertex3fv(xi[j])
        elif n.ndim == 2:
            for xi,ni,ci in zip(x,n,c):
                GL.glColor3fv(ci)
                GL.glNormal3fv(ni)
                for j in range(nplex):
                    GL.glVertex3fv(xi[j])
        elif n.ndim == 3:
            for xi,ni,ci in zip(x,n,c):
                GL.glColor3fv(ci)
                for j in range(nplex):
                    GL.glNormal3fv(ni[j])
                    GL.glVertex3fv(xi[j])
                    
    elif c.ndim == 3:
        if n is None:
            for xi,ci in zip(x.reshape((-1,3)),c.reshape((-1,3))):
                GL.glColor3fv(ci[0])
                GL.glVertex3fv(xi[0])
        elif n.ndim == 2:
            for xi,ni,ci in zip(x,n,c):
                GL.glNormal3fv(ni)
                for j in range(nplex):
                    GL.glColor3fv(ci[j])
                    GL.glVertex3fv(xi[j])
        elif n.ndim == 3:
            for xi,ni,ci in zip(x.reshape((-1,3)),n.reshape((-1,3)),c.reshape((-1,3))):
                GL.glColor3fv(ci)
                GL.glNormal3fv(ni)
                GL.glVertex3fv(xi)
    GL.glEnd()

    
### End

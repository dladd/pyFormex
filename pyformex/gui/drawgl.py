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
external functions in the compiled library.
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


def glObjType(nplex):
    if nplex == 1:
        objtype = GL.GL_POINTS
    elif nplex == 2:
        objtype = GL.GL_LINES
    elif nplex == 3:
        objtype = GL.GL_TRIANGLES
    elif nplex == 4:
        objtype = GL.GL_QUADS
    else:
        objtype = GL.GL_POLYGON
    return objtype


def drawPolygons(x,n,c,alpha):
    """Draw a collection of polygons.

    x : float (nel,nplex,3) : coordinates.
    n : float (nel,3) or (nel,nplex,3) : normals.
    c : float (nel,3) or (nel,nplex,3) : color(s)
    If nplex colors per element are given, and shading mode is flat,
    the last color will be used.
    """
    x = x.astype(float32)
    nplex = x.shape[1]
    if n is not None:
        n = n.astype(float32)
    if c is not None:
        c = c.astype(float32)

    objtype = glObjType(nplex)
    if nplex <= 4:

        GL.glBegin(objtype)
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

    else:

        if c is None:
            if n is None:
                for xi in x:
                    GL.glBegin(objtype)
                    for j in range(nplex):
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()
            elif n.ndim == 2:
                for xi,ni in zip(x,n):
                    GL.glBegin(objtype)
                    GL.glNormal3fv(ni)
                    for j in range(nplex):
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()
            elif n.ndim == 3:
                for i in range(x.shape[0]):
                    GL.glBegin(objtype)
                    for xij,nij in zip(x[i],n[i]):
                        GL.glNormal3fv(nij)
                        GL.glVertex3fv(xij)
                    GL.glEnd()

        elif c.ndim == 2:
            if n is None:
                for xi,ci in zip(x,c):
                    GL.glBegin(objtype)
                    GL.glColor3fv(ci)
                    for j in range(nplex):
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()
            elif n.ndim == 2:
                for xi,ni,ci in zip(x,n,c):
                    GL.glBegin(objtype)
                    GL.glColor3fv(ci)
                    GL.glNormal3fv(ni)
                    for j in range(nplex):
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()
            elif n.ndim == 3:
                for xi,ni,ci in zip(x,n,c):
                    GL.glBegin(objtype)
                    GL.glColor3fv(ci)
                    for j in range(nplex):
                        GL.glNormal3fv(ni[j])
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()

        elif c.ndim == 3:
            if n is None:
                for xi,ci in zip(x,c):
                    GL.glBegin(objtype)
                    GL.glColor3fv(ci)
                    for j in range(nplex):
                        GL.glColor3fv(ci[j])
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()
            elif n.ndim == 2:
                for xi,ni,ci in zip(x,n,c):
                    GL.glBegin(objtype)
                    GL.glNormal3fv(ni)
                    for j in range(nplex):
                        GL.glColor3fv(ci[j])
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()
            elif n.ndim == 3:
                for xi,ni,ci in zip(x,n,c):
                    GL.glBegin(objtype)
                    for j in range(nplex):
                        GL.glColor3fv(ci[j])
                        GL.glNormal3fv(ni[j])
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()

        

def pickPolygons(x):
    """Mimics drawPolygons for picking purposes.

    x : float (nel,nplex,3) : coordinates.
    """
    nplex = x.shape[1]
    objtype = glObjType(nplex)

    for i,xi in enumerate(x): 
        GL.glPushName(i)
        GL.glBegin(objtype)
        for xij in xi:
            GL.glVertex3fv(xij)
        GL.glEnd()
        GL.glPopName()
    
### End

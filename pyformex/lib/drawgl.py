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
"""Basic OpenGL drawing functions.

The functions in this module should be exact emulations of the
external functions in the compiled library lib.drawgl.

These are low level functions that should normally not be used by
the user.
"""

# There should be no other imports here than OpenGL and numpy
from OpenGL import GL,GLU
from numpy import *

accelerated = False

def get_version():
    return 0


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


def glTexture(texid):
    """Render-time texture environment setup"""
    # Configure the texture rendering parameters
    GL.glEnable(GL.GL_TEXTURE_2D)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_DECAL)
    # Re-select our texture, could use other generated textures
    # if we had generated them earlier...
    GL.glBindTexture(GL.GL_TEXTURE_2D,texid)


## def draw_tex_polygons(x,t,texid,objtype):
##     """Draw a collection of polygons.

##     x : float (nel,nplex,3) : coordinates.
##     t : float (nplex,3) or (nel,nplex,3) : texture coords
##     texid : id of an OpenGL 2D texture object
##     objtype : GL Object type (-1 = auto)
##     """
##     glTexture(texid)
##     x = x.astype(float32)
##     nplex = x.shape[1]
##     if t is not None:
##         t = t.astype(float32)
##     if objtype < 0:
##         objtype = glObjType(nplex)
        
##     if nplex <= 4 and glObjType(nplex) == objtype:
##         GL.glBegin(objtype)
##         if t.ndim == 2:
##             for xi in x:
##                 for j in range(nplex):
##                     GL.glTexCoord2fv(t[j]) 
##                     GL.glVertex3fv(xi[j])

##         elif t.ndim == 3:
##             for xi,ti in zip(x.reshape((-1,3)),t.reshape((-1,3))):
##                 GL.glTexCoord2fv(t) 
##                 GL.glVertex3fv(xi)
##         GL.glEnd()

                    
def draw_tex_polygons(x,n,c,t,alpha,texid,objtype):
    """Draw a collection of polygons.

    x : float (nel,nplex,3) : coordinates.
    n : float (nel,3) or (nel,nplex,3) : normals.
    c : float (nel,3) or (nel,nplex,3) : color(s)
    t : float (nplex,3) or (nel,nplex,3) : texture coords
    alpha : float
    texid : id of an OpenGL 2D texture object
    objtype : GL Object type (-1 = auto)

    If nplex colors per element are given, and shading mode is flat,
    the last color will be used.
    """
    print "draw_tex_polygons"
    x = x.astype(float32)
    nelems,nplex = x.shape[:2]
    ndim = cdim = tdim = 0
    if n is not None:
        n = n.astype(float32)
        ndim = n.ndim
    if c is not None:
        c = c.astype(float32)
        cdim = c.ndim
    if t is not None:
        t = t.astype(float32)
        tdim = t.ndim
        glTexture(texid)
    if objtype < 0:
        objtype = glObjType(nplex)

    print nelems,nplex,ndim,cdim,tdim,texid,objtype
    
    simple = nplex <= 4 and glObjType(nplex) == objtype
    if simple:
        GL.glBegin(objtype)
        if cdim == 1:       # single color 
            glColor(c,alpha)
        for i in range(nelems):
            if cdim == 2:
                glColor(c[i],alpha)
            if ndim == 2:
                GL.glNormal3fv(n[i])
            for j in range(nplex):
                if cdim == 3:
                    glColor(c[i,j],alpha)
                if ndim == 3:
                    GL.glNormal3fv(n[i,j])
                if tdim == 2:
                    GL.glTexCoord2fv(t[j]) 
                elif tdim == 3:
                    GL.glTexCoord2fv(t[i,j]) 
                GL.glVertex3fv(x[i,j])
        GL.glEnd()

    else:
        if cdim == 1:       # single color 
            glColor(c,alpha)
        for i in range(nelems):
            GL.glBegin(objtype)
            if cdim == 2:
                glColor(c[i],alpha)
            if ndim == 2:
                GL.glNormal3fv(n[i])
            for j in range(nplex):
                if cdim == 3:
                    glColor(c[i,j],alpha)
                if ndim == 3:
                    GL.glNormal3fv(n[i,j])
                if tdim == 2:
                    GL.glTexCoord2fv(t[j]) 
                elif tdim == 3:
                    GL.glTexCoord2fv(t[i,j]) 
                GL.glVertex3fv(x[i,j])
            GL.glEnd()

                    
def draw_polygons(x,n,c,alpha,objtype):
    """Draw a collection of polygons.

    x : float (nel,nplex,3) : coordinates.
    n : float (nel,3) or (nel,nplex,3) : normals.
    c : float (nel,3) or (nel,nplex,3) : color(s)
    alpha : float
    objtype : GL Object type (-1 = auto)

    If nplex colors per element are given, and shading mode is flat,
    the last color will be used.
    """
    x = x.astype(float32)
    nplex = x.shape[1]
    if n is not None:
        n = n.astype(float32)
    if c is not None:
        c = c.astype(float32)
    if objtype < 0:
        objtype = glObjType(nplex)
        
    if nplex <= 4 and glObjType(nplex) == objtype:
        GL.glBegin(objtype)
        if c is None or c.ndim == 1:       # no or single color 
            if c is not None:                      # single color
                glColor(c,alpha)
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
                    glColor(ci,alpha)
                    for j in range(nplex):
                        GL.glVertex3fv(xi[j])
            elif n.ndim == 2:
                for xi,ni,ci in zip(x,n,c):
                    glColor(ci,alpha)
                    GL.glNormal3fv(ni)
                    for j in range(nplex):
                        GL.glVertex3fv(xi[j])
            elif n.ndim == 3:
                for xi,ni,ci in zip(x,n,c):
                    glColor(ci,alpha)
                    for j in range(nplex):
                        GL.glNormal3fv(ni[j])
                        GL.glVertex3fv(xi[j])

        elif c.ndim == 3:
            if n is None:
                for xi,ci in zip(x.reshape((-1,3)),c.reshape((-1,3))):
                    glColor(ci,alpha)
                    GL.glVertex3fv(xi)
            elif n.ndim == 2:
                for xi,ni,ci in zip(x,n,c):
                    GL.glNormal3fv(ni)
                    for j in range(nplex):
                        glColor(ci[j],alpha)
                        GL.glVertex3fv(xi[j])
            elif n.ndim == 3:
                for xi,ni,ci in zip(x.reshape((-1,3)),n.reshape((-1,3)),c.reshape((-1,3))):
                    glColor(ci,alpha)
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
                    glColor(ci,alpha)
                    for j in range(nplex):
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()
            elif n.ndim == 2:
                for xi,ni,ci in zip(x,n,c):
                    GL.glBegin(objtype)
                    glColor(ci,alpha)
                    GL.glNormal3fv(ni)
                    for j in range(nplex):
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()
            elif n.ndim == 3:
                for xi,ni,ci in zip(x,n,c):
                    GL.glBegin(objtype)
                    glColor(ci,alpha)
                    for j in range(nplex):
                        GL.glNormal3fv(ni[j])
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()

        elif c.ndim == 3:
            if n is None:
                for xi,ci in zip(x,c):
                    GL.glBegin(objtype)
                    glColor(ci,alpha)
                    for j in range(nplex):
                        glColor(ci[j],alpha)
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()
            elif n.ndim == 2:
                for xi,ni,ci in zip(x,n,c):
                    GL.glBegin(objtype)
                    GL.glNormal3fv(ni)
                    for j in range(nplex):
                        glColor(ci[j],alpha)
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()
            elif n.ndim == 3:
                for xi,ni,ci in zip(x,n,c):
                    GL.glBegin(objtype)
                    for j in range(nplex):
                        glColor(ci[j],alpha)
                        GL.glNormal3fv(ni[j])
                        GL.glVertex3fv(xi[j])
                    GL.glEnd()

         

def pick_polygons(x,objtype):
    """Mimics draw_polygons for picking purposes.

    x : float (nel,nplex,3) : coordinates.
    objtype : GL Object type (-1 = auto)
    """
    nplex = x.shape[1]

    if objtype < 0:
        objtype = glObjType(nplex)

    for i,xi in enumerate(x): 
        GL.glPushName(i)
        GL.glBegin(objtype)
        for xij in xi:
            GL.glVertex3fv(xij)
        GL.glEnd()
        GL.glPopName()

        
def draw_polygon_elems(x,e,n,c,alpha,objtype):
    """Draw a collection of polygon elements.

    This function is like draw_polygons, but the vertices of the polygons
    are specified by a (coords,elems) tuple.
    x : float (npts,3) : coordinates
    e : int32 (nel,nplex) : element connectivity
    n : float (nel,3) or (nel,nplex,3) normals.
    c : float (nel,3) or (nel,nplex,3) colors
    alpha : float
    objtype : GL Object type (-1 = auto)
    """
    draw_polygons(x[e],n,c,alpha,objtype)
    

def pick_polygon_elems(x,e,objtype):
    """Mimics draw_polygon_elems for picking purposes.

    x : float (npts,3) : coordinates
    e : int32 (nel,nplex) : element connectivity
    objtype : GL Object type (-1 = auto)
    """
    pick_polygons(x[e],objtype)
    
### End

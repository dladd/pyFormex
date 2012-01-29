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
import pyformex as pf
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
    try:
        return [GL.GL_POINTS,GL.GL_LINES,GL.GL_TRIANGLES,GL.GL_QUADS][nplex-1]
    except:
        return GL.GL_POLYGON

                    
def draw_polygons(x,n,c,t,alpha,objtype):
    """Draw a collection of polygons.

    x : float (nel,nplex,3) : coordinates.
    n : float (nel,3) or (nel,nplex,3) : normals.
    c : float (nel,3) or (nel,nplex,3) : color(s)
    t : float (nplex,2) or (nel,nplex,2) : texture coords
    alpha : float
    objtype : GL Object type (-1 = auto)

    If nplex colors per element are given, and shading mode is flat,
    the last color will be used.
    """
    pf.debug("draw_tex_polygons",pf.DEBUG.DRAW)
    x = x.astype(float32)
    nelems,nplex = x.shape[:2]
    ndn = ndc = ndt = 0
    if n is not None:
        n = n.astype(float32)
        ndn = n.ndim
    if c is not None:
        c = c.astype(float32)
        ndc = c.ndim
    if t is not None:
        t = t.astype(float32)
        ndt = t.ndim
    if objtype < 0:
        objtype = glObjType(nplex)

    print nelems,nplex,ndn,ndc,ndt,objtype
    
    simple = nplex <= 4 and objtype == glObjType(nplex)
    if simple:
        GL.glBegin(objtype)
        if ndc == 1:
            glColor(c,alpha)
        for i in range(nelems):
            if ndc == 2:
                glColor(c[i],alpha)
            if ndn == 2:
                GL.glNormal3fv(n[i])
            for j in range(nplex):
                if ndn == 3:
                    GL.glNormal3fv(n[i,j])
                if ndc == 3:
                    glColor(c[i,j],alpha)
                if ndt == 2:
                    GL.glTexCoord2fv(t[j]) 
                elif ndt == 3:
                    GL.glTexCoord2fv(t[i,j]) 
                GL.glVertex3fv(x[i,j])
        GL.glEnd()

    else:
        if ndc == 1:
            glColor(c,alpha)
        for i in range(nelems):
            GL.glBegin(objtype)
            if ndc == 2:
                glColor(c[i],alpha)
            if ndn == 2:
                GL.glNormal3fv(n[i])
            for j in range(nplex):
                if ndn == 3:
                    GL.glNormal3fv(n[i,j])
                if ndc == 3:
                    glColor(c[i,j],alpha)
                if ndt == 2:
                    GL.glTexCoord2fv(t[j]) 
                elif ndt == 3:
                    GL.glTexCoord2fv(t[i,j]) 
                GL.glVertex3fv(x[i,j])
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

        
def draw_polygon_elems(x,e,n,c,t,alpha,texid,objtype):
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
    draw_polygons(x[e],n,c,t,alpha,texid,objtype)
    

def pick_polygon_elems(x,e,objtype):
    """Mimics draw_polygon_elems for picking purposes.

    x : float (npts,3) : coordinates
    e : int32 (nel,nplex) : element connectivity
    objtype : GL Object type (-1 = auto)
    """
    pick_polygons(x[e],objtype)
    
### End

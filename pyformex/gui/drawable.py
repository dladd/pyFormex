# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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
"""OpenGL drawing functions and base class for all drawable objects."""

import pyformex as pf

from OpenGL import GL,GLU

from colors import *
from numpy import *
from formex import *

import simple
import utils
import olist

from pyformex.lib import drawgl

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


### Some drawing functions ###############################################

def glColor(color,alpha=1.0):
    """Set the OpenGL color, possibly with transparency.

    color is a tuple of 3 real values.
    alpha is a single real value.
    All values are between 0.0 and 1.0
    """
    if color is not None:
        if alpha == 1.0:
            GL.glColor3fv(color)
        else:
            GL.glColor4fv(append(color,alpha)) 


#
# Though all three functions drawPoints, drawLines and drawPolygons
# call the same low level drawgl.draw_polygons function, we keep 3 separate
# functions on the higher level, because of special characteristics
# of nplex < 3:   no computation of normals, marksize (nplex=1)
#


# DRAWPOINTS should also be modified to accept an (x,e) model
# (Yes, it makes sense to create a Point mesh
def drawPoints(x,color=None,alpha=1.0,size=None):
    """Draw a collection of points with default or given size and color.

    x is a (npoints,3) shaped array of coordinates.
    If size (float) is given, it specifies the point size.
    If color is given it is an (npoints,3) array of RGB values.
    """
    if pf.options.safelib:
        x = x.astype(float32).reshape(-1,3)
        if color is not None:
            color = resize(color.astype(float32),x.shape)
    if size:
        GL.glPointSize(size)
    x = x.reshape(-1,1,3)
    drawgl.draw_polygons(x,None,color,alpha,-1)
    

def drawPolygons(x,e,mode,color=None,alpha=1.0,normals=None,objtype=-1):
    """Draw a collection of polygon elements.

    This function is like drawPolygons, but the vertices of the polygons
    are specified by:
    coords (npts,3) : the coordinates of the points
    elems (nels,nplex): the connectivity of nels polygons of plexitude nplex

    objtype sets the OpenGL drawing mode. The default (-1) will select the
    appropriate value depending on the plexitude of the elements:
    1: point, 2: line, 3: triangle, 4: quad, >4: polygon.
    The value can be set to GL.GL_LINE_LOOP to draw the element's circumference
    independent from the drawing mode.
    """
    pf.debug("drawPolygons")
    if e is None:
        nelems = x.shape[0]
    else:
        nelems = e.shape[0]
    n = None
    if mode.startswith('smooth') and objtype==-1:
        if normals is None:
            pf.debug("Computing normals")
            if mode == 'smooth_avg' and e is not None:
                n = interpolateNormals(x,e,treshold=pf.cfg['render/avgnormaltreshold'])
                mode = 'smooth'
            else:
                if e is None:
                    n = polygonNormals(x)
                else:
                    n = polygonNormals(x[e])
                pf.debug("NORMALS:%s" % str(n.shape))
        else:
            try:
                n = asarray(normals)
                if not (n.ndim in [2,3] and n.shape[0] == nelems and n.shape[-1] == 3):
                    raise
            except:
                raise ValueError,"""Invalid normals specified"""

    if pf.options.safelib:
        x = x.astype(float32)
        if e is not None:
            e = e.astype(int32)
        if n is not None:
            n = n.astype(float32)
        if color is not None:
            color = color.astype(float32)
            pf.debug("COLORS:%s" % str(color.shape))
            if (color.shape[0] != nelems or
                color.shape[-1] != 3):
                color = None
    if e is None:
        drawgl.draw_polygons(x,n,color,alpha,objtype)
    else:
        drawgl.draw_polygon_elems(x,e,n,color,alpha,objtype)


def drawPolyLines(x,e,color):
    """Draw the circumference of polygons."""
    pf.debug("drawPolyLines")
    drawPolygons(x,e,mode='wireframe',color=color,alpha=1.0,objtype=GL.GL_LINE_LOOP)


def drawLines(x,e,color):
    """Draw straight line segments."""
    pf.debug("drawLines")
    drawPolygons(x,e,mode='wireframe',color=color,alpha=1.0)


def drawBezier(x,color=None,objtype=GL.GL_LINE_STRIP,granularity=100):
    """Draw a collection of Bezier curves.

    x: (4,3,3) : control points
    color: (4,) or (4,4): colors
    """
    GL.glMap1f(GL.GL_MAP1_VERTEX_3,0.0,1.0,x)
    GL.glEnable(GL.GL_MAP1_VERTEX_3)
    if color is not None and color.shape == (4,4):
        GL.glMap1f(GL.GL_MAP1_COLOR_4,0.0,1.0,color)
        GL.glEnable(GL.GL_MAP1_COLOR_4)

    u = arange(granularity+1) / float(granularity)
    if color is not None and color.shape == (4,):
        GL.glColor4fv(color)
        color = None
        
    GL.glBegin(objtype)
    for ui in u:
        #  For multicolors, this will generate both a color and a vertex  
        GL.glEvalCoord1f(ui)
    GL.glEnd()

    GL.glDisable(GL.GL_MAP1_VERTEX_3)
    if color is not None:
        GL.glDisable(GL.GL_MAP1_COLOR_4)


def drawBezierPoints(x,color=None,granularity=100):
    drawBezier(x,color=None,objtype=GL.GL_POINTS,granularity=granularity)


def color_multiplex(color,nparts):
    """Multiplex a color array over nparts of the elements.

    This function will repeat the colors in an array a number of times
    so that all parts of the same element are colored the same.
    """
    s = list(color.shape)
    s[1:1] = [1]
    color = color.reshape(*s).repeat(nparts,axis=1)
    s[1] = nparts # THIS APPEARS NOT TO BE DOING ANYTHING ?
    return color.reshape(-1,3)


def draw_parts(x,e,mode,color=None,alpha=1.0):
    """Draw a collection of faces.

    (x,e) are one of:
    - x is a (nelems,nfaces,nplex,3) shaped coordinates and e is None,
    - x is a (ncoords,3) shaped coordinates and e is a (nelems,nfaces,nplex)
    connectivity array.
       
    Each of the nfaces sets of nplex points defines a polygon. 

    If color is given it is an (nel,3) array of RGB values. This function
    will multiplex the colors, so that n faces are drawn in the same color.
    This is e.g. convenient when drawing faces of a solid element.
    """
    if e is None:
        nfaces,nplex = x.shape[1:3]
        x = x.reshape(-1,nplex,3)
    else:
        nfaces,nplex = e.shape[1:3]
        e = e.reshape(-1,nplex)

    if color is not None:
        if color.ndim < 3:
            pf.debug("COLOR SHAPE BEFORE MULTIPLEXING %s" % str(color.shape))
            color = color_multiplex(color,nfaces)
            pf.debug("COLOR SHAPE AFTER  MULTIPLEXING %s" % str(color.shape))

    drawPolygons(x,e,mode,color,alpha)


def drawEdges(x,e,edges,color=None):
    """Draw the edges of a geometry.

    This function draws the edges of a geometry collection, usually of a higher
    dimensionality (i.c. a surface or a volume).
    The edges are identified by a constant indices into all element vertices.

    The geometry is specified by x or (x,e)
    The edges are specified by a list of lists. Each list defines a single
    edge of the solid, in local vertex numbers (0..nplex-1). 
    """
    pf.debug("drawEdges")
    fa = asarray(edges)
    if e is None:
        coords = x[:,fa,:]
        elems = None
    else:
        coords = x
        elems = e[:,fa]
    pf.debug("COORDS SHAPE: %s" % str(coords.shape))
    if elems is not None:
        pf.debug("ELEMS SHAPE: %s" % str(elems.shape))
    if color is not None and color.ndim==3:
        pf.debug("COLOR SHAPE BEFORE EXTRACTING: %s" % str(color.shape))
        # select the colors of the matching points
        color = color[:,fa,:]#.reshape((-1,)+color.shape[-2:])
        color = color.reshape((-1,)+color.shape[-2:])
        pf.debug("COLOR SHAPE AFTER EXTRACTING: %s" % str(color.shape))
    draw_parts(coords,elems,'wireframe',color,1.0)


def drawFaces(x,e,faces,mode,color=None,alpha=1.0):
    """Draw the faces of a geometry.

    This function draws the faces of a geometry collection, usually of a higher
    dimensionality (i.c. a volume).
    The faces are identified by a constant indices into all element vertices.

    The geometry is specified by x or (x,e)
    The faces are specified by a list of lists. Each list defines a single
    face of the solid, in local vertex numbers (0..nplex-1). The faces are
    sorted and collected according to their plexitude before drawing them. 
    """
    pf.debug("drawFaces")
    # We may have faces with different plexitudes!
    # We collect them according to plexitude.
    # But first convert to a list, so that we can call this function
    # with an array too (in case of a single plexitude)
    faces = list(faces)
    for fac in olist.collectOnLength(faces).itervalues():
        fa = asarray(fac)
        nplex = fa.shape[1]
        if e is None:
            coords = x[:,fa,:]
            elems = None
        else:
            coords = x
            elems = e[:,fa]
        pf.debug("COORDS SHAPE: %s" % str(coords.shape))
        if elems is not None:
            pf.debug("ELEMS SHAPE: %s" % str(elems.shape))
        if color is not None and color.ndim==3:
            pf.debug("COLOR SHAPE BEFORE EXTRACTING: %s" % str(color.shape))
            # select the colors of the matching points
            color = color[:,fa,:]
            color = color.reshape((-1,)+color.shape[-2:])
            pf.debug("COLOR SHAPE AFTER EXTRACTING: %s" % str(color.shape))
        draw_parts(coords,elems,mode,color,alpha)


def drawAtPoints(x,mark,color=None):
    """Draw a copy of a 3D actor mark at all points in x.

    x is a (npoints,3) shaped array of coordinates.
    mark is any 3D Actor having a display list attribute.
    If color is given it is an (npoints,3) array of RGB values. It only
    makes sense if the mark was created without color!
    """
    for i,xi in enumerate(x):
        if color is not None:
            GL.glColor3fv(color[i])
        GL.glPushMatrix()
        GL.glTranslatef(*xi)
        GL.glCallList(mark)
        GL.glPopMatrix()


def Shape(a):
    """Return the shape of an array or None"""
    try:
        return a.shape
    except:
        return None


# CANDIDATE FOR C LIBRARY
def average_close(a,tol=0.5):
    """Average values from an array according to some specification.

    The default is to have a direction that is nearly the same.
    a is a 2-dim array
    """
    if a.ndim != 2:
        raise ValueError,"array should be 2-dimensional!"
    n = normalize(a)
    nrow = a.shape[0]
    cnt = zeros(nrow,dtype=int32)
    while cnt.min() == 0:
        w = where(cnt==0)
        nw = n[w]
        wok = where(dotpr(nw[0],nw) >= tol)
        wi = w[0][wok[0]]
        cnt[wi] = len(wi)
        a[wi] = a[wi].sum(axis=0)
    return a,cnt


# CANDIDATE FOR C LIBRARY
def nodalSum2(val,elems,tol):
    """Compute the nodal sum of values defined on elements.

    val   : (nelems,nplex,nval) values at points of elements.
    elems : (nelems,nplex) nodal ids of points of elements.
    work  : a work space (unused) 

    The return value is a tuple of two arrays:
    res:
    cnt
    On return each value is replaced with the sum of values at that node.
    """
    nodes = unique(elems)
    for i in nodes:
        wi = where(elems==i)
        vi = val[wi]
        ai,ni = average_close(vi,tol=tol)
        ai /= ni.reshape(ai.shape[0],-1)
        val[wi] = ai

def nodalSum(val,elems,avg=False,return_all=True,direction_treshold=None):
    """Compute the nodal sum of values defined on elements.

    val is a (nelems,nplex,nval) array of values defined at points of elements.
    elems is a (nelems,nplex) array with nodal ids of all points of elements.

    The return value is a (nelems,nplex,nval) array where each value is
    replaced with the sum of its value at that node.
    If avg=True, the values are replaced with the average instead.
    (DOES NOT WORK YET)
    If return_all==True(default), returns an array with shape (nelems,nplex,3),
    else, returns an array with shape (maxnodenr+1,3). In the latter case,
    nodes not occurring in elems will have all zero values.

    If a direction_tolerance is specified and nval > 1, values will only be
    summed if their direction is close (projection of one onto the other is
    higher than the specified tolerance).
    """
    if val.ndim != 3:
        val.reshape(val.shape+(1,))
    if elems.shape != val.shape[:2]:
        raise RuntimeError,"shape of val and elems does not match"
    work = zeros((elems.max()+1,val.shape[2]))
    if pf.options.safelib:
        val = val.astype(float32)
        elems = elems.astype(int32)
        work = work.astype(float32)
    if val.shape[2] > 1 and direction_treshold is not None:
        nodalSum2(val,elems,direction_treshold)
    else:
        misc.nodalSum(val,elems,work,avg)
    if return_all:
        return val
    else:
        return work


def interpolateNormals(coords,elems,atNodes=False,treshold=None):
    """Interpolate normals in all points of elems.

    coords is a (ncoords,3) array of nodal coordinates.
    elems is an (nel,nplex) array of element connectivity.
    
    The default return value is an (nel,nplex,3) array with the averaged
    unit normals in all points of all elements.
    If atNodes == True, a more compact array with the unique averages
    at the nodes is returned.
    """
    n = polygonNormals(coords[elems])
    n = nodalSum(n,elems,return_all=not atNodes,direction_treshold=treshold)
    return normalize(n)


def drawQuadraticCurves(x,color=None,n=8):
    """Draw a collection of curves.

    x is a (nlines,3,3) shaped array of coordinates.
    For each member a quadratic curve through its points is drawn.
    The quadratic curve is approximated with 2*n straight segments.

    If color is given it is an (nlines,3) array of RGB values.
    """
    H = simple.quadraticCurve(identity(3),n)
    for i in range(x.shape[0]):
        if color is not None:
            GL.glColor3fv(color[i])
        P = dot(H,x[i])
        pf.debug("P.shape=%s"%str(P.shape))
        GL.glBegin(GL.GL_LINE_STRIP)
        for p in P:
            GL.glVertex3fv(p)
        GL.glEnd()


def drawNurbsCurves(x,color=None):
    """Draw a collection of curves.

    x is an (nlines,4,3) or (nlines,3,3) shaped array of coordinates.

    If color is given it is an (nlines,3) array of RGB values.
    """
    nurb = GLU.gluNewNurbsRenderer()
    if not nurb:
        raise RuntimeError,"Could not create a new NURBS renderer"
        return
    
    if x.shape[1] == 4:
        knots = array([0.,0.,0.,0.,1.0,1.0,1.0,1.0])
    if x.shape[1] == 3:
        knots = array([0.,0.,0.,1.0,1.0,1.0])
    for i,xi in enumerate(x):
        if color is not None:
            GL.glColor3fv(color[i])
        GLU.gluBeginCurve(nurb)
        GLU.gluNurbsCurve(nurb,knots,xi,GL.GL_MAP1_VERTEX_3)
        GLU.gluEndCurve(nurb)

    
def drawCube(s,color=[red,cyan,green,magenta,blue,yellow]):
    """Draws a centered cube with side 2*s and colored faces.

    Colors are specified in the order [FRONT,BACK,RIGHT,LEFT,TOP,BOTTOM].
    """
    vertices = [[s,s,s],[-s,s,s],[-s,-s,s],[s,-s,s],[s,s,-s],[-s,s,-s],[-s,-s,-s],[s,-s,-s]]
    planes = [[0,1,2,3],[4,5,6,7],[0,3,7,4],[1,2,6,5],[0,1,5,4],[3,2,6,7]]
    GL.glBegin(GL.GL_QUADS)
    for i in range(6):
        #glNormal3d(0,1,0);
        GL.glColor(*color[i])
        for j in planes[i]:
            GL.glVertex3f(*vertices[j])
    GL.glEnd()


def drawSphere(s,color=cyan,ndiv=8):
    """Draws a centered sphere with radius s in given color."""
    quad = GLU.gluNewQuadric()
    GLU.gluQuadricNormals(quad, GLU.GLU_SMOOTH)
    GL.glColor(*color)
    GLU.gluSphere(quad,s,ndiv,ndiv)


def drawGridLines(x0,x1,nx):
    """Draw a 3D rectangular grid of lines.
        
    A grid of lines parallel to the axes is drawn in the domain bounded
    by the rectangular box [x0,x1]. The grid has nx divisions in the axis
    directions, thus lines will be drawn at nx[i]+1 positions in direction i.
    If nx[i] == 0, lines are only drawn for the initial coordinate x0.
    Thus nx=(0,2,3) results in a grid of 3x4 lines in the plane // (y,z) at
    coordinate x=x0[0].
    """
    x0 = asarray(x0)
    x1 = asarray(x1)
    nx = asarray(nx)

    for i in range(3):
        if nx[i] > 0:
            axes = (asarray([1,2]) + i) % 3
            base = simple.regularGrid(x0[axes],x1[axes],nx[axes]).reshape((-1,2))
            x = zeros((base.shape[0],2,3))
            x[:,0,axes] = base
            x[:,1,axes] = base
            x[:,0,i] = x0[i]
            x[:,1,i] = x1[i]
            GL.glBegin(GL.GL_LINES)
            for p in x.reshape((-1,3)):
                GL.glVertex3fv(p)
            GL.glEnd()

    
def drawGridPlanes(x0,x1,nx):
    """Draw a 3D rectangular grid of planes.
        
    A grid of planes parallel to the axes is drawn in the domain bounded
    by the rectangular box [x0,x1]. The grid has nx divisions in the axis
    directions, thus planes will be drawn at nx[i]+1 positions in direction i.
    If nx[i] == 0, planes are only drawn for the initial coordinate x0.
    Thus nx=(0,2,3) results in a grid of 3x4 planes // x and
    one plane // (y,z) at coordinate x=x0[0].
    """
    x0 = asarray(x0)
    x1 = asarray(x1)
    nx = asarray(nx)

    for i in range(3):
        axes = (asarray([1,2]) + i) % 3
        if all(nx[axes] > 0):
            j,k = axes
            base = simple.regularGrid(x0[i],x1[i],nx[i]).ravel()
            x = zeros((base.shape[0],4,3))
            corners = array([x0[axes],[x1[j],x0[k]],x1[axes],[x0[j],x1[k]]])
            for j in range(4):
                x[:,j,i] = base
            x[:,:,axes] = corners
            GL.glBegin(GL.GL_QUADS)
            for p in x.reshape((-1,3)):
                #print p
                GL.glVertex3fv(p)
            GL.glEnd()


######################## Picking functions ########################

def pickPolygons(x,e=None,objtype=-1):
    """Mimics drawPolygons for picking purposes."""
    if pf.options.safelib:
        x = x.astype(float32)
        if e is not None:
            e = e.astype(int32)
    if e is None:
        drawgl.pick_polygons(x,objtype)
    else:
        drawgl.pick_polygon_elems(x,e,objtype)


def pickPolygonEdges(x,e,edg):
    warning("pickPolygonEdges IS NOT IMPLEMENTED YET!")


def pickPoints(x):
    x = x.reshape((-1,1,3))
    pickPolygons(x)


### Settings ###############################################
#
# These are not intended for users but to sanitize user input
#

def saneLineWidth(linewidth):
    """Return a sane value for the line width.

    A sane value is one that will be usable by the draw method.
    It can be either of the following:

    - a float value indicating the line width to be set by draw()
    - None: indicating that the default line width is to be used

    The line width is used in wireframe mode if plex > 1
    and in rendering mode if plex==2.
    """
    if linewidth is not None:
        linewidth = float(linewidth)
    return linewidth


def saneLineStipple(stipple):
    """Return a sane line stipple tuple.

    A line stipple tuple is a tuple (factor,pattern) where
    pattern defines which pixels are on or off (maximum 16 bits),
    factor is a multiplier for each bit.   
    """
    try:
        stipple = map(int,stipple)
    except:
        stipple = None
    return stipple
    
   
def saneColor(color=None):
    """Return a sane color array derived from the input color.

    A sane color is one that will be usable by the draw method.
    The input value of color can be either of the following:

    - None: indicates that the default color will be used,
    - a single color value in a format accepted by colors.GLColor,
    - a tuple or list of such colors,
    - an (3,) shaped array of RGB values, ranging from 0.0 to 1.0,
    - an (n,3) shaped array of RGB values,
    - an (n,) shaped array of integer color indices.

    The return value is one of the following:
    - None, indicating no color (current color will be used),
    - a float array with shape (3,), indicating a single color, 
    - a float array with shape (n,3), holding a collection of colors,
    - an integer array with shape (n,), holding color index values.

    !! Note that a single color can not be specified as integer RGB values.
    A single list of integers will be interpreted as a color index !
    Turning the single color into a list with one item will work though.
    [[ 0, 0, 255 ]] will be the same as [ 'blue' ], while
    [ 0,0,255 ] would be a color index with 3 values. 
    """
    if color is None:
        # no color: use canvas color
        return None

    # detect color index
    try:
        c = asarray(color)
        if c.dtype.kind == 'i':
            # We have a color index
            return c
    except:
        pass

    # not a color index: it must be colors
    try:
        color = GLColor(color)
    except ValueError:

        try:
            color = map(GLColor,color)
        except ValueError:
            pass

    # Convert to array
    try:
        color = asarray(color).squeeze()
        if color.dtype.kind == 'f' and color.shape[-1] == 3:
            # Looks like we have a sane color array
            return color.astype(float32)
    except:
        pass

    return None


def saneColorArray(color,shape):
    """Makes sure the shape of the color array is compatible with shape.

    shape is an (nelems,nplex) tuple
    A compatible color.shape is equal to shape or has either or both of its
    dimensions equal to 1.
    Compatibility is enforced in the following way:
    - if color.shape[1] != nplex and color.shape[1] != 1: take out first
      plane in direction 1
    - if color.shape[0] != nelems and color.shape[0] != 1: repeat the plane
      in direction 0 nelems times
    """
    color = asarray(color)
    if color.ndim == 1:
        return color
    if color.ndim == 3:
        if color.shape[1] > 1 and color.shape[1] != shape[1]:
            color = color[:,0]
    if color.shape[0] > 1 and color.shape[0] != shape[0]:
        color = resize(color,(shape[0],color.shape[1]))
    return color


def saneColorSet(color=None,colormap=None,shape=(1,),canvas=None):
    """Return a sane set of colors.

    A sane set of colors is one that guarantees correct use by the
    draw functions. This means either
    - no color (None)
    - a single color
    - at least as many colors as the shape argument specifies
    - a color index and a color map with enough colors to satisfy the index.
    The return value is a tuple color,colormap. colormap will be None,
    unless color is an integer array, meaning a color index.
    """
    if type(shape) == int:  # make sure we get a tuple
        shape = (shape,)
    color = saneColor(color)
    if color is not None:
        if color.dtype.kind == 'i':
            ncolors = color.max()+1
            if colormap is None:
                if canvas:
                    colormap = canvas.settings.colormap
                else:
                    colormap = pf.cfg['canvas/colormap']
            colormap = saneColor(colormap)
            colormap = saneColorArray(colormap,(ncolors,))
        else:
            color = saneColorArray(color,shape)
            colormap = None

    return color,colormap


### Drawable Objects ###############################################

class Drawable(object):
    """A Drawable is anything that can be drawn on the OpenGL Canvas.

    This defines the interface for all drawbale objects, but does not
    implement any drawable objects.
    Drawable objects should be instantiated from the derived classes.
    Currently, we have the following derived classes:
      Actor: a 3-D object positioned and oriented in the 3D scene. Defined
             in actors.py.
      Mark: an object positioned in 3D scene but not undergoing the camera
             axis rotations and translations. It will always appear the same
             to the viewer, but will move over the screen according to its
             3D position. Defined in marks.py.
      Decor: an object drawn in 2D viewport coordinates. It will unchangeably
             stick on the viewport until removed. Defined in decors.py.
    
    A Drawable subclass should minimally reimplement the following methods:
      bbox(): return the actors bounding box.
      nelems(): return the number of elements of the actor.
      drawGL(mode): to draw the object. Takes a mode argument so the
        drawing function can act differently depending on the rendering mode.
        There are currently 5 modes:
           wireframe, flat, smooth, flatwire, smoothwire.
      drawGL should only contain OpenGL calls that are allowed inside a display
      list. This may include calling the display list of another actor but NOT
      creating a new display list.
    """
    
    def __init__(self):
        self.trans = False
        self.list = None
        ## self.atype = 'unknown'

    def drawGL(self,**kargs):
        """Perform the OpenGL drawing functions to display the actor."""
        raise NotImplementedError

    def pickGL(self,**kargs):
        """Mimick the OpenGL drawing functions to pick (from) the actor."""
        pass

    def draw(self,**kargs):
        #print "draw with args %s" % kargs
        if self.list is None:
            self.create_list(**kargs)
        self.use_list()

    def redraw(self,**kargs):
        self.delete_list()
        self.draw(**kargs)

    def use_list(self):
        if self.list:
            GL.glCallList(self.list)

    def create_list(self,**kargs):
        self.list = GL.glGenLists(1)
        GL.glNewList(self.list,GL.GL_COMPILE)
        ok = False
        try:
            self.drawGL(**kargs)
            ok = True
        finally:
            if not ok:
                pf.debug("Error while creating a display list")
            GL.glEndList()

    def delete_list(self):
        if self.list:
            GL.glDeleteLists(self.list,1)
        self.list = None
    
    def setLineWidth(self,linewidth):
        """Set the linewidth of the Actor."""
        self.linewidth = saneLineWidth(linewidth)
    
    def setLineStipple(self,linestipple):
        """Set the linewidth of the Actor."""
        self.linestipple = saneLineStipple(linestipple)

    def setColor(self,color=None,colormap=None,ncolors=1):
        """Set the color of the Drawable."""
        self.color,self.colormap = saneColorSet(color,colormap,shape=(ncolors,))

### End

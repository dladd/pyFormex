# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
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
"""OpenGL drawing functions and base class for all drawable objects."""

import pyformex as GD

from OpenGL import GL,GLU

from colors import *
from numpy import *
from formex import *

import simple
import utils
import olist

from pyformex.lib import drawgl

### Some drawing functions ###############################################

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


#
# Though all three functions drawPoints, drawLines and drawPolygons
# call the same low level drawgl.draw_polygons function, we keep 3 separate
# functions on the higher level, because of special characteristics
# of nplex < 3:   no computation of normals, marksize (nplex=1)
#

def drawPoints(x,color=None,alpha=1.0,size=None):
    """Draw a collection of points with default or given size and color.

    x is a (npoints,3) shaped array of coordinates.
    If size (float) is given, it specifies the point size.
    If color is given it is an (npoints,3) array of RGB values.
    """
    if GD.options.safelib:
        x = x.astype(float32).reshape(-1,3)
        if color is not None:
            color = resize(color.astype(float32),x.shape)
    if size:
        GL.glPointSize(size)
    x = x.reshape(-1,1,3)
    drawgl.draw_polygons(x,None,color,alpha)
    

def drawLines(x,color=None,alpha=1.0):
    """Draw a collection of lines.

    x is a (nlines,2,3) shaped array of coordinates.

    If color is given it is an (nlines,3), (nlines,1,3) or (nlines,2,3)
    array of RGB values.
    If two colors are given, make sure that smooth shading is on,
    or the color rendering will be flat with the second color.
    """
    if GD.options.safelib:
        x = x.astype(float32)
        if color is not None:
            color = color.astype(float32)
            if (color.shape[0] != x.shape[0] or
                color.shape[-1] != 3):
                color = None
    drawgl.draw_polygons(x,None,color,alpha)


def drawPolygons(x,e,mode,color=None,alpha=1.0,normals=None):
    """Draw a collection of polygon elements.

    This function is like drawPolygons, but the vertices of the polygons
    are specified by:
    coords (npts,3) : the coordinates of the points
    elems (nels,nplex): the connectivity of nels polygons of plexitude nplex
    """
    if e is None:
        nelems = x.shape[0]
    else:
        nelems = e.shape[0]
    n = None
    if mode.startswith('smooth'):
        if normals is None:
            if mode == 'smooth-avg' and e is not None:
                n = interpolateNormals(x,e,treshold=GD.cfg['render/avgnormaltreshold'])
                mode = 'smooth'
            else:
                if e is None:
                    n = polygonNormals(x)
                else:
                    n = polygonNormals(x[e])
                GD.debug("NORMALS:%s" % str(n.shape))
        else:
            try:
                n = asarray(normals)
                if not (n.ndim in [2,3] and n.shape[0] == nelems and n.shape[-1] == 3):
                    raise
            except:
                raise ValueError,"""Invalid normals specified"""

    if GD.options.safelib:
        x = x.astype(float32)
        if e is not None:
            e = e.astype(int32)
        if n is not None:
            n = n.astype(float32)
        if color is not None:
            color = color.astype(float32)
            #GD.debug(color.shape)
            if (color.shape[0] != nelems or
                color.shape[-1] != 3):
                color = None
            GD.debug("COLORS:%s" % str(color.shape))

    if e is None:
        drawgl.draw_polygons(x,n,color,alpha)
    else:
        drawgl.draw_polygon_elems(x,e,n,color,alpha)


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
    nodes = unique1d(elems)
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
    if GD.options.safelib:
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
    

def drawLineElems(x,elems,color=None,alpha=1.0):
    """Draw a collection of lines.

    This is the same as drawLines, except that the lines are defined
    by an array of points and a connection table.
    x is a (ncoords,3) coordinate array.
    elems is a (nlines,2) integer array of connected node numbers.

    If color is given it is an (nlines,3) array of RGB values.
    """
    drawLines(x[elems],color,alpha)


def drawPolyLineElems(x,elems,color=None,alpha=1.0):
    nplex = elems.shape[1]
    verts = range(nplex)
    lines = column_stack([verts,roll(verts,-1)])
    els = elems[:,lines].reshape(-1,2)
    drawgl.draw_polygon_elems(x,els,None,None,alpha)
       

def drawEdges(x,color=None,alpha=1.0):
    """Draw a collection of edges.

    x is a (nel,2*n,3) shaped array of coordinates. Each of the n pairs
    define a line segment. 

    If color is given it is an (nel,3) array of RGB values.
    """
    n = x.shape[1] / 2
    x = x.reshape(-1,2,3)
    if color is not None:
        s = list(color.shape)
        s[1:1] = [1]
        color = color.reshape(*s).repeat(n,axis=1)
        s[1] = n
        color = color.reshape(-1,3)
    drawLines(x,color,alpha)


def drawEdgeElems(x,edges,color=None):
    """Draw a collection of edges.

    This function is like drawEdges, but the coordinates of the edges are
    specified by:
    x (nel,nplex) : the coordinates of solid elements
    edges (nedges,2): the definition of nedges edges of the solid,
      each with plexitude 2. Each line of edges defines a single
      edge of the solid, in local vertex numbers (0..nplex-1)
    """
    drawEdges(x[:,asarray(edges).ravel(),:],color)


def draw_faces(x,e,nplex,mode,color=None,alpha=1.0):
    """Draw a collection of faces.

    (x,e) are one of:
       x is a (nelems,nplex*nfaces,3) shaped coordinates and e is None,
       x is a (ncoords,3) shaped coordinates and e is a (nelems,nplex*nfaces)
       connectivity
       
    Each of the nfaces sets of nplex points defines a polygon. 

    If color is given it is an (nel,3) array of RGB values. This function
    will multiplex the colors, so that n faces are drawn in the same color.
    This is e.g. convenient when drawing faces of a solid element.
    """
    if e is None:
        nelpts = x.shape[1]
    else:
        nelpts = e.shape[1]
    nfaces = nelpts / nplex
    if e is None:
        x = x.reshape(-1,nplex,3)
    else:
        e = e.reshape(-1,nplex)
        
    if color is not None:
        # multiply element color
        s = list(color.shape)
        s[1:1] = [1]
        color = color.reshape(*s).repeat(nfaces,axis=1)
        s[1] = nfaces
        color = color.reshape(-1,3)
    drawPolygons(x,e,mode,color,alpha)


def drawFaces(x,faces,mode,color=None,alpha=1.0):
    """Draw a collection of faces.

    This function is like draw_faces, but the coordinates of the faces are
    specified by:
    x (nel,nplex) : the coordinates of solid elements
    faces (nfaces,fplex): the definition of nfaces faces of the solid,
      each with plexitude fplex. Each line of faces defines a single
      face of the solid, in local vertex numbers (0..nplex-1)
    """
    # We may have faces with different plexitudes!
    for fac in olist.collectOnLength(faces).itervalues():
        fa = asarray(fac)
        draw_faces(x[:,fa.ravel(),:],None,fa.shape[1],mode,color,alpha)


def drawPolyLines(x,c=None,close=True):
    """Draw a collection of polylines, closed or not.

    x : float (npoly,n,3) : coordinates.
    c : float (npoly,3) or (npoly,n,3) : color(s) at vertices
    If rendering is flat, the last color will  be used for each segment.
    """
    if close:
        glmode = GL.GL_LINE_LOOP
    else:
        glmode = GL.GL_LINE_STRIP
        
    if c is None:
        for i in range(x.shape[0]):
            GL.glBegin(glmode)
            for j in range(x.shape[1]):
                GL.glVertex3fv(x[i][j])
            GL.glEnd()

    elif c.ndim == 2:
        for i in range(x.shape[0]):
            GL.glBegin(glmode)
            GL.glColor3fv(c[i])
            for j in range(x.shape[1]):
                GL.glVertex3fv(x[i][j])
            GL.glEnd()

    elif c.ndim == 3:
        for i in range(x.shape[0]):
            GL.glBegin(glmode)
            for j in range(x.shape[1]):
                GL.glColor3fv(c[i][j])
                GL.glVertex3fv(x[i][j])
            GL.glEnd()


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
        GD,debug("P.shape=%s"%str(P.shape))
        GL.glBegin(GL.GL_LINE_STRIP)
        for p in P:
            GL.glVertex3fv(p)
        GL.glEnd()


def drawNurbsCurves(x,color=None):
    """Draw a collection of curves.

    x is a (nlines,3,3) shaped array of coordinates.

    If color is given it is an (nlines,3) array of RGB values.
    """
    nurb = GLU.gluNewNurbsRenderer()
##    nkots = 7
##    knots = arange(nkots+1) / float(nkots)
    if x.shape[1] == 4:
        knots = array([0.,0.,0.,0.,1.0,1.0,1.0,1.0])
    if x.shape[1] == 3:
        knots = array([0.,0.,0.,1.0,1.0,1.0])
    if not nurb:
        return
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
                GL.glVertex3fv(p)
            GL.glEnd()


######################## Draw mimicking for picking ########################



def pickPolygons(x):
    """Mimics drawing polygons for picking purposes."""
    if GD.options.safelib:
        x = x.astype(float32)
    drawgl.pick_polygons(x)


def pickPoints(x):
    x = x.reshape((-1,1,3))
    pickPolygons(x)


def pickPolygonElems(x,e):
    pickPolygons(x[e])


def pickPolygonEdges(x,e):
    """Basic element picking function."""
    pickPolygons(x[e])


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
        # no color: use canvas fgcolor
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

    A compatible shape is equal to shape or has either or both of its
    dimensions equal to 1.
    Compatibility is enforced in the following way:
    - if shape[1] != np and shape[1] != 1: take out first plane in direction 1
    - if shape[0] != ne and shape[0] != 1: repeat plane in direction 0 ne times
    """
    if color.ndim == 1:
        return color
    if color.ndim == 3:
        if color.shape[1] > 1 and color.shape[1] != shape[1]:
            color = color[:,0]
    if color.shape[0] > 1 and color.shape[0] != shape[0]:
        color = resize(color,(shape[0],color.shape[1]))
    return color


def saneColorSet(color=None,colormap=None,shape=(1,)):
    """Return a sane set of colors.

    A sane set of colors is one that guarantees correct use by the
    draw functions. This means either
    - no color (None)
    - a single color
    - at least as many colors as the shape ncolors specifies
    - a color index and a color map with enough colors to satisfy the index.
    The return value is a tuple color,colormap. colormap will return
    unchanged, unless color is an integer array, meaning a color index.
    """
    if type(shape) == int:  # make sure we get a tuple
        shape = (shape,)
    color = saneColor(color)
    if color is not None:
        if color.dtype.kind == 'i':
            ncolors = color.max()+1
            if colormap is None:
                colormap = GD.canvas.settings.propcolors
            colormap = saneColor(colormap)
            colormap = saneColorArray(colormap,(ncolors,))
        else:
            color = saneColorArray(color,shape)

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
        self.atype = 'unknown'

    def drawGL(self,**kargs):
        """Perform the OpenGL drawing functions to display the actor."""
        raise NotImplementedError

    def pickGL(self,**kargs):
        """Mimick the OpenGL drawing functions to pick (from) the actor."""
        pass

    def draw(self,**kargs):
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
                GD.debug("Error while creating a display list")
            GL.glEndList()

    def delete_list(self):
        if self.list:
            GL.glDeleteLists(self.list,1)
        self.list = None
    
    def setLineWidth(self,linewidth):
        """Set the linewidth of the Actor."""
        self.linewidth = saneLineWidth(linewidth)

    def setColor(self,color=None,colormap=None,ncolors=1):
        """Set the color of the Drawable."""
        self.color,self.colormap = saneColorSet(color,colormap)


### End

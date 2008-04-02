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
"""OpenGL drawing functions and base class for all drawable objects."""

import globaldata as GD

from OpenGL import GL,GLU

from colors import *
from numpy import *
from formex import *

import simple

# Loading the low level drawing library
if GD.options.uselib is None:
    GD.options.uselib = True

if GD.options.uselib:
    try:
        import lib.drawgl as D
        GD.debug("Succesfully loaded the pyFormex compiled library")
    except ImportError:
        GD.debug("Error while loading the pyFormex compiled library")
        GD.debug("Reverting to scripted versions")
        GD.options.uselib = False
        GD.options.safelib = False
        
if not GD.options.uselib:
    GD.debug("Using the (slower) Python drawing functions")
    import drawgl as D


def rotMatrix(v,n=3):
    """Create a rotation matrix that rotates axis 0 to the given vector.

    Return either a 3x3(default) or 4x4(if n==4) rotation matrix.
    """
    if n != 4:
        n = 3
    #v = array(v,dtype=float64)
    vl = length(v)
    if vl == 0.0:
        return identity(n)
    v /= vl
    w = cross([0.,0.,1.],v)
    wl = length(w)
    if wl == 0.0:
        w = cross(v,[0,1,0])
        wl = length(w)
    w /= wl
    x = cross(v,w)
    x /= length(x)
    m = row_stack([v,w,x])
    if n == 3:
        return m
    else:
        a = identity(4)
        a[0:3,0:3] = m
        return a

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


def drawPoints(x,size=None,color=None):
    """Draw a collection of points with default or given size and color.

    x is a (npoints,3) shaped array of coordinates.
    If size (float) is given, it specifies the point size.
    If color is given it is an (npoints,3) array of RGB values.
    """
    x = x.reshape(-1,3)
    if color is not None:
        color = resize(color,x.shape)
    if size:
        GL.glPointSize(size)
    GL.glBegin(GL.GL_POINTS)
    for i,xi in enumerate(x):
        if color is not None:
            GL.glColor3fv(color[i])
        GL.glVertex3fv(xi)
    GL.glEnd()


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


def drawLines(x,color=None):
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
    D.drawLines(x,color)


def drawTriangles(x,mode,color=None,alpha=1.0):
    """Draw a collection of triangles.

    x is a (ntri,3*n,3) shaped array of coordinates.
    Each row contains n triangles drawn with the same color.

    If color is given it is an (npoly,3) array of RGB values.

    mode is either 'flat' or 'smooth' : in 'smooth' mode the normals
    for the lighting are calculated and set
    """
    x = x.reshape(-1,3,3)
    n = None
    if mode.startswith('smooth'):
        n = vectorPairNormals(x[:,1] - x[:,0], x[:,2] - x[:,1])
    if GD.options.safelib:
        x = x.astype(float32)
        if n is not None:
            n = n.astype(float32)
        if color is not None:
            color = color.astype(float32)
            if (color.shape[0] != x.shape[0] or
                color.shape[-1] != 3):
                color = None
    D.drawPolygons(x,n,color,alpha)


def drawPolygons(x,mode,color=None,alpha=1.0):
    """Draw a collection of polygons.

    mode is either 'flat' or 'smooth' : in 'smooth' mode the normals
    for the lighting are calculated and set
    """
    #print "DrawPolygons"
    #print "coords: %s" % str(x.shape)
    if color is not None:
        print "colors: %s" % str(color.shape)
    n = None
    if mode.startswith('smooth'):
        ni = arange(x.shape[1])
        nj = roll(ni,1)
        nk = roll(ni,-1)
        v1 = x-x[:,nj]
        v2 = x[:,nk]-x
        n = vectorPairNormals(v1.reshape(-1,3),v2.reshape(-1,3)).reshape(x.shape)
        #print "normals: %s" % str(n.shape)

    if GD.options.safelib:
        x = x.astype(float32)
        if n is not None:
            n = n.astype(float32)
        if color is not None:
            color = color.astype(float32)
            if (color.shape[0] != x.shape[0] or
                color.shape[-1] != 3):
                color = None
    #print n
    D.drawPolygons(x,n,color,alpha)


def drawLineElems(x,elems,color=None):
    """Draw a collection of lines.

    This is the same as drawLines, except that the lines are defined
    by an array of points and a connection table.
    x is a (ncoords,3) coordinate array.
    elems is a (nlines,2) integer array of connected node numbers.

    If color is given it is an (nlines,3) array of RGB values.
    """
    drawLines(x[elems],color)
 

def drawTriangleElems(x,elems,mode,color=None,alpha=1.0):
    drawTriangles(x[elems],mode,color,alpha)
       

def drawEdges(x,color=None):
    """Draw a collection of edges.

    x is a (nel,2*n,3) shaped array of coordinates. Each of the n pairs
    define a line segment. 

    If color is given it is an (nel,3) array of RGB values.
    """
    n = x.shape[1] / 2
    x = x.reshape(-1,2,3)
    if color is not None:
        s = list(color.shape)
        s[1:1] = 1
        color = color.reshape(*s).repeat(n,axis=1)
        s[1] = n
        color = color.reshape(*s)
    drawLines(x,color)


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
    
## def drawPolygons(x,mode,color=None):
##     """Draw a collection of polygones.

##     x is a (npoly,n,3) shaped array of coordinates.
##     Each row contains n triangles drawn with the same color.

##     If color is given it is an (npoly,3) array of RGB values.

##     mode is either 'flat' or 'smooth' : in 'smooth' mode the normals
##     for the lighting are calculated and set

##     REM: THE SMOOTH MODE HAS NOT BEEN IMPLEMENTED YET!
##     """
##     for i,xi in enumerate(x):
##         if color is not None:
##             GL.glColor3fv(color[i])
##         GL.glBegin(GL.GL_POLYGON)
##         for xij in xi:
##             GL.glVertex3fv(xij)
##         GL.glEnd()


## def drawPolygonColor(x,mode,color,alpha=1.0,normals=False):
##     """Draw a collection of polygones with smooth color shading.

##     x is a (ntri,nplex,3) shaped array of coordinates of the vertices.
##     nplex should be >= 3.
##     color is a (ntri,nplex,3) shaped array of color values at the vertices.
##     """
##     #if mode == 'smooth':
##     #    normal = vectorPairNormals(x[:,1] - x[:,0], x[:,2] - x[:,1])
##     if x.shape != color.shape:
##         raise RuntimeError,"Shape of x and color should be the same"
##     nplex = x.shape[:2]
##     if nplex == 3:
##         GL.glBegin(GL.GL_TRIANGLES)
##     elif nplex == 4:
##         GL.glBegin(GL.GL_QUADS)
##     else:
##         GL.glBegin(GL.GL_POLYGON)
##     x = x.reshape((-1,3))
##     color = color.reshape((-1,3))
##     for i in range(x.shape[0]):
##         glColor(color[i][j],alpha)
##         GL.glVertex3fv(x[i][j])
##     GL.glEnd()


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

    
def drawQuadrilaterals(x,mode,color=None):
    """Draw a collection of quadrilaterals.

    x is a (nquad,4*n,3) shaped array of coordinates.
    Each row contains n quads drawn with the same color.

    If color is given it is an (npoly,3) array of RGB values.

    mode is either 'flat' or 'smooth' : in 'smooth' mode the normals
    for the lighting are calculated and set
    """
    nplex = x.shape[1]
    if mode == 'smooth':
        edge = [ x[:,i,:] - x[:,i-1,:] for i in range(nplex) ]
        normal = [ vectorPairNormals(edge[i],edge[(i+1) % nplex]) for i in range(nplex) ]
    GL.glBegin(GL.GL_QUADS)
    for i in range(x.shape[0]):
        if color is not None:
            GL.glColor3fv(color[i])
        for j in range(nplex):
            if mode == 'smooth':
                GL.glNormal3fv(normal[j][i])
            GL.glVertex3fv(x[i][j])
    GL.glEnd()

    
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


def pickPoints(x):
    x = x.reshape((-1,3))
    for i,xi in enumerate(x):
        GL.glPushName(i)
        GL.glBegin(GL.GL_POINTS)
        GL.glVertex3fv(xi)
        GL.glEnd()
        GL.glPopName()


def pickPolygons(x):
    """Basic element picking function."""
    for i,xi in enumerate(x): 
        GL.glPushName(i)
        GL.glBegin(GL.GL_POLYGON)
        for xij in xi:
            GL.glVertex3fv(xij)
        GL.glEnd()
        GL.glPopName()


def pickPolygonElems(x,e):
    """Basic element picking function."""
    for i,ei in enumerate(e): 
        GL.glPushName(i)
        GL.glBegin(GL.GL_POLYGON)
        for eij in ei:
            GL.glVertex3fv(x[eij])
        GL.glEnd()
        GL.glPopName()


def pickPolygonEdges(x,e):
    """Basic element picking function."""
    for i,ei in enumerate(e): 
        GL.glPushName(i)
        GL.glBegin(GL.GL_LINES)
        for eij in ei:
            GL.glVertex3fv(x[eij])
        GL.glEnd()
        GL.glPopName()


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
        if c.dtype.kind == 'i' and c.ndim == 1:
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

#NOTE: we uncommented the default bbox and nelems methods,
#because they mask the corresponding Formex/Surface methods in the actors       

##     def bbox(self):
##         return array([[0.0,0.0,0.0],[1.0,1.0,1.0]])
        
##     def nelems(self):
##         return 1
        
        
    def atype(self):
        return 'unknown'

    def drawGL(self,**kargs):
        """Perform the OpenGL drawing functions to display the actor."""
        raise NotImplementedError

    def pickGL(self,**kargs):
        """Mimick the OpenGL drawing functions to pick (from) the actor."""
        pass

    def draw(self,**kargs):
        if self.list is None:
            self.create_list(**kargs)
        GL.glCallList(self.list)

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

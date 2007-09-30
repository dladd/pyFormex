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
"""OpenGL actors for populating the 3D scene."""

import globaldata as GD

from OpenGL import GL,GLU

from colors import *
from formex import *

import simple
from plugins import elements
from plugins.surface import Surface


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

    If color is given it is an (nlines,3) array of RGB values.
    """
    GL.glBegin(GL.GL_LINES)
    for i,xi in enumerate(x):
        if color is not None:
            GL.glColor3fv(color[i])
        GL.glVertex3fv(xi[0])
        GL.glVertex3fv(xi[1])
    GL.glEnd()


def drawLineElems(x,elems,color=None):
    """Draw a collection of lines.

    This is the same as drawLines, except that the lines are defined
    by an array of points and a connection table.
    x is a (ncoords,3) coordinate array.
    elems is a (nlines,2) integer array of connected node numbers.

    If color is given it is an (nlines,3) array of RGB values.
    """
    drawLines(x[elems],color)
##     GL.glBegin(GL.GL_LINES)
##     for i,e in enumerate(elems):
##         if color is not None:
##             GL.glColor3fv(color[i])
##         GL.glVertex3fv(x[e[0]])
##         GL.glVertex3fv(x[e[1]])
##     GL.glEnd()



## PERHAPS THIS COULD BE REPLACED WITH drawLines by reshaping the x ARRAY
def drawEdges(x,color=None):
    """Draw a collection of edges.

    x is a (ntri,2*n,3) shaped array of coordinates. Each of the n pairs
    define a line segment. 

    If color is given it is an (ntri,3) array of RGB values.
    """
    GL.glBegin(GL.GL_LINES)
    for i in range(x.shape[0]):
        for j in range(0,x.shape[1],2):
            if color is not None:
                GL.glColor3fv(color[i])
            GL.glVertex3fv(x[i][j])
            GL.glVertex3fv(x[i][j+1])
    GL.glEnd()


def drawPolyLines(x,color=None,close=True):
    """Draw a collection of polylines.

    x is a (npoly,n,3) shaped array of coordinates. Each polyline consists
    of n or n-1 line segments, depending on whether the polyline is closed
    or not. The default is to close the polyline (connecting the last node
    to the first.

    If color is given it is an (npoly,3) array of RGB values.
    """
    for i in range(x.shape[0]):
        if close:
            GL.glBegin(GL.GL_LINE_LOOP)
        else:
            GL.glBegin(GL.GL_LINE_STRIP)
        if color is not None:
            GL.glColor3fv(color[i])
        for j in range(x.shape[1]):
            GL.glVertex3fv(x[i][j])
        GL.glEnd()


def drawQuadraticCurves(x,color=None,n=8):
    """Draw a collection of curves.

    x is a (nlines,3,3) shaped array of coordinates.
    For each member a quadratic curve through its points is drawn.
    The quadratic curve is approximated with 2*n straight segments.

    If color is given it is an (nlines,3) array of RGB values.
    """
    import simple
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
    nkots = 7
    knots = arange(nkots+1) / float(nkots)
    knots = array([0.,0.,0.,0.,1.,1.,1.,1.])
    
    if not nurb:
        return
    for i,xi in enumerate(x):
        if color is not None:
            GL.glColor3fv(color[i])
        print knots
        print xi
        GLU.gluBeginCurve(nurb)
        GLU.gluNurbsCurve(nurb,knots,xi,GL.GL_MAP1_VERTEX_3)
        GLU.gluEndCurve(nurb)


def drawTriangles(x,mode,color=None,alpha=1.0):
    """Draw a collection of triangles.

    x is a (ntri,3*n,3) shaped array of coordinates.
    Each row contains n triangles drawn with the same color.

    If color is given it is an (npoly,3) array of RGB values.

    mode is either 'flat' or 'smooth' : in 'smooth' mode the normals
    for the lighting are calculated and set
    """
    if mode == 'smooth':
        normal = vectorPairNormals(x[:,1] - x[:,0], x[:,2] - x[:,1])
    GL.glBegin(GL.GL_TRIANGLES)
    for i in range(x.shape[0]):
        if color is not None:
            glColor(color[i],alpha)
        if mode == 'smooth':
            GL.glNormal3fv(normal[i])
        for j in range(x.shape[1]):
            GL.glVertex3fv(x[i][j])
    GL.glEnd()


def drawTriangleElems(coords,elems,mode,color=None,alpha=1.0):
    drawTriangles(coords[elems],mode,color,alpha)

# Experiment using arrays
## def drawTriArray(x,c,mode):
##     GL.glVertexPointerf(x)
##     GL.glColorPointerf(c)
##     GL.glEnable(GL.GL_VERTEX_ARRAY)
##     GL.glEnable(GL.GL_COLOR_ARRAY)
##     if mode == 'smooth':
##         normal = vectorPairNormals(x[:,1] - x[:,0], x[:,2] - x[:,1])
##         GL.glNormalPointerf(normal)
##         GL.glEnable(GL.GL_NORMAL_ARRAY)
##     GL.glBegin(GL.GL_TRIANGLES)
##     GL.glDrawArrays(GL.GL_TRIANGLES,0,x.shape[0])
##     GL.glEnd()

    
def drawPolygons(x,mode,color=None):
    """Draw a collection of polygones.

    x is a (npoly,n,3) shaped array of coordinates.
    Each row contains n triangles drawn with the same color.

    If color is given it is an (npoly,3) array of RGB values.

    mode is either 'flat' or 'smooth' : in 'smooth' mode the normals
    for the lighting are calculated and set
    """
    for i,xi in enumerate(x):
        if color is not None:
            GL.glColor3fv(color[i])
        GL.glBegin(GL.GL_POLYGON)
        for xij in xi:
            GL.glVertex3fv(xij)
        GL.glEnd()

    
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
##        normal = [ cross(edge[i],edge[(i+1) % nplex]) for i in range(nplex) ]
##        normal /= column_stack([sqrt(sum(normal*normal,-1))])
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
        return None # use canvas fgcolor

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
        if color.dtype.kind == 'f' and color.ndim <= 2 and color.shape[-1] == 3:
            # Looks like we have a sane color array
            return color
    except:
        pass

    return None


def saneColorSet(color=None,colormap=None,ncolors=1):
    """Return a sane set of colors.

    A sane set of colors is one that guarantees correct use by the
    draw functions. This means either
    - no color (None)
    - a single color
    - at least as many colors as the value ncolors specifies
    - a color index and a color map with enough colors to satisfy the index.
    The return value is a tuple color,colormap. colormap will return
    unchanged, unless color is an integer array, meaning a color index.
    """
    #GD.debug("COLOR IN: %s" % str(color))
    color = saneColor(color)
    if color is not None:
        if color.dtype.kind == 'i':
            ncolors = color.max()+1
            if colormap is None:
                colormap = GD.canvas.settings.propcolors
            colormap = saneColor(colormap)
            if colormap.shape[0] < ncolors:
                colormap = resize(colormap,(ncolors,3))
        else:
            if color.ndim == 2 and color.shape[0] < ncolors:
                color = resize(color,(ncolors,3))

    #GD.debug("COLOR OUT: %s" % str(color))
    #if colormap is not None:
        #GD.debug("MAP: %s" % str(colormap))
    return color,colormap


### Actors ###############################################

class Actor(object):
    """An Actor is anything that can be drawn in an OpenGL 3D Scene.

    The visualisation of the Scene Actors is dependent on camera position and
    angles, clipping planes, rendering mode and lighting.
    
    An Actor subclass should minimally reimplement the following methods:
      bbox(): return the actors bounding box.
      drawGL(mode): to draw the actor. Takes a mode argument so the
        drawing function can act differently depending on the mode. There are
        currently 5 modes: wireframe, flat, smooth, flatwire, smoothwire.
      drawGL should only contain OpenGL calls that are allowed inside a display
        list. This may include calling the display list of another actor but NOT
        creating a new display list.
    """
    
    def __init__(self):
        self.trans = False
        self.list = None

    def bbox(self):
        return array([[0.0,0.0,0.0],[1.0,1.0,1.0]])

    def drawGL(self,mode):
        """Perform the OpenGL drawing functions to display the actor."""
        raise NotImplementedError

    def draw(self,mode):
        if self.list is None:
            self.create_list(mode)
        GL.glCallList(self.list)

    def redraw(self,mode):
        self.delete_list()
        self.create_list(mode)
        GL.glCallList(self.list)

    def use_list(self):
        if self.list:
            GL.glCallList(self.list)

    def create_list(self,mode):
        self.list = GL.glGenLists(1)
        GL.glNewList(self.list,GL.GL_COMPILE)
        try:
            self.drawGL(mode)
        finally:
            GL.glEndList()

    def delete_list(self):
        if self.list:
            GL.glDeleteLists(self.list,1)
        self.list = None
        
    def nelems(self):
        return 1
    
    def setLineWidth(self,linewidth):
        """Set the linewidth of the Actor."""
        self.linewidth = saneLineWidth(linewidth)

    def setColor(self,color=None,colormap=None):
        """Set the color of the Actor."""
        self.color,self.colormap = saneColorSet(color=color) 

     
class TranslatedActor(Actor):
    """An Actor translated to another position."""

    def __init__(self,A,trl=(0.,0.,0.)):
        Actor.__init__(self)
        self.actor = A
        self.trans = A.trans
        self.trl = asarray(trl)

    def bbox(self):
        return self.actor.bbox() + self.trl

    def drawGL(self,mode):
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glTranslate(*self.trl)
        self.actor.use_list()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()


     
class RotatedActor(Actor):
    """An Actor rotated to another position."""

    def __init__(self,A,normal=(1.,0.,0.),twist=0.0):
        """Created a new rotated actor.

        The rotation is specified by the direction of the local 0 axis
        of the actor.
        """
        Actor.__init__(self)
        self.actor = A
        self.trans = A.trans
        self.rot = rotMatrix(normal,4)

    def bbox(self):
        return self.actor.bbox() # TODO : rotate the bbox !

    def drawGL(self,mode):
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glMultMatrixf(self.rot)
        self.actor.use_list()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()


class CubeActor(Actor):
    """An OpenGL actor with cubic shape and 6 colored sides."""

    def __init__(self,size,color=[red,cyan,green,magenta,blue,yellow]):
        FacingActor.__init__(self)
        self.size = size
        self.color = color

    def bbox(self):
        return (0.5 * self.size) * array([[-1.,-1.,-1.],[1.,1.,1.]])

    def drawGL(self,mode='wireframe'):
        """Draw the cube."""
        drawCube(self.size,self.color)


# This could be subclassed from GridActor
class BboxActor(Actor):
    """Draws a bbox."""

    def __init__(self,bbox,color=None,linewidth=None):
        Actor.__init__(self)
        self.color = color
        self.linewidth = linewidth
        self.bb = bbox
        self.vertices = array(elements.Hex8.vertices) * (bbox[1]-bbox[0]) + bbox[0]
        #print "VERTICES",self.vertices
        self.edges = array(elements.Hex8.edges)
        self.facets = array(elements.Hex8.faces)

    def bbox():
        return self.bb

    def drawGL(self,mode,color=None):
        """Always draws a wireframe model of the bbox."""
        if color is None:
            color = self.color
        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)
        drawLineElems(self.vertices,self.edges,color)
            

class TriadeActor(Actor):
    """An OpenGL actor representing a triade of global axes."""

    def __init__(self,size,color=[red,green,blue,cyan,magenta,yellow]):
        Actor.__init__(self)
        self.size = size
        self.color = color

    def bbox(self):
        return (0.5 * self.size) * array([[0.,0.,0.],[1.,1.,1.]])

    def drawGL(self,mode='wireframe'):
        """Draw the triade."""
        GL.glShadeModel(GL.GL_FLAT)
        GL.glPolygonMode(GL.GL_FRONT, GL.GL_FILL)
        GL.glPolygonMode(GL.GL_BACK, GL.GL_LINE)
        GL.glBegin(GL.GL_TRIANGLES)
        GL.glColor(*self.color[0])
        GL.glVertex3f(0.0,0.0,0.0)
        GL.glVertex3f(1.0,0.0,0.0)
        GL.glVertex3f(0.0,1.0,0.0)
        GL.glColor(*self.color[1])
        GL.glVertex3f(0.0,0.0,0.0)
        GL.glVertex3f(0.0,1.0,0.0)
        GL.glVertex3f(0.0,0.0,1.0)
        GL.glColor(*self.color[2])
        GL.glVertex3f(0.0,0.0,0.0)
        GL.glVertex3f(0.0,0.0,1.0)
        GL.glVertex3f(1.0,0.0,0.0)
##        GL.glColor(*self.color[3])
##        GL.glVertex3f(0.0,0.0,0.0)
##        GL.glVertex3f(0.0,1.0,0.0)
##        GL.glVertex3f(1.0,0.0,0.0)
##        GL.glColor(*self.color[4])
##        GL.glVertex3f(0.0,0.0,0.0)
##        GL.glVertex3f(0.0,0.0,1.0)
##        GL.glVertex3f(0.0,1.0,0.0)
##        GL.glColor(*self.color[5])
##        GL.glVertex3f(0.0,0.0,0.0)
##        GL.glVertex3f(1.0,0.0,0.0)
##        GL.glVertex3f(0.0,0.0,1.0)
        GL.glEnd()
        GL.glBegin(GL.GL_LINES)
        GL.glColor3f(*black)
        GL.glVertex3f(0.0,0.0,0.0)
        GL.glVertex3f(2.0,0.0,0.0)
        GL.glVertex3f(0.0,0.0,0.0)
        GL.glVertex3f(0.0,2.0,0.0)
        GL.glVertex3f(0.0,0.0,0.0)
        GL.glVertex3f(0.0,0.0,2.0)
        GL.glEnd()

  
class GridActor(Actor):
    """Draws a (set of) grid(s) in one of the coordinate planes."""

    def __init__(self,nx=(1,1,1),ox=(0.0,0.0,0.0),dx=(1.0,1.0,1.0),linecolor=black,linewidth=None,planecolor=white,alpha=0.2,lines=True,planes=True):
        Actor.__init__(self)
        self.linecolor = saneColor(linecolor)
        self.planecolor = saneColor(planecolor)
        self.linewidth = linewidth
        self.alpha = alpha
        self.trans = True
        self.lines = lines
        self.planes = planes
        self.nx = asarray(nx)
        self.x0 = asarray(ox)
        self.x1 = self.x0 + self.nx * asarray(dx)

    def bbox(self):
        return array([self.x0,self.x1])

    def drawGL(self,mode):
        """Draw the grid."""

        #print "BBOX %s" % self.bbox()
        if self.lines:
            if self.linewidth:
                GL.glLineWidth(self.linewidth)
            glColor(self.linecolor)
            drawGridLines(self.x0,self.x1,self.nx)
        
        if self.planes:
            glColor(self.planecolor,self.alpha)
            drawGridPlanes(self.x0,self.x1,self.nx)

            
class CoordPlaneActor(Actor):
    """Draws a set of 3 coordinate planes."""

    def __init__(self,nx=(1,1,1),ox=(0.0,0.0,0.0),dx=(1.0,1.0,1.0),linecolor=black,linewidth=None,planecolor=white,alpha=0.5,lines=True,planes=True):
        Actor.__init__(self)
        self.linecolor = saneColor(linecolor)
        self.planecolor = saneColor(planecolor)
        self.linewidth = linewidth
        self.alpha = alpha
        self.trans = True
        self.lines = lines
        self.planes = planes
        self.nx = asarray(nx)
        self.x0 = asarray(ox)
        self.x1 = self.x0 + self.nx * asarray(dx)

    def bbox(self):
        return array([self.x0,self.x1])

    def drawGL(self,mode):
        """Draw the grid."""

        for i in range(3):
            nx = self.nx.copy()
            nx[i] = 0
            
            if self.lines:
                glColor(self.linecolor)
                drawGridLines(self.x0,self.x1,nx)

            if self.planes:
                glColor(self.planecolor,self.alpha)
                drawGridPlanes(self.x0,self.x1,nx)

            
class PlaneActor(Actor):
    """A plane in a 3D scene."""

    def __init__(self,P,N,nx=(2,2,2),ox=(0.,0.,0.),size=(0.0,1.0,1.0),linecolor=black,linewidth=None,planecolor=white,alpha=0.5,lines=True,planes=True):
        """Create a new Plane.

        The plane is defined by a point P in the plane and a normal N.
        """
        Actor.__init__(self)
        self.linecolor = saneColor(linecolor)
        self.planecolor = saneColor(planecolor)
        self.linewidth = linewidth
        self.alpha = alpha
        self.trans = True
        self.lines = lines
        self.planes = planes
        self.nx = asarray(nx)
        ox = asarray(ox)
        sz = 0.5*asarray(size)
        self.x0,self.x1 = ox-sz, ox+sz
        #print self.x0

    def bbox(self):
        return array([self.x0,self.x1])

    def drawGL(self,mode):
        """Draw the grid."""

        for i in range(3):
            nx = self.nx.copy()
            nx[i] = 0
            
            if self.lines:
                glColor(self.linecolor)
                drawGridLines(self.x0,self.x1,nx)

            if self.planes:
                glColor(self.planecolor,self.alpha)
                drawGridPlanes(self.x0,self.x1,nx)
        

###########################################################################

quadratic_curve_ndiv = 8
class FormexActor(Actor,Formex):
    """An OpenGL actor which is a Formex."""
    mark = False

    def __init__(self,F,color=None,colormap=None,bkcolor=None,bkcolormap=None,linewidth=None,marksize=None,eltype=None,alpha=1.0):
        """Create a multicolored Formex actor.

        The colors argument specifies a list of OpenGL colors for each
        of the property values in the Formex. If the list has less
        values than the PropSet, it is wrapped around. It can also be
        a single OpenGL color, which will be used for all elements.
        For surface type elements, a bkcolor color can be given for
        the backside (inside) of the surface. Default will be the same
        as the front color.
        The user can specify a linewidth to be used when drawing
        in wireframe mode.

        plex-1: if eltype == 'point3D', a 3D cube with 6 differently colored
                faces is drawn, else a fixed size dot is drawn.
        """
        Actor.__init__(self)
        # Initializing with F alone gives problems with self.p !
        Formex.__init__(self,F.f,F.p)
        self.eltype = eltype
        
        self.setLineWidth(linewidth)
        self.setColor(color,colormap)
        self.setBkColor(bkcolor,bkcolormap)
        self.setAlpha(alpha)
        
        if self.nplex() == 1:
            self.setMarkSize(marksize)
        self.list = None


    def setColor(self,color=None,colormap=None):
        """Set the color of the Actor."""
        self.color,self.colormap = saneColorSet(color,colormap,self.nelems()) 


    def setBkColor(self,color=None,colormap=None):
        """Set the backside color of the Actor."""
        self.bkcolor,self.bkcolormap = saneColorSet(color,colormap,self.nelems())

    def setAlpha(self,alpha):
        """Set the Actors alpha value."""
        self.alpha = float(alpha)
        self.trans = self.alpha < 1.0
        #if self.trans:
        #    GD.debug("Setting Actor's ALPHA value to %f" % alpha)


    def setMarkSize(self,marksize):
        """Set the mark size.

        The mark size is currently only used with plex-1 Formices.
        """
#### DEFAULT MARK SIZE SHOULD BECOME A CANVAS SETTING!!
        
        if marksize is None:
            marksize = 4.0 # default size 
        if self.eltype == 'point3d':
            # ! THIS SHOULD BE SET FROM THE SCENE SIZE
            #   RATHER THAN FORMEX SIZE 
            marksize = self.size() * marksize
            if marksize <= 0.0:
                marksize = 1.0
            self.setMark(marksize,"cube")
        self.marksize = marksize


    def setMark(self,size,type):
        """Create a symbol for drawing 3D points."""
        self.mark = GL.glGenLists(1)
        GL.glNewList(self.mark,GL.GL_COMPILE)
        if type == "sphere":
            drawSphere(size)
        else:
            drawCube(size)
        GL.glEndList()
        

    bbox = Formex.bbox


    def drawGL(self,mode='wireframe',color=None,alpha=None):
        """Draw the formex.

        if color is None, it is drawn with the color specified on creation.
        if color == 'prop' and a colormap was installed, props define color.
        else, color should be an array of RGB values, either with shape
        (3,) for a single color, or (nelems,3) for differently colored
        elements 

        if mode ends with wire (smoothwire or flatwire), two drawing
        operations are done: one with wireframe and color black, and
        one with mode[:-4] and self.color.
        """

        if mode.endswith('wire'):
            self.drawGL(mode[:-4],color=color)
            self.drawGL('wireframe',color=asarray(black))
            return

        if alpha is None:
            alpha = self.alpha
            
        ## CURRENTLY, ONLY color=None is used
        
        if color is None:  
            color = self.color
        
        if color is None:  # no color
            #GD.debug("NO COLOR")
            pass
        
        elif color.dtype.kind == 'f' and color.ndim == 1:  # single color
            #GD.debug("SINGLE COLOR %s ALPHA %s" % (str(color),alpha))
            GL.glColor(append(color,alpha))
            #print append(color,alpha)
            color = None

        elif color.dtype.kind == 'i': # color index
            #GD.debug("COLOR INDEX %s\n%s" % (str(color),str(self.colormap)))
            color = self.colormap[color]

        else: # a full color array : use as is
            pass

##        if color is not None:
##            GD.debug("FULL COLOR %s" % str(color))

        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)
            
        nnod = self.nplex()
        if nnod == 1:
            x = self.f.reshape((-1,3))
            if self.eltype == 'point3d':
                drawAtPoints(x,self.mark,color)
            else:
                drawPoints(x,self.marksize,color)
                
        elif nnod == 2:
            drawLines(self.f,color)
        
        elif nnod == 3 and self.eltype == 'curve':
            drawQuadraticCurves(self.f,color,n=quadratic_curve_ndiv)
            
        elif mode=='wireframe' :
            if self.eltype == 'tet':
                edges = [ 0,1, 0,2, 0,3, 1,2, 1,3, 2,3 ]
                coords = self.f[:,edges,:]
                drawEdges(coords,color)
            elif self.eltype == 'hex':
                edges = [ 0,1, 1,2, 2,3, 0,3, 0,4, 1,5, 2,6, 3,7, 4,5, 5,6, 6,7, 7,4]
                coords = self.f[:,edges,:]
                drawEdges(coords,color)
            else:
                drawPolyLines(self.f,color)
                
        elif nnod == 3:
            drawTriangles(self.f,mode,color,alpha)
            
        elif nnod == 4:
            if self.eltype=='tet':
                faces = [ 0,1,2, 0,2,3, 0,3,1, 3,2,1 ]
                coords = self.f[:,faces,:]
                drawTriangles(coords,mode,color)
            else: # (possibly non-plane) quadrilateral
                drawQuadrilaterals(self.f,mode,color)

        elif nnod == 8:
            if self.eltype=='hex':
                faces = [0,1,2,3, 4,5,6,7, 0,3,7,4, 1,2,6,5, 0,1,5,4, 3,2,6,7]
                coords = self.f[:,faces,:]
                drawQuadrilaterals(coords,mode,color)
        
        else:
            drawPolygons(self.f,mode,color=None)


#############################################################################

class SurfaceActor(Actor,Surface):
    """Draws a triangulated surface specified by points and connectivity."""

    def __init__(self,S,color=None,colormap=None,bkcolor=None,bkcolormap=None,linewidth=None,alpha=1.0):
        
        Actor.__init__(self)
        Surface.__init__(self,S.coords,S.edges,S.faces)
        
        self.setLineWidth(linewidth)
        self.setColor(color,colormap)
        self.setBkColor(bkcolor,bkcolormap)
        self.setAlpha(alpha)

        self.list = None


    def setColor(self,color=None,colormap=None):
        """Set the color of the Actor."""
        self.color,self.colormap = saneColorSet(color,colormap,self.nelems()) 


    def setBkColor(self,color=None,colormap=None):
        """Set the backside color of the Actor."""
        self.bkcolor,self.bkcolormap = saneColorSet(color,colormap,self.nelems())

    def setAlpha(self,alpha):
        """Set the Actors alpha value."""
        self.alpha = float(alpha)
        self.trans = self.alpha < 1.0


    bbox = Surface.bbox


    def drawGL(self,mode,color=None,alpha=None):
        """Draw the surface."""
        if mode.endswith('wire'):
            self.drawGL(mode[:-4],color=color)
            self.drawGL('wireframe',color=asarray(black))
            return

        if alpha is None:
            alpha = self.alpha           

        if color == None:
            color = self.color

##                 if mode == 'wireframe':
##                     # adapt color array to edgeselect
##                     color = concatenate([self.color,self.color,self.color],axis=-1)
##                     color = color.reshape((-1,2))[self.edgeselect]
        
        if color is None:  # no color
            pass
        
        elif color.dtype.kind == 'f' and color.ndim == 1:  # single color
            GL.glColor(append(color,alpha))
            color = None

        elif color.dtype.kind == 'i': # color index
            color = self.colormap[color]

        else: # a full color array : use as is
            pass


        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)

        if mode=='wireframe' :
            drawLines(self.coords[self.edges],color)
        else:
            self.refresh()
            drawTriangles(self.coords[self.elems],mode,color,alpha)

### End

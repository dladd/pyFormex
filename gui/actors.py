# actors.py
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Mon Feb 26 08:57:40 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""OpenGL actors for populating the 3D scene."""

from OpenGL import GL,GLU
from colors import *
from formex import *
from plugins import elements


### Some drawing functions ###############################################


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


def drawPoints(x,c,s):
    """Draw a collection of points with color c and size s.

    x is a (npoints,3) shaped array of coordinates.
    c is the point color, either (npoints,3) for differently colored points
      or (3,) when all points have same color
    s is the point size
    """
    GL.glPointSize(s)
    multicolor = len(c.shape) > 1
    if not multicolor:
        GL.glColor3fv(c)
    GL.glBegin(GL.GL_POINTS)
    for i in range(x.shape[0]):
        if multicolor:
            GL.glColor3fv(c[i])
        GL.glVertex3fv(x[i])
    GL.glEnd()


def drawAtPoints(x,mark):
    """Draw a copy of a 3D actor mark at all points in x.

    x is a (npoints,3) shaped array of coordinates.
    mark is any 3D Actor having a display list.
    """
    for xi in x:
        GL.glPushMatrix()
        GL.glTranslatef(*xi)
        GL.glCallList(mark)
        GL.glPopMatrix()


def drawLines(x,c,w):
    """Draw a collection of lines.

    x is a (nlines,2,3) shaped array of coordinates.
    c is the line color, either (nlines,3) for differently colored lines
      or (3,) when all lines have same color
    w is the linewidth.
    """
    GL.glLineWidth(w)
    multicolor = len(c.shape) > 1
    if not multicolor:
        GL.glColor3fv(c)
    GL.glBegin(GL.GL_LINES)
    for i in range(x.shape[0]):
        if multicolor:
            GL.glColor3fv(c[i])
        GL.glVertex3fv(x[i][0])
        GL.glVertex3fv(x[i][1])
    GL.glEnd()


def drawEdges(x,edges,c,w):
    """Draw a collection of edges.

    x is a (ncoords,3) coordinate array.
    edges is a (nedges,2) integer array of connected node numbers.
    c is a 3-component color or a (nedges,3) color array.
    w is the linewidth.
    """
    GL.glLineWidth(w)
    multicolor = len(c.shape) > 1
    if not multicolor:
        GL.glColor3fv(c)
    GL.glBegin(GL.GL_LINES)
    for i,e in enumerate(edges):
        if multicolor:
            GL.glColor3fv(c[i])
        GL.glVertex3fv(x[e[0]])
        GL.glVertex3fv(x[e[1]])
    GL.glEnd()


def drawTriEdges(x,c,w):
    """Draw a collection of lines.

    x is a (ntri,2*n,3) shaped array of coordinates.
    c is a 3-component color or a (ntri,3) color array.
    w is the linewidth.
    """
    GL.glLineWidth(w)
    multicolor = len(c.shape) > 1
    if not multicolor:
        GL.glColor3fv(c)
    GL.glBegin(GL.GL_LINES)
    for i in range(x.shape[0]):
        for j in range(0,x.shape[1],2):
            if multicolor:
                GL.glColor3fv(c[i])
            GL.glVertex3fv(x[i][j])
            GL.glVertex3fv(x[i][j+1])
    GL.glEnd()


def drawPolyLines(x,c,w,close=True):
    """Draw a collection of polylines.

    x is a (npts,n,3) shaped array of coordinates. Each polyline consists
    of n or n-1 line segments, depending on whether the polyline is closed
    or not. The default is to close the polyline (connecting the last node
    to the first.
    c is a 3-component color or a (npts(+1),3) color array.
    w is the linewidth.
    """
    GL.glLineWidth(w)
    multicolor = len(c.shape) > 1
    if not multicolor:
        GL.glColor3fv(c)
    for i in range(x.shape[0]):
        if close:
            GL.glBegin(GL.GL_LINE_LOOP)
        else:
            GL.glBegin(GL.GL_LINE_STRIP)
        if multicolor:
            GL.glColor3fv(c[i])
        for j in range(x.shape[1]):
            GL.glVertex3fv(x[i][j])
        GL.glEnd()


def drawTriangles(x,c,mode):
    """Draw a collection of triangles.

    x is a (ntri,3*n,3) shaped array of coordinates.
    Each row contains n triangles drawn with the same color.
    c is a 3-component color or a (ntri,3) color array.
    mode is either 'flat' or 'smooth'
    """
    if mode == 'smooth':
        normal = vectorPairNormals(x[:,1] - x[:,0], x[:,2] - x[:,1])
    multicolor = len(c.shape) > 1
    if not multicolor:
        GL.glColor3fv(c)
    GL.glBegin(GL.GL_TRIANGLES)
    for i in range(x.shape[0]):
        if multicolor:
            GL.glColor3fv(c[i])
        if mode == 'smooth':
            GL.glNormal3fv(normal[i])
        for j in range(x.shape[1]):
            GL.glVertex3fv(x[i][j])
    GL.glEnd()
       

def drawTriArray(x,c,mode):
    GL.glVertexPointerf(x)
    GL.glColorPointerf(c)
    GL.glEnable(GL.GL_VERTEX_ARRAY)
    GL.glEnable(GL.GL_COLOR_ARRAY)
    if mode == 'smooth':
        normal = vectorPairNormals(x[:,1] - x[:,0], x[:,2] - x[:,1])
        GL.glNormalPointerf(normal)
        GL.glEnable(GL.GL_NORMAL_ARRAY)
    GL.glBegin(GL.GL_TRIANGLES)
    GL.glDrawArrays(GL.GL_TRIANGLES,0,x.shape[0])
    GL.glEnd()

    
def drawQuadrilaterals(x,c,mode):
    """Draw a collection of quadrilaterals.

    x is a (nquad,4*n,3) shaped array of coordinates.
    Each row contains n quads drawn with the same color.
    c is a (nquad,3) shaped array of RGB values.
    mode is either 'flat' or 'smooth'
    """
    nplex = x.shape[1]
    if mode == 'smooth':
        edge = [ x[:,i,:] - x[:,i-1,:] for i in range(nplex) ]
        normal = [ vectorPairNormals(edge[i],edge[(i+1) % nplex]) for i in range(nplex) ]
##        normal = [ cross(edge[i],edge[(i+1) % nplex]) for i in range(nplex) ]
##        normal /= column_stack([sqrt(sum(normal*normal,-1))])
    GL.glBegin(GL.GL_QUADS)
    for i in range(x.shape[0]):
        GL.glColor3fv(c[i])
        for j in range(nplex):
            if mode == 'smooth':
                GL.glNormal3fv(normal[j][i])
            GL.glVertex3fv(x[i][j])
    GL.glEnd()

 
### Actors ###############################################

class Actor(object):
    """An Actor is anything that can be drawn in an OpenGL 3D Scene.

    The visualisation of the Scene Actors is dependent on camera position and
    angles, clipping planes, rendering mode and lighting.
    
    An Actor subclass should minimally have the following three methods:
    __init__(): to initialize the actor.
    bbox(): return the actors bounding box.
    draw(mode='wireframe'): to draw the actor. Takes a mode argument so the
      drawing function can act differently depending on the mode. There are
      currently 5 modes: wireframe, flat, smooth, flatwire, smoothwire.
    """
    
    def __init__(self):
        pass

    def bbox(self):
        return array([[0.0,0.0,0.0],[1.0,1.0,1.0]])

    def draw(self):
        pass

    

class CubeActor(Actor):
    """An OpenGL actor with cubic shape and 6 colored sides."""

    def __init__(self,size,color=[red,cyan,green,magenta,blue,yellow]):
        FacingActor.__init__(self)
        self.size = size
        self.color = color

    def bbox(self):
        return (0.5 * self.size) * array([[-1.,-1.,-1.],[1.,1.,1.]])

    def draw(self,mode='wireframe'):
        """Draw the cube."""
        drawCube(self.size,self.color)



class BboxActor(Actor):
    """Draws a bbox."""

    def __init__(self,bbox,color=None,linewidth=None):
        Actor.__init__(self)
        self.color = color
        self.linewidth = linewidth
        self.bb = bbox
        self.vertices = array(elements.Hex8.nodes) * (bb[1]-bb[0]) + bb[0]
        self.edges = array(elements.Hex8.edges)
        self.facets = array(elements.Hex8.faces)

    def bbox():
        return self.bb

    def draw(self,mode,color=None):
        """Always draws a wireframe model of the bbox."""
        if color is None:
            color = self.color
        drawEdges(self.vertices,self.edges,color,self.linewidth)
            
    

class TriadeActor(Actor):
    """An OpenGL actor representing a triade of global axes."""

    def __init__(self,size,color=[red,green,blue,cyan,magenta,yellow]):
        Actor.__init__(self)
        self.size = size
        self.color = color

    def bbox(self):
        return (0.5 * self.size) * array([[0.,0.,0.],[1.,1.,1.]])

    def draw(self,mode='wireframe'):
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


class FormexActor(Actor,Formex):
    """An OpenGL actor which is a Formex."""
    mark = False

    def __init__(self,F,color=[black],bkcolor=None,linewidth=1.0,markscale=0.02,marksize=None,eltype=None):
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
        if F.p is None:
            # Create a NEW formex because we have to add properties
            p = arange(F.f.shape[0])
            Formex.__init__(self,F.f,p)
        else:
            # Initializing with F alone gives problems with self.p !
            Formex.__init__(self,F.f,F.p)
        self.list = None
        #print self.f.shape
        #print self.p.shape
        # make the color list large enough
        #
        # WE COULD KEEP A SINGLE COLOR AS SPECIAL CASE,
        # THEN FORMICES WITHOUT PROPERTIES WOULD BE SINGLE COLOR
        #
        mprop = max(self.propSet()) + 1
        color = array(color)
        self.color = resize(color,(mprop,3))
        if bkcolor is None:
            self.bkcolor = self.color
        else:
            self.bkcolor = resize(array(bkcolor),mprop)
        self.linewidth = float(linewidth)
        if self.nplex() == 1:
            if marksize is None:
                if eltype == 'point3d':
                    # ! THIS SHOULD BE SET FROM THE SCENE SIZE
                    #   RATHER THAN FORMEX SIZE 
                    marksize = self.size() * float(markscale)
                    if marksize <= 0.0:
                        marksize = 1.0
                    self.setMark(marksize,"cube")
                else:
                    marksize = 4.0
            self.marksize = marksize
        self.eltype = eltype


    bbox = Formex.bbox


    def draw(self,mode='wireframe',color=None):
        """Draw the formex.

        If a color is specified, it should be either a single color
        or an (nelems,3) array of colors.
        If not, the Formex will be drawn with the colors set on creating
        the FormexActor.

        if mode enda with wire (smoothwire or flatwire), two drawing
          operationsis are done: one with wirframe and color black, and
          one with mode[:-4] and specified color.
        """

        if mode.endswith('wire'):
            self.draw(mode[:-4],color=color)
            self.draw('wireframe',color=[0.,0.,0.])
            return
        
        nnod = self.nplex()
        print "color = %s" % color
        if color is None:
            color = self.color[self.p]
        color = asarray(color)
        
        if nnod == 1:
            x = self.f.reshape((-1,3))
            if self.eltype == 'point3d':
                drawAtPoints(x,self.mark)
            else:
                drawPoints(x,color,self.marksize)
                
        elif nnod == 2:
            drawLines(self.f,color,self.linewidth)
            
        elif mode=='wireframe' :
            if not self.eltype:
                drawPolyLines(self.f,color,self.linewidth)
            elif self.eltype == 'tet':
                edges = [ 0,1, 0,2, 0,3, 1,2, 1,3, 2,3 ]
                coords = self.f[:,edges,:]
                drawTriEdges(coords,color,self.linewidth)
                
        elif nnod == 3:
            drawTriangles(self.f,color,mode)
            
        elif nnod == 4:
            if self.eltype=='tet':
                faces = [ 0,1,2, 0,2,3, 0,3,1, 3,2,1 ]
                coords = self.f[:,faces,:]
                drawTriangles(coords,color,mode)
            else: # (possibly non-plane) quadrilateral
                drawQuadrilaterals(self.f,color,mode)
            
        else:
            for prop,elem in zip(self.p,self.f):
                col = self.color[int(prop)]
                GL.glColor3f(*(col))
                GL.glBegin(GL.GL_POLYGON)
                for nod in elem:
                    GL.glVertex3f(*nod)
                GL.glEnd()


    def setMark(self,size,type):
        """Create a symbol for drawing 3D points."""
        self.mark = GL.glGenLists(1)
        GL.glNewList(self.mark,GL.GL_COMPILE)
        if type == "sphere":
            drawSphere(size)
        else:
            drawCube(size)
        GL.glEndList()





class SurfaceActor(Actor):
    """Draws a triangulated surface."""

    def __init__(self,nodes,elems,color=black,linewidth=1.0):
        
        Actor.__init__(self)
        self.color = asarray(color)
        self.linewidth = linewidth
        self.nodes = nodes
        self.elems = elems
        self.bb = boundingBox(nodes)
        edges = elems[:,elements.Tri3.edges].reshape((-1,2))
        self.edges = edges[edges[:,0] < edges[:,1]]


    def bbox(self):
        return self.bb


    def draw(self,mode,color=None):
        """Draw the surface."""
        if mode.endswith('wire'):
            #self.draw('wireframe',color=[0.,0.,0.])
            self.draw(mode[:-4])
            self.draw('wireframe',color=[0.,0.,0.])
            return

        if color == None:
            color = self.color
        if mode=='wireframe' :
            drawEdges(self.nodes,self.edges,color,self.linewidth)
        else:
            if color.ndim == 1:
                color = zeros((self.elems.shape[0],3))
                color[:,:] = self.color
            drawTriangles(self.nodes[self.elems],color,mode)

### End

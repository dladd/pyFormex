# canvas.py
# $Id$
"""OpenGL actors for populating the 3D scene(3D)."""

from OpenGL import GL,GLU
from colors import *
from formex import *
from plugins import elements


markscale = 0.001

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


def drawLines(x,c,w):
    """Draw a collection of lines.

    x is a (nlines,2,3) shaped array of coordinates.
    c is a (nlines,3) shaped array of RGB values.
    w is the linewidth.
    """
    GL.glLineWidth(w)
    GL.glBegin(GL.GL_LINES)
    for i in range(x.shape[0]):
        GL.glColor3f(*(c[i]))
        GL.glVertex3fv(x[i][0])
        GL.glVertex3fv(x[i][1])
    GL.glEnd()


def drawEdges(x,edges,color=None,width=None):
    """Draw a collection of edges.

    x is a (ncoords,3) coordinate array.
    edges is a (nedges,2) integer array of connected node numbers.
    color is a 3-component color.
    width is the linewidth.
    """
    print "DRAWEDGES"
    print x.shape
    print edges.shape
    if color is not None:
        GL.glColor3fv(color)
    if width is not None:
        GL.glLineWidth(width)
    GL.glBegin(GL.GL_LINES)
    for e in edges:
        GL.glVertex3fv(x[e[0]])
        GL.glVertex3fv(x[e[1]])
    GL.glEnd()


def drawColoredEdges(x,edges,color,width=None):
    """Draw a collection of edges.

    x is a (ncoords,3) coordinate array.
    edges is a (nedges,2) integer array of connected node numbers.
    color is a (nedges,3) color array.
    width is the linewidth.
    """
    print "DRAWCOLOREDEDGES"
    print x.shape
    print edges.shape
    print color.shape
    if width is not None:
        GL.glLineWidth(width)
    GL.glBegin(GL.GL_LINES)
    for c,e in zip(color,edges):
        print x[e[0]],x[e[1]]
        GL.glColor3fv(c)
        GL.glVertex3fv(x[e[0]])
        GL.glVertex3fv(x[e[1]])
    GL.glEnd()



def drawTriEdges(x,c,w):
    """Draw a collection of lines.

    x is a (ntri,2*n,3) shaped array of coordinates.
    c is a (ntri,3) shaped array of RGB values.
    w is the linewidth.
    """
    GL.glLineWidth(w)
    GL.glBegin(GL.GL_LINES)
    for i in range(x.shape[0]):
        for j in range(0,x.shape[1],2):
            GL.glColor3f(*(c[i]))
            GL.glVertex3f(*(x[i][j]))
            GL.glVertex3f(*(x[i][j+1]))
    GL.glEnd()


def drawPolyLines(x,c,w,close=True):
    """Draw a collection of polylines.

    x is a (ntri,n,3) shaped array of coordinates. Each polyline consists
    of n or n-1 line segments, depending on whether the polyline is closed
    or not. The default is to close the polyline (connecting the last node
    to the first.
    c is a (ntri,3) shaped array of RGB values.
    w is the linewidth.
    """
    GL.glLineWidth(w)
    for i in range(x.shape[0]):
        if close:
            GL.glBegin(GL.GL_LINE_LOOP)
        else:
            GL.glBegin(GL.GL_LINE_STRIP)
        GL.glColor3f(*(c[i]))
        for j in range(x.shape[1]):
            GL.glVertex3f(*(x[i][j]))
        GL.glEnd()


def drawTriangles(x,c,mode):
    """Draw a collection of triangles.

    x is a (ntri,3*n,3) shaped array of coordinates.
    Each row contains n triangles drawn with the same color.
    c is a (ntri,3) shaped array of RGB values.
    mode is either 'flat' or 'smooth'
    """
    if mode == 'smooth':
        normal = cross(x[:,1,:] - x[:,0,:], x[:,2,:] - x[:,1,:])
    GL.glBegin(GL.GL_TRIANGLES)
    for i in range(x.shape[0]):
        GL.glColor3f(*c[i])
        if mode == 'smooth':
            GL.glNormal3f(*normal[i])
        for j in range(x.shape[1]):
            GL.glVertex3f(*(x[i][j]))
    GL.glEnd()


def drawTri(x,c,mode):
    GL.glVertexPointerf(x)
    GL.glColorPointerf(c)
    GL.glEnable(GL.GL_VERTEX_ARRAY)
    GL.glEnable(GL.GL_COLOR_ARRAY)
    if mode == 'smooth':
        normal = cross(x[:,1,:] - x[:,0,:], x[:,2,:] - x[:,1,:])
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
        normal = [ cross(edge[i],edge[(i+1) % nplex]) for i in range(nplex) ]
    GL.glBegin(GL.GL_QUADS)
    for i in range(x.shape[0]):
        GL.glColor3f(*c[i])
        for j in range(nplex):
            if mode == 'smooth':
                GL.glNormal3f(*normal[j][i])
            GL.glVertex3f(*(x[i][j]))
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
      currently 3 modes: wireframe, flat, smooth.
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
        Actor.__init__(self)
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

    def draw(self,mode):
        """Always draws a wireframe model of the bbox."""
        if self.color is not None and len(self.color.shape) == 2:
            drawColoredEdges(self.vertices,self.edges,self.color,self.linewidth)
        else:
            drawEdges(self.vertices,self.edges,self.color,self.linewidth)
            
    

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

    def __init__(self,F,color=[black],bkcolor=None,linewidth=1.0,eltype=None):
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
        """
        Actor.__init__(self)
        Formex.__init__(self,F.f,F.p)
        self.list = None
        if self.p is None:
            self.setProp(arange(self.nelems()))
        # make the color list large enough
        #
        # WE COULD KEEP A SINGLE COLOR
        #
        mprop = max(self.propSet()) + 1
        color = array(color)
        self.color = resize(array(color),(mprop,3))
        if bkcolor is None:
            self.bkcolor = self.color
        else:
            self.bkcolor = resize(array(bkcolor),mprop)
        self.linewidth = float(linewidth)
        if self.nnodel() == 1:
            # ! THIS SHOULD BETTER BE SET FROM THE SCENE SIZE ! 
            sz = self.size()
            print("Size %s" % sz)
            if sz <= 0.0:
                sz = 1.0
            self.setMark(sz*markscale,"cube")
        self.eltype = eltype

    def draw(self,mode='wireframe'):
        """Draw the formex."""
        nnod = self.nnodel()
        nelem = self.nelems()
        
        if nnod == 1:
            for elem in self.f:
                GL.glPushMatrix()
                GL.glTranslatef (*elem[0])
                GL.glCallList(self.mark)
                GL.glPopMatrix()
                
        elif nnod == 2:
            drawLines(self.f,self.color[self.p],self.linewidth)
            
        elif mode=='wireframe' :
            if not self.eltype:
                drawPolyLines(self.f,self.color[self.p],self.linewidth)
            elif self.eltype == 'tet':
                edges = [ 0,1, 0,2, 0,3, 1,2, 1,3, 2,3 ]
                coords = self.f[:,edges,:]
                drawTriEdges(coords,self.color[self.p],self.linewidth)
                
        elif nnod == 3:
            drawTriangles(self.f,self.color[self.p],mode)
            
        elif nnod == 4:
            if self.eltype=='tet':
                faces = [ 0,1,2, 0,2,3, 0,3,1, 3,2,1 ]
                coords = self.f[:,faces,:]
                drawTriangles(coords,self.color[self.p],mode)
            else: # (possibly non-plane) quadrilateral
                drawQuadrilaterals(self.f,self.color[self.p],mode)
            
        else:
            for prop,elem in zip(self.p,self.f):
                col = self.color[int(prop)]
                GL.glColor3f(*(col))
                GL.glBegin(GL.GL_POLYGON)
                for nod in elem:
                    GL.glVertex3f(*nod)
                GL.glEnd()


    def setMark(self,size,type):
        """Create a symbol for drawing vertices."""
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
        self.color = array(color)
        self.linewidth = linewidth
        self.vertices = nodes
        self.bbox = boundingBox(nodes)
        edges = elems[:,elements.Tri3.edges].reshape((-1,2))
        self.edges = edges[edges[:,0] < edges[:,1]]
        self.facets = elems
        print "SURFACE ACTOR"
        print self.color.shape
        print self.vertices.shape
        print self.edges.shape
        print self.facets.shape

    def draw(self,mode):
        """Always draws a wireframe model of the bbox."""
        if self.color.ndim == 2:
        
            drawColoredEdges(self.vertices,self.edges,self.color,self.linewidth)
        else:
            drawEdges(self.vertices,self.edges,self.color,self.linewidth)

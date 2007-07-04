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

import globaldata as GD

from OpenGL import GL,GLU
from colors import *
from formex import *
from plugins import elements


### Some drawing functions ###############################################


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
    GL.glBegin(GL.GL_LINES)
    for i,e in enumerate(elems):
        if color is not None:
            GL.glColor3fv(color[i])
        GL.glVertex3fv(x[e[0]])
        GL.glVertex3fv(x[e[1]])
    GL.glEnd()



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


def drawTriangles(x,mode,color=None):
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
            GL.glColor3fv(color[i])
        if mode == 'smooth':
            GL.glNormal3fv(normal[i])
        for j in range(x.shape[1]):
            GL.glVertex3fv(x[i][j])
    GL.glEnd()

    
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
        drawLineElems(self.vertices,self.edges,color,self.linewidth)
            
    

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

    def __init__(self,F,color=None,bkcolor=None,colormap=None,linewidth=None,markscale=0.02,marksize=None,eltype=None):
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
        self.setColor(color,bkcolor)
        self.setLineWidth(linewidth)
        self.eltype = eltype
        if self.nplex() == 1:
            self.setMarkSize(marksize)
        self.list = None


    def setColor(self,color=None,bkcolor=None,colormap=None):
        """Set the color of the Formex, and possible the backplane color.

        If color is None, it will be drawn with the current foreground color.
        
        """
        #GD.debug("SETTING COLOR")
        #GD.debug(color)
        if color is None:
            pass # use canvas fgcolor
        
        elif color == 'prop':
            #GD.debug(self.p)
            if self.p is None:
                # no properties defined: use viewport fgcolor
                color = None
            else:
                self.setColorMap(colormap)
               
        elif color == 'random':
            # create random colors
            color = random.random((self.nelems(),3))
            
        elif type(color) == str:
            # convert named color to RGB tuple
            color = asarray(GLColor(color))
            
        elif isinstance(color,ndarray) and color.shape[-1] == 3:
            pass
        
        elif (type(color) == tuple or type(color) == list) and len(color) == 3:
            color = asarray(color)
        
        else:
            # The input should be compatible to a list of color compatible items.
            # An array with 3 colums will be fine.
            color = map(GLColor,color)
            color = asarray(color)

        #GD.debug(color)
        self.color = color

##### CURRENTLY WE DO NOT SET THE bkcolor
##        if bkcolor is None:
##            self.bkcolor = self.color
##        else:
##            self.bkcolor = resize(asarray(bkcolor),mprop)


    def setColorMap(self,colormap=None):
        """Set the color map for propcolor drawing.

        colormap should be a list of GLColors (RGB triplets).
        If none is given, the default colormap fom the current canvas
        is used.
        In either case, the colormap is extended to provide enough
        entries for all values in the property array.
        """
        if colormap is None:
            colormap = GD.canvas.settings.propcolors
        # make sure we have enough entries in the colormap
        mprop = max(self.propSet()) + 1
        self.colormap = resize(colormap,(mprop,3))
        #GD.debug("USING COLORMAP")
        #GD.debug(self.colormap)


    def setMarkSize(self,marksize):
        """Set the mark size.

        The mark size is currently only used with plex-1 Formices.
        """
#### DEFAULT MARK SIZE SHOULD BECOME A CANVAS SETTING!!
        
        if marksize is None:
            if self.eltype == 'point3d':
                # ! THIS SHOULD BE SET FROM THE SCENE SIZE
                #   RATHER THAN FORMEX SIZE 
                marksize = self.size() * float(markscale)
                if marksize <= 0.0:
                    marksize = 1.0
                self.setMark(marksize,"cube")
            else:
                marksize = 4.0 # default size 
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


    def setLineWidth(self,linewidth):
        """Set the line width.

        The line width is used in wireframe mode if plex > 1
        and in rendering mode if plex==2.
        If linewidth is None, the default canvas linewidth is used.
        """
        if linewidth is None:
            self.linewidth = None
        else:
            self.linewidth = float(linewidth)
        

    bbox = Formex.bbox


    def draw(self,mode='wireframe',color=None):
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
            self.draw(mode[:-4],color=color)
            self.draw('wireframe',color=black)
            return

        if color is None:
            color = self.color
        
        if isinstance(color,ndarray):
            #GD.debug(color.shape)
            pass
        
        if color is None:
            pass
        elif color == 'prop':
            color = self.colormap[self.p]
            #GD.debug("PROPCOLOR")
            #GD.debug(self.p.shape)
        else:
            color = asarray(color)
            if len(color.shape) == 1:
                GL.glColor3fv(color)
                color = None
            else:
                # color should be a full color array
                pass


        #GD.debug("Final color is %s" % str(color))
        ## !! WE SHOULD CHECK THE COLOR MORE THOROUGHLY !

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
            
        elif mode=='wireframe' :
            if self.eltype == 'tet':
                edges = [ 0,1, 0,2, 0,3, 1,2, 1,3, 2,3 ]
                coords = self.f[:,edges,:]
                drawEdges(coords,color)
            else:
                drawPolyLines(self.f,color)
                
        elif nnod == 3:
            drawTriangles(self.f,mode,color)
            
        elif nnod == 4:
            if self.eltype=='tet':
                faces = [ 0,1,2, 0,2,3, 0,3,1, 3,2,1 ]
                coords = self.f[:,faces,:]
                drawTriangles(coords,mode,color)
            else: # (possibly non-plane) quadrilateral
                drawQuadrilaterals(self.f,mode,color)
            
        else:
            drawPolygons(self.f,mode,color=None)


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

        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)

        if mode=='wireframe' :
            drawLineElems(self.nodes,self.edges,color)
        else:
            if color.ndim == 1:
                color = zeros((self.elems.shape[0],3))
                color[:,:] = self.color
            drawTriangles(self.nodes[self.elems],mode,color)

### End

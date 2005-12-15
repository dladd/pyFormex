# canvas.py
# $Id$
"""This implements an OpenGL drawing widget for painting 3D scenes."""
#
# TODO : we want to move the Qt dependencies as much as possible out of
#        this module
# TODO : we want to move the actual GL actors out of this module


import sys,math

import OpenGL.GL as GL
import OpenGL.GLU as GLU

import qt
import qtgl

from colors import *
from formex import *
from camera import *
from utils import stuur
import vector
   
def drawGrid(x1, y1, x2, y2, nx, ny):
    """Draw a rectangular grid of lines
        
    The rectangle has (x1,y1) and and (x2,y2) as opposite corners.
    There are (nx,ny) subdivisions along the (x,y)-axis. So the grid
    has (nx+1) * (ny+1) lines. nx=ny=1 draws a rectangle. nx=0 draws 1
    vertical line (at x1). nx=-1 gives only horizontal lines.
    
    """
    GL.glBegin(GL.GL_LINES)
    ix = range(nx+1)
    jx = [ nx-i for i in ix ]
    for i,j in zip(ix,jx):
        x = (i*x1+j*x2)/nx
        GL.glVertex2f(x, y1)
        GL.glVertex2f(x, y2)
        iy = range(ny+1)
        jy = [ ny-i for i in iy ]
        for i,j in zip(iy,jy):
            y = (i*y1+j*y2)/ny
            GL.glVertex2f(x1, y)
            GL.glVertex2f(x2, y)
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
#
# Actors are anything that can be drawn on an openGL canvas.
# An actor minimally needs two functions:
#   bbox() : to calculate the bounding box of the actor.
#   draw() : to actually draw the actor.

class CubeActor:
    """An OpenGL actor with cubic shape and 6 colored sides."""

    def __init__(self,size,color=[red,cyan,green,magenta,blue,yellow]):
        self.size = size
        self.color = color

    def bbox(self):
        return (0.5 * self.size) * array([[-1.,-1.,-1.],[1.,1.,1.]])

    def draw(self,wireframe=False):
        """Draw the cube."""
        drawCube(self.size,self.color)

class TriadeActor:
    """An OpenGL actor representing a triade of global axes."""

    def __init__(self,size,color=[red,green,blue,cyan,magenta,yellow]):
        self.size = size
        self.color = color

    def bbox(self):
        return (0.5 * self.size) * array([[0.,0.,0.],[1.,1.,1.]])

    def draw(self,wireframe=False):
        """Draw the cube."""
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


class FormexActor(Formex):
    """An OpenGL actor which is a Formex."""

    def __init__(self,F,color=black):
        Formex.__init__(self,F.data())
        self.list = None
        self.color = color
        if self.plexitude() == 1:
            self.setMark(self.size()/200,"cube")
        
    def draw(self,wireframe=True):
        """Draw the formex."""
        GL.glColor3f(*self.color)
        nnod = self.plexitude()
        if nnod == 2:
            GL.glBegin(GL.GL_LINES)
            for el in self.data():
                for nod in el:
                    GL.glVertex3f(*nod)
            GL.glEnd()
            
        elif nnod == 1:
            for el in self.data():
                GL.glPushMatrix()
                GL.glTranslatef (*el[0])
                GL.glCallList(self.mark)
                GL.glPopMatrix()
                
        elif wireframe:
            for el in self.data():
                GL.glBegin(GL.GL_LINE_LOOP)
                for nod in el:
                    GL.glVertex3f(*nod)
                GL.glEnd()
        elif nnod == 3:
            GL.glBegin(GL.GL_TRIANGLES)
            for el in self.data():
                for nod in el:
                    GL.glVertex3f(*nod)
            GL.glEnd()
        elif nnod == 4:
            GL.glBegin(GL.GL_QUADS)
            for el in self.data():
                for nod in el:
                    GL.glVertex3f(*nod)
            GL.glEnd()
        else:
            for el in self.data():
                GL.glBegin(GL.GL_POLYGON)
                for nod in el:
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

class CFormexActor(Formex):
    """An OpenGL actor which is a multicolored Formex.

    This is a variant of the FormexActor allowing for multiple colors. 
    """

    def __init__(self,F,colorset):
        """Create a multicolored Formex actor.

        The colors argument specifies a list of OpenGL colors for each
        of the property values in the Formex. If the list has less
        values that the PropSet, it is wrapped around.
        """
        Formex.__init__(self,F.data(),F.prop())
        self.list = None
        mprop = max(F.propSet()) + 1
        self.colorset = [ colorset[v % len(colorset)] for v in range(mprop) ]
        if self.plexitude() == 1:
            self.setMark(self.size()/200,"cube")

    # We need to further implement draw(). only nnod=2 done 
    def draw(self,wireframe=True):
        """Draw the formex."""
        nnod = self.plexitude()
        if nnod == 2:
            GL.glBegin(GL.GL_LINES)
            for i in range(self.nelems()):
                col = self.colorset[int(self.p[i])]
                GL.glColor3f(*(col))
                for nod in self[i]:
                    GL.glVertex3f(*nod)
            GL.glEnd()
            
        elif nnod == 1:
            for el in self.data():
                GL.glPushMatrix()
                GL.glTranslatef (*el[0])
                GL.glCallList(self.mark)
                GL.glPopMatrix()
                
        elif wireframe:
            for el in self.data():
                GL.glBegin(GL.GL_LINE_LOOP)
                for nod in el:
                    GL.glVertex3f(*nod)
                GL.glEnd()
        elif nnod == 3:
            GL.glBegin(GL.GL_TRIANGLES)
            for el in self.data():
                for nod in el:
                    GL.glVertex3f(*nod)
            GL.glEnd()
        elif nnod == 4:
            GL.glBegin(GL.GL_QUADS)
            for el in self.data():
                for nod in el:
                    GL.glVertex3f(*nod)
            GL.glEnd()
        else:
            for el in self.data():
                GL.glBegin(GL.GL_POLYGON)
                for nod in el:
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


class titleActor:
    """A viewport decoration showing a colorscale legend."""
    def __init__(self,title):
        """Create a title"""
        self.title = str(title)

    def draw(self):
        """draw the title."""
        
    

class colorLegendActor:
    """A viewport decoration showing a colorscale legend."""
    def __init__(self,colorscale):
        self.scale = colorscale


##################################################################
#
#  The Canvas
#
class Canvas(qtgl.QGLWidget):
    """A canvas for OpenGL rendering."""
    
    def __init__(self,w=640,h=480,*args):
        """Initialize an empty canvas with default settings.
        """
        qtgl.QGLWidget.__init__(self,*args)
        self.setFocusPolicy(qt.QWidget.StrongFocus)
        self.actors = []       # an empty scene
        self.views = { 'front': (0.,0.,0.),
                       'back': (180.,0.,0.),
                       'right': (90.,0.,0.),
                       'left': (270.,0.,0.),
                       'top': (0.,90.,0.),
                       'bottom': (0.,-90.,0.),
                       'iso': (45.,45.,0.),
                       }   # default views
        # angles are: longitude, latitude, twist
        self.setBbox()
        self.bgcolor = mediumgrey
        self.wireframe = True
        self.dynamic = None    # what action on mouse move
        self.makeCurrent()     # set GL context before creating the camera
        self.camera = Camera()
        
    # These three are defined by the qtgl API
    def initializeGL(self):
        self.glinit()

    def	resizeGL(self,w,h):
        self.resize(w,h)

    def	paintGL(self):
        self.display()

    # The rest are our functions

    # our own name for the canvas update function
    def update(self):
        self.updateGL()

# Do we use these??
##    def setColor(self,s):
##        """Set the OpenGL color to the named color"""
##        self.qglColor(qt.QColor(s))

##    def clearGLColor(self,s):
##        """Clear the OpenGL widget with the named background color"""
##        self.qglClearColor(qt.QColor(s))

    def glinit(self,mode="wireframe"):
	GL.glClearColor(*RGBA(self.bgcolor))# Clear The Background Color
	GL.glClearDepth(1.0)	       # Enables Clearing Of The Depth Buffer
	GL.glDepthFunc(GL.GL_LESS)	       # The Type Of Depth Test To Do
	GL.glEnable(GL.GL_DEPTH_TEST)	       # Enables Depth Testing
        if mode == "wireframe":
            self.wireframe = True
            GL.glShadeModel(GL.GL_FLAT)      # Enables Flat Color Shading
            GL.glDisable(GL.GL_LIGHTING)
        elif mode == "render":
            self.wireframe = False
            GL.glShadeModel(GL.GL_SMOOTH)    # Enables Smooth Color Shading
            #print "set up lights"
            GL.glLightModel(GL.GL_LIGHT_MODEL_AMBIENT,(0.5,0.5,0.5,1))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (1.0, 1.0, 1.0, 1.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (-1.0, -1.0, 5.0))
            GL.glEnable(GL.GL_LIGHT0)
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, (1.0, 1.0, 1.0))
            GL.glEnable(GL.GL_LIGHT1)
            GL.glEnable(GL.GL_LIGHTING)
            #print "set up materials"
            GL.glEnable(GL.GL_COLOR_MATERIAL)
            GL.glColorMaterial ( GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE )

    def setLinewidth(self,lw):
        """Set the linewidth for line rendering."""
        GL.glLineWidth (lw)

    def setBbox(self,bbox=None):
        """Set the bounding box of the scene you want to be visible."""
        # TEST: use last actor
        if bbox:
            self.bbox = bbox
        else:
            if len(self.actors) > 0:
                self.bbox = self.actors[-1].bbox()
            else:
                self.bbox = [[-1.,-1.,-1.],[1.,1.,1.]]
        #print "canvas.bbox=",self.bbox
         
    def addActor(self,actor):
        """Add an actor to the scene."""
        self.makeCurrent()
        actor.list = GL.glGenLists(1)
        GL.glNewList(actor.list,GL.GL_COMPILE)
        actor.draw(self.wireframe)
        GL.glEndList()
        self.actors.append(actor)

    def removeActor(self,actor):
        """Remove an actor from the scene"""
        self.makeCurrent()
        self.actors.remove(actor)
        GL.glDeleteLists(actor.list,1)

    def removeAllActors(self):
        """Remove all actors from the scene"""
        for a in self.actors:
            GL.glDeleteLists(a.list,1)
        self.actors = []
        self.setBbox()

    def recreateActor(self,actor):
        """Recreate an actor in the scene"""
        self.removeActor(actor)
        self.addActor(actor) 

    def redrawActors(self,actorlist=None):
        """Redraw (some) actors in the scene.

        This redraws the specified actors (recreating their display list).
        This should e.g. be used after changing an actor's properties.
        Only actors that are in the current actor list will be redrawn.
        If no actor list is specified, the whole current actorlist is redrawn.
        """
        self.makeCurrent()
        if actorlist == None:
            actorlist = self.actors
        for actor in actorlist:
            if actor.list:
                GL.glDeleteLists(actor.list,1)
            actor.list = GL.glGenLists(1)
            GL.glNewList(actor.list,GL.GL_COMPILE)
            actor.draw(self.wireframe)
            GL.glEndList() 
        self.display()

    def redrawAll(self):
        """Redraw all actors in the scene."""
        self.redrawActors(self.actors)

    def clear(self):
        self.makeCurrent()
	GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
	GL.glClearColor(*RGBA(self.bgcolor))   # Clear The Background Color

    def display(self):
        """(Re)display all the actors in the scene.

        This should e.g. be used when actors are added to the scene,
        or after changing  camera position or lens.
        """
        self.makeCurrent()
	GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
	GL.glClearColor(*RGBA(self.bgcolor))   # Clear The Background Color
        self.camera.loadProjection()
        self.camera.loadMatrix()
        for i in self.actors:
            GL.glCallList(i.list)
        #GL.glMatrixMode (GL.GL_PROJECTION)
        #GL.glLoadIdentity()
        #GLU.gluOrtho2D (0, self.width(), 0, self.height())
        #drawGrid(10,10,100,100,9,9)

        
    def resize (self,w,h):
        self.makeCurrent()
	if h == 0:	# Prevent A Divide By Zero If The Window Is Too Small 
            h = 1
	GL.glViewport(0, 0, w, h)
        self.aspect = float(w)/h
        self.camera.setLens(aspect=self.aspect)
        self.display()

    def createView(self,name,angles):
        """Create a named view for camera orientation long,lat.

        By default, the following views are created:
        'front', 'back', 'left', 'right', 'bottom', 'top', 'iso'.
        The user can add/delete/overwrite any number of predefined views.
        """
        self.views[name] = angles
        
    def useView(self,bbox=None,side='front'):
        """Sets the camera looking from one of the named views.

        On startup, the predefined views are 'front', 'back', 'left',
        'right', 'top', 'bottom' and 'iso'.
        This function does not only set the camera angles, but also
        adjust the zooming.
        If a bbox is specified, the camera will be zoomed to view the
        whole bbox.
        If no bbox is specified, the current scene bbox will be used.
        If no current bbox has been set, it will be calculated as the
        bbox of the whole scene.
        """
        self.makeCurrent()
        # select view angles: if undefined use (0,0,0)
        angles = self.views.get(side,(0,0,0))
        # go to a distance to have a good view with a 45 degree angle lens
        if bbox == None:
            bbox = self.bbox
        else:
            self.bbox = bbox
        center,size = centerDiff(*bbox)
        #print "Setting view for bbox",bbox
        #print "center=",center
        #print "size=",size
        # calculating the bounding circle: this is rather conservative
        dist = length(size)
        #print "dist = ",dist
        self.camera.setCenter(*center)
        self.camera.setRotation(*angles)
        self.camera.setDist(dist)
        self.camera.setLens(45.,self.aspect)
        self.camera.setClip(0.01*dist,100*dist)

    def zoom(self,f):
        self.camera.setDist(f*self.camera.getDist())

    def dyna(self,x,y):
        """Perform dynamic zoom/pan/rotation functions"""
        w,h = self.width(),self.height()
        if self.dynamic == "trirotate":
            # set all three rotations from mouse movement
            # tangential movement sets twist,
            # but only if initial vector is big enough
            x0 = self.state        # initial vector
            d = vector.length(x0)
            if d > h/8:
                x1 = [x-w/2, h/2-y, 0]     # new vector
                a0 = math.atan2(x0[0],x0[1])
                a1 = math.atan2(x1[0],x1[1])
                an = (a1-a0) / math.pi * 180
                ds = stuur(d,[-h/4,h/8,h/4],[-1,0,1],2)
                twist = - an*ds
                #print "an,d,ds = ",an,d,ds,twist
                self.camera.rotate(twist,0.,0.,1.)
                self.state = x1
            # radial movement rotates around vector in lens plane
            x0 = [self.statex-w/2, h/2-self.statey, 0]    # initial vector
            dx = [x-self.statex, self.statey-y,0]         # movement
            b = vector.projection(dx,x0)
            #print "x0,dx,b=",x0,dx,b
            if abs(b) > 5:
                val = stuur(b,[-2*h,0,2*h],[-180,0,+180],1)
                rot =  [ abs(val),-dx[1],dx[0],0 ]
                #print "val,rot=",val,rot
                self.camera.rotate(*rot)
                self.statex,self.statey = (x,y)

        elif self.dynamic == "pan":
            dist = self.camera.getDist() * 0.5
            # hor movement sets x value of center
            # vert movement sets y value of center
            #panx = stuur(x,[0,self.statex,w],[-dist,0.,+dist],1.0)
            #pany = stuur(y,[0,self.statey,h],[-dist,0.,+dist],1.0)
            #self.camera.setCenter (self.state[0] - panx, self.state[1] + pany, self.state[2])
            dx,dy = (x-self.statex,y-self.statey)
            panx = stuur(dx,[-w,0,w],[-dist,0.,+dist],1.0)
            pany = stuur(dy,[-h,0,h],[-dist,0.,+dist],1.0)
            #print dx,dy,panx,pany
            self.camera.translate(panx,-pany,0)
            self.statex,self.statey = (x,y)

        elif self.dynamic == "zoom":
            # hor movement is lens zooming
            f = stuur(x,[0,self.statex,w],[180,self.statef,0],1.2)
            self.camera.setLens(f)

        elif self.dynamic == "combizoom":
            # hor movement is lens zooming
            f = stuur(x,[0,self.statex,w],[180,self.state[1],0],1.2)
            self.camera.setLens(f)
            # vert movement is dolly zooming
            d = stuur(y,[0,self.statey,h],[0.2,1,5],1.2)
            self.camera.setDist(d*self.state[0])
        self.update()


    # Any keypress with focus in the canvas generates a 'wakeup' signal.
    # This is used to break out of a wait status.
    # An 's' keypress will generate a 'save' signal.
    # Events not handled here could also be handled by the toplevel
    # event handler.
    def keyPressEvent (self,e):
        self.emit(qt.PYSIGNAL("wakeup"),())
        if e.text() == 's':
            self.emit(qt.PYSIGNAL("save"),())
        e.ignore()
        
    def mousePressEvent(self,e):
        # Remember the place of the click
        self.statex = e.x()
        self.statey = e.y()
        self.camera.loadMatrix()
        # Other initialisations for the mouse move actions are done here 
        if e.button() == qt.Qt.LeftButton:
            self.dynamic = "trirotate"
            # the vector from the screen center to the clicked point
            # this is used for the twist angle
            self.state = [self.statex-self.width()/2, -(self.statey-self.height()/2), 0.]
        elif e.button() == qt.Qt.MidButton:
            self.dynamic = "pan"
            self.state = self.camera.getCenter()
        elif e.button() == qt.Qt.RightButton:
            self.dynamic = "combizoom"
            self.state = [self.camera.getDist(),self.camera.fovy]
        
    def mouseReleaseEvent(self,e):
        self.dynamic = None
        self.camera.saveMatrix()          
        
    def mouseMoveEvent(self,e):
        if self.dynamic:
            self.dyna(e.x(),e.y())

    def save(self,fn,fmt='PNG'):
        """Save the current rendering as an image file."""
        self.makeCurrent()
        GL.glFinish()
        qim = self.grabFrameBuffer()
        qim.save(fn,fmt)

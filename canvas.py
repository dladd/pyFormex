# canvas.py
# $Id$
##
## This file is part of pyFormex 0.2 Release Mon Jan  3 14:54:38 2005
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Copyright (C) 2004 Benedict Verhegghe (benedict.verhegghe@ugent.be)
## Copyright (C) 2004 Bart Desloovere (bart.desloovere@telenet.be)
## Distributed under the General Public License, see file COPYING for details
##
#
#
# This implements an OpenGL drawing widget for painting 3D scenes.
#
# TODO : we want to move the Qt dependencies as much as possible out of
#        this module
# TODO : we want to move the actual GL actors out of this module

"""This implements an OpenGL drawing widget"""

import sys,math

import OpenGL.GL as GL
import OpenGL.GLU as GLU

import qt
import qtgl

from colors import *
from formex import *
from camera import *
import vector

def stuur(x,xval,yval,exp=2.5):
    """Returns a nonlinear response on the input x.

    xval and yval should be lists of 3 values: [xmin,x0,xmax], [ymin,y0,yma].
    Together with the exponent exp, they define the response curve as function
    of x. With an exponent > 0, the variation will be slow in the neighbourhood
    of (x0,y0). For values x < xmin or x > xmax, the limit value ymin or ymax
    is returned.
    """
    xmin,x0,xmax = xval
    ymin,y0,ymax = yval 
    if x < xmin:
        return ymin
    elif x < x0:
        xr = float(x-x0) / (xmin-x0)
        return y0 + (ymin-y0) * xr**exp
    elif x < xmax:
        xr = float(x-x0) / (xmax-x0)
        return y0 + (ymax-y0) * xr**exp
    else:
        return ymax

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


class CubeActor:
    """An OpenGL actor with cubic shape and 6 colored sides."""

    def __init__(self,size,color=[red,cyan,green,magenta,blue,yellow]):
        self.size = size
        self.color = color

    def display(self):
        """Draw the cube."""
        GLU.gluCube(self.size,self.color)


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

##################################################################
#
#  The Canvas
#
class Canvas(qtgl.QGLWidget):
    """A canvas for OpenGL rendering."""
    
    def __init__(self,w=640,h=480,*args):
        self.actors = []
        self.camera = Camera() # default Camera settings are adequate
        self.dynamic = None    # what action on mouse move
        self.trl = [0.0,0.0,0.0]  # model
        self.rot = [0.0,0.0,0.0,1.0]
        qtgl.QGLWidget.__init__(self,*args)
        self.setFocusPolicy(qt.QWidget.StrongFocus)
        self.resize(w,h)
        self.glinit("wireframe") # default mode is wireframe

    # These three are defined by the qtgl API
    def initializeGL(self):
        """Set up the OpenGL rendering state, and define display list"""
        self.glinit()
    def repaintGL(self):
        self.display()
    def	resizeGL(self,w,h):
        """Set up the OpenGL view port, matrix mode, etc.

        This will get called automatically on creating the QGLWidget!
        """
        self.resize(w,h)

    # The rest are our functions
    def setGLColor(self,s):
        """Set the OpenGL color to the named color"""
        self.qglColor(qtgl.QColor(s))

    def clearGLColor(self,s):
        """Clear the OpenGL widget with the named background color"""
        self.qglClearColor(qtgl.QColor(s))

    def glinit(self,mode="wireframe"):
	GL.glClearColor(*RGBA(mediumgrey))# Clear The Background Color
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
        GL.glFinish()
        #print "set up camera"
	self.camera.loadProjection()
        self.loadModelMatrix()

    def loadModelMatrix(self):
        """Load the model transformation matrix"""
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glTranslatef(*self.trl)
        GL.glRotatef(*self.rot)

##    def setViewingVolume(self,bbox):
##        print "bbox=",bbox
##        x0,y0,z0 = bbox[0]
##        x1,y1,z1 = bbox[1]
##        corners = [[x,y,z] for x in [x0,x1] for y in [y0,y1] for z in [z0,z1]]
##        #self.camera.lookAt(array(corners))
##        #print self.camera.center,self.camera.eye
##        #self.camera.setProjection()

         
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

    def recreateActor(self,actor):
        """Recreate an actor in the scene"""
        self.removeActor(actor)
        self.addActor(actor) 

    def redrawAll(self):
        """Redraw all actors in the scene.

        This is different from display() in that it recreates
        each actors display list.
        This should e.g. be used after changing drawing modes.
        """
        self.makeCurrent()
        for actor in self.actors:
            if actor.list:
                GL.glDeleteLists(actor.list,1)
            actor.list = GL.glGenLists(1)
            GL.glNewList(actor.list,GL.GL_COMPILE)
            actor.draw(self.wireframe)
            GL.glEndList() 
        self.display()

    def clear(self):
        self.makeCurrent()
	GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
	GL.glClearColor(*RGBA(lightgrey))   # Clear The Background Color
        self.updateGL()

    def display(self):
        """(Re)display all the actors in the scene.

        This should e.g. be used when actors are added to the scene,
        or after changing  camera position or lens.
        """
        self.makeCurrent()
	GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
	GL.glClearColor(*RGBA(lightgrey))   # Clear The Background Color
        self.camera.loadProjection()
        GL.glLoadIdentity()
        self.camera.loadMatrix()
        for i in self.actors:
            GL.glCallList(i.list)
        self.updateGL()
        
    def resize (self,w,h):
        self.makeCurrent()
	if h == 0:	# Prevent A Divide By Zero If The Window Is Too Small 
            h = 1
	GL.glViewport(0, 0, w, h)
        self.aspect = float(w)/h
        self.camera.setLens(aspect=self.aspect)
        self.display()

    def setView(self,bbox,side='front'):
        """Sets the camera looking at one of the sides of the bbox"""
        self.makeCurrent()
        center = (bbox[0]+bbox[1])/2
        self.camera.setCenter(*center)
        size = bbox[1]-bbox[0]
        # print "size=",size
        if side == 'front':
            hsize,vsize,depth = size[0],size[1],size[2]
            long,lat = 0.,0.
        elif side == 'back':
            hsize,vsize,depth = size[0],size[1],size[2]
            long,lat = 180.,0.
        elif side == 'right':
            hsize,vsize,depth = size[2],size[1],size[0]
            long,lat = 90.,0.
        elif side == 'left':
            hsize,vsize,depth = size[2],size[1],size[0]
            long,lat = 270.,0.
        elif side == 'top':
            hsize,vsize,depth = size[0],size[2],size[1]
            long,lat = 0.,90.
        elif side == 'bottom':
            hsize,vsize,depth = size[0],size[2],size[1]
            long,lat = 0.,-90.
        elif side == 'iso':
            hsize = vsize = depth = vector.distance(bbox[1],bbox[0])
            long,lat = 45.,45.
        # go to a distance to have a good view with a 45 degree angle lens
        dist = max(0.6*depth, 1.5*max(hsize/self.aspect,vsize))
        self.camera.setEye(long,lat,dist)
        self.camera.setLens(45.,self.aspect)
        self.camera.setClip(0.1*dist,10*dist)
        self.camera.loadProjection()


    def dyna(self,x,y):
        """Perform dynamic zoom/pan/rotation functions"""
        w,h = self.width(),self.height()
        if self.dynamic == "zoom":
            # hor movement is lens zooming
            f = stuur(x,[0,self.statex,w],[180,self.statef,0],1.5)
            self.camera.setLens(f)
            self.display()
        elif self.dynamic == "combizoom":
            # hor movement is lens zooming
            f = stuur(x,[0,self.statex,w],[180,self.state[1],0],1.5)
            self.camera.setLens(f)
            # vert movement is dolly zooming
            d = stuur(y,[0,self.statey,h],[0.1,1,10],1.5)
            self.camera.setDistance(d*self.state[0])
            self.display()
        elif self.dynamic == "rotate":
            # hor movement sets azimuth
            a = stuur(x,[0,self.statex,w],[-360,0,+360],1.5)
            # vert movement sets elevation
            e = stuur(y,[0,self.statey,h],[-180,0,+180],1.5)
            self.camera.setDirection(self.state[0] - a,self.state[1] + e)
            self.display()
        elif self.dynamic == "trirotate":
            cx,cy = w/2,h/2
            # hor movement sets azimuth
            a = stuur(x,[0,self.statex,w],[-360,0,+360],1.5)
            # vert movement sets elevation
            e = stuur(y,[0,self.statey,h],[-180,0,+180],1.5)
            self.camera.setDirection(self.state[0] - a,self.state[1] + e)
            self.display()
        elif self.dynamic == "pan":
            dist = self.camera.getDistance()
            # hor movement sets x value of center
            panx = stuur(x,[0,self.statex,w],[-dist,0.,+dist],1.0)
            # vert movement sets y value of center
            pany = stuur(y,[0,self.statey,h],[-dist,0.,+dist],1.0)
            self.camera.ctr[0] = self.state[0] - panx
            self.camera.ctr[1] = self.state[1] + pany
            self.display()

    ## Een ENTER keypress in het teken canvas stuurt een wakeup event.
    ## Dit kan bijvoorbeeld gebruikt worden om een wait af te breken.
    ## Wij zouden dit ook in de top-event handler kunnen doen voorzover
    ## wij de events hier niet afhandelen.
    def keyPressEvent (self,e):
        self.emit(qt.PYSIGNAL("wakeup"),())
        e.ignore()
        
    def mousePressEvent(self,e):
        self.statex = e.x()
        self.statey = e.y()
        if e.button() == qt.Qt.LeftButton:
            self.dynamic = "trirotate"
            self.state = self.camera.getEye()
        elif e.button() == qt.Qt.MidButton:
            self.dynamic = "pan"
            self.state = self.camera.getCenter()
        elif e.button() == qt.Qt.RightButton:
            self.dynamic = "combizoom"
            self.state = [self.camera.getDistance(),self.camera.fovy]
        
    def mouseReleaseEvent(self,e):
        self.dynamic = None
        
    def mouseMoveEvent(self,e):
        if self.dynamic:
            self.dyna(e.x(),e.y())

    def save(self,fn,fmt='PNG'):
        """Save the current rendering as an image file."""
        self.makeCurrent()
        GL.glFinish()
        qim = self.grabFrameBuffer()
        qim.save(fn,fmt)

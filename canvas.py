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

    xval and yval should be lists of 3 values: [xmin,x0,xmax], [ymin,y0,ymax].
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
        qtgl.QGLWidget.__init__(self,*args)
        self.setFocusPolicy(qt.QWidget.StrongFocus)
        self.actors = []       # an empty scene
        self.setBbox()
        self.wireframe = True
        self.dynamic = None    # what action on mouse move
        self.rot = [ 0.,1.,0.,0.]
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

    def setColor(self,s):
        """Set the OpenGL color to the named color"""
        self.qglColor(qt.QColor(s))

    def clearGLColor(self,s):
        """Clear the OpenGL widget with the named background color"""
        self.qglClearColor(qt.QColor(s))

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

    def applyModelMatrix(self):
        """Apply the model transformation matrix on top of existing matrix"""
        print "Camera data'",self.camera.data
        print "Extra rotation is ",self.rot
        tr=self.camera.getCenter()
        GL.glTranslatef(tr[0],tr[1],tr[2])
        GL.glRotatef(*self.rot)
##        GL.glRotatef(self.rot[2], 0.0, 0.0, 1.0)
##        GL.glRotatef(self.rot[0], 1.0, 0.0, 0.0)
##        GL.glRotatef(self.rot[1], 0.0, 1.0, 0.0)
        GL.glTranslatef(-tr[0],-tr[1],-tr[2])

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
        print "canvas.bbox=",self.bbox
         
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

    def display(self):
        """(Re)display all the actors in the scene.

        This should e.g. be used when actors are added to the scene,
        or after changing  camera position or lens.
        """
        self.makeCurrent()
	GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
	GL.glClearColor(*RGBA(lightgrey))   # Clear The Background Color
        self.camera.loadProjection()
        self.camera.loadMatrix()
##        self.applyModelMatrix()
        for i in self.actors:
            GL.glCallList(i.list)
        
    def resize (self,w,h):
        self.makeCurrent()
	if h == 0:	# Prevent A Divide By Zero If The Window Is Too Small 
            h = 1
	GL.glViewport(0, 0, w, h)
        self.aspect = float(w)/h
        self.camera.setLens(aspect=self.aspect)
        self.display()

    def setView(self,bbox=None,side='front'):
        """Sets the camera looking at one of the sides of the bbox.

        If a bbox is specified, 
        If no bbox is specified, the current scene bbox will be used.
        If no current bbox has been set, it will be calculated as the
        bbox of the whole scene.
        The view side can be one of the predefined sides. If none is
        given, 'front' is used.
        """
        self.makeCurrent()
        if bbox == None:
            bbox = self.bbox
        else:
            self.bbox = bbox
        center,size = centerDiff(*bbox)
        print "Setting view for bbox",bbox
        print "center=",center
        print "size=",size
        self.camera.setCenter(*center)
        if side == 'front':
            hsize,vsize,depth = size[0],size[1],size[2]
            lat,long = 0.,0.
        elif side == 'back':
            hsize,vsize,depth = size[0],size[1],size[2]
            lat,long = 180.,0.
        elif side == 'right':
            hsize,vsize,depth = size[2],size[1],size[0]
            lat,long = 90.,0.
        elif side == 'left':
            hsize,vsize,depth = size[2],size[1],size[0]
            lat,long = 270.,0.
        elif side == 'top':
            hsize,vsize,depth = size[0],size[2],size[1]
            lat,long = 0.,90.
        elif side == 'bottom':
            hsize,vsize,depth = size[0],size[2],size[1]
            lat,long = 0.,-90.
        elif side == 'iso':
            hsize = vsize = depth = vector.distance(bbox[1],bbox[0])
            lat,long = 45.,45.
        # go to a distance to have a good view with a 45 degree angle lens
        dist = max(0.6*depth, 1.5*max(hsize/self.aspect,vsize))
        print "dist = ",dist
        self.camera.setEye(lat,long,dist)
        self.camera.setTwist(0.)
        self.camera.setLens(45.,self.aspect)
        self.camera.setClip(0.01*dist,100*dist)


    def zoom(self,f):
        self.camera.setDistance(f*self.camera.getDist())

    def dyna(self,x,y):
        """Perform dynamic zoom/pan/rotation functions"""
        w,h = self.width(),self.height()
        if self.dynamic == "rotate":
            # hor movement sets azimuth
            a = stuur(x,[0,self.statex,w],[-360,0,+360],1.5)
            # vert movement sets elevation
            e = stuur(y,[0,self.statey,h],[-180,0,+180],1.5)
            self.camera.setDirection(self.state[0] - a,self.state[1] + e)
        elif self.dynamic == "trirotate":
            # set all three rotations from mouse movement
            x0 = [self.statex-w/2, self.statey-h/2,0] # initial vector
            dx = [x-self.statex, y-self.statey,0]     # movement
            x1 = [x-w/2, y-h/2,0]           # new vector
            a0 = math.atan2(x0[0],x0[1])
            a1 = math.atan2(x1[0],x1[1])
            an = (a1-a0) / math.pi * 180
            # tangential movement set twist
            # but not if to close to center
##            d = vector.length(x1)
##            if d < h/8:
##                ds = 0
##            else:
##                ds = stuur(d,[-h/4,h/8,h/4],[-1,0,1],2)
##            print "ds = ",ds
##            self.camera.setTwist(self.state[2] - an*ds)
            # radial movement rotates around vector in lens plane
            b = -vector.projection(dx,x0)
            self.rot[0] = stuur(b,[-2*h,0,2*h],[-360,0,+360],1.5)
            self.rot[1:4] = self.state[0:3]
        elif self.dynamic == "pan":
            dist = self.camera.getDistance() * 0.5
            # hor movement sets x value of center
            # vert movement sets y value of center
            panx = stuur(x,[0,self.statex,w],[-dist,0.,+dist],1.0)
            pany = stuur(y,[0,self.statey,h],[-dist,0.,+dist],1.0)
            self.camera.setCenter (self.state[0] - panx, self.state[1] + pany, self.state[2])
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
            self.camera.setDistance(d*self.state[0])
        self.update()


    ## Een ENTER keypress in het teken canvas stuurt een wakeup event.
    ## Dit kan bijvoorbeeld gebruikt worden om een wait af te breken.
    ## Wij zouden dit ook in de top-event handler kunnen doen voorzover
    ## wij de events hier niet afhandelen.
    def keyPressEvent (self,e):
        self.emit(qt.PYSIGNAL("wakeup"),())
        e.ignore()
        
    def mousePressEvent(self,e):
        # Remember the place of the click
        self.statex = e.x()
        self.statey = e.y()
        self.camera.loadMatrix()
        self.rot = [ 0.,1.,0.,0.]
        # Other initialisations for the mouse move actions are done here 
        if e.button() == qt.Qt.LeftButton:
            self.dynamic = "trirotate"
            # the vector from the screen center to the clicked point
            print "x,y=",self.statex,self.statey
            v = [self.statex-self.width()/2, -(self.statey-self.height()/2), 0.]
            print "v=",v
            # rotate vector 90 degrees around z-axis
            v = [ v[1], -v[0], v[2] ]
            print "vrot=",v
            dist = vector.length(v)
            nv = self.camera.toWorld(v)
            self.state = nv + [ dist ]
            print "State:",nv,dist,self.state
        elif e.button() == qt.Qt.MidButton:
            self.dynamic = "pan"
            self.state = self.camera.getCenter()
        elif e.button() == qt.Qt.RightButton:
            self.dynamic = "combizoom"
            self.state = [self.camera.getDistance(),self.camera.fovy]
        
    def mouseReleaseEvent(self,e):
        if self.dynamic == "trirotate":
            self.camera.saveMatrix()          
            self.rot = [ 0.,1.,0.,0.]
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

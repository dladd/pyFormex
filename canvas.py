# canvas.py
# $Id$
##
## This file is part of pyformex 0.1.2 Release Fri Jul  9 14:48:57 2004
## pyformex is a python implementation of Formex algebra
## (c) 2004 Benedict Verhegghe (email: benedict.verhegghe@ugent.be)
## Releases can be found at ftp://mecatrix.ugent.be/pub/pyformex/
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

from qt import *
from qtgl import *

import sys,math

from OpenGL.GL import *
from OpenGL.GLU import *

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


class FormexActor(Formex):
    """An OpenGL actor which is a Formex"""

    def __init__(self,F,color=black):
        Formex.__init__(self,F.data())
        self.color = color
        
    def display(self,wireframe=True):
        """Draw a formex.

        """
        glColor3f(*self.color)
        nnod = self.plexitude()
        if nnod == 2:
            glBegin(GL_LINES)
            for el in self.data():
                for nod in el:
                    glVertex3f(*nod)
            glEnd()
        elif nnod == 1:
            radius = self.size() / 100
            slices = 16
            stacks = 16
            quad = gluNewQuadric()
            gluQuadricNormals(quad, GLU_SMOOTH) # Create Smooth Normals
            for el in self.data():
                glPushMatrix()
                glTranslatef (*el[0])
                gluSphere(quad,radius,slices,stacks)
                glPopMatrix()
                
        elif wireframe:
            for el in self.data():
                glBegin(GL_LINE_LOOP)
                for nod in el:
                    glVertex3f(*nod)
                glEnd()
        elif nnod == 3:
            print "Triangles"
            glBegin(GL_TRIANGLES)
            for el in self.data():
                for nod in el:
                    glVertex3f(*nod)
            glEnd()
        elif nnod == 4:
            glBegin(GL_QUADS)
            for el in self.data():
                for nod in el:
                    glVertex3f(*nod)
            glEnd()
        else:
            for el in self.data():
                glBegin(GL_POLYGON)
                for nod in el:
                    glVertex3f(*nod)
                glEnd()
                                


class Canvas(QGLWidget):
    """A canvas for OpenGL rendering."""
    
    def __init__(self,w=640,h=480,*args):
        self.actors = []
        self.camera = Camera() # default Camera settings are adequate
        self.dynamic = None    # what action on mouse move
        QGLWidget.__init__(self,*args)
        self.resize(w,h)
        self.wireframe = True  # default mode is wireframe 
        self.glinit()

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

    def setGLColor(self,s):
        """Set the OpenGL color to the named color"""
        self.qglColor(QColor(s))

    def clearGLColor(self,s):
        """Clear the OpenGL widget with the named background color"""
        self.qglClearColor(QColor(s))

    def glinit(self,mode="wireframe"):
	glClearColor(*RGBA(mediumgrey))# Clear The Background Color
	glClearDepth(1.0)	       # Enables Clearing Of The Depth Buffer
	glDepthFunc(GL_LESS)	       # The Type Of Depth Test To Do
	glEnable(GL_DEPTH_TEST)	       # Enables Depth Testing
        if mode == "wireframe":
            self.wireframe = True
            glShadeModel(GL_FLAT)      # Enables Flat Color Shading
        elif mode == "render":
            self.wireframe = False
            glShadeModel(GL_SMOOTH)    # Enables Smooth Color Shading

##        #print "set up lights"
##	glLightfv(GL_LIGHT0, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
##	glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
##	glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
##	glLightfv(GL_LIGHT0, GL_POSITION, (200.0, 200.0, 0.0))
##	glEnable(GL_LIGHT0)
##	glLightfv(GL_LIGHT1, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
##	glLightfv(GL_LIGHT1, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
##	glLightfv(GL_LIGHT1, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
##	glLightfv(GL_LIGHT1, GL_POSITION, (-500.0, 100.0, 0.0))
##	glEnable(GL_LIGHT1)
##        glEnable(GL_LIGHTING)
        
##        #print "set up materials"
##        glEnable ( GL_COLOR_MATERIAL )
##        glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE )
##        glMaterial(GL_FRONT, GL_SPECULAR, (0,0,0,1));
##        glMaterial(GL_FRONT, GL_EMISSION, (0,0,0,1));
        glFinish()

        #print "set up camera"
	self.camera.loadProjection()

##    def setViewingVolume(self,bbox):
##        print "bbox=",bbox
##        x0,y0,z0 = bbox[0]
##        x1,y1,z1 = bbox[1]
##        corners = [[x,y,z] for x in [x0,x1] for y in [y0,y1] for z in [z0,z1]]
##        #self.camera.lookAt(array(corners))
##        #print self.camera.center,self.camera.eye
##        #self.camera.setProjection()
         
    def addActor(self,actor):
        """Add an actor to the scene

        All an actor needs is a display function
        """
        self.makeCurrent()
        list = glGenLists(1)
        glNewList(list,GL_COMPILE)
        actor.display(self.wireframe)
        glEndList ()
        self.actors.append(list)
        actor.list = list

    def removeActor(self,actor):
        """Remove an actor from the scene"""
        self.makeCurrent()
        self.actors.remove(actor.list)
        glDeleteLists(actor.list,1)

    def removeAllActors(self):
        """Remove all actors from the scene"""
        for list in self.actors:
            glDeleteLists(list,1)
        self.actors = []

    def recreateActor(self,actor):
        """Recreate an actor in the scene"""
        self.removeActor(actor)
        self.addActor(actor) 

    def clear(self):
        self.makeCurrent()
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glClearColor(*RGBA(lightgrey))   # Clear The Background Color
        self.updateGL()

    def display(self):
        self.makeCurrent()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor3f(*black)
        self.camera.loadProjection()
        glLoadIdentity()
        self.camera.loadMatrix()
        for i in self.actors:
            glCallList(i)
        self.updateGL()
        
    def resize (self,w,h):
        self.makeCurrent()
	if h == 0:	# Prevent A Divide By Zero If The Window Is Too Small 
            h = 1
	glViewport(0, 0, w, h)
        self.aspect = float(w)/h
        self.camera.setLens(aspect=self.aspect)
        self.display()

    def setView(self,bbox,side='front'):
        """Sets the camera looking at one of the sides of the bbox"""
        self.makeCurrent()
        center = (bbox[0]+bbox[1])/2
        self.camera.setCenter(*center)
        size = bbox[1]-bbox[0]
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
        self.camera.setPos(long,lat,dist)
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
            self.camera.pos[0] = self.state[0] - a
            self.camera.pos[1] = self.state[1] + e
            self.display()
        elif self.dynamic == "pan":
            # hor movement sets azimuth
            a = stuur(x,[0,self.statex,w],[-360,0,+360],1.5)
            # vert movement sets elevation
            e = stuur(y,[0,self.statey,h],[-180,0,+180],1.5)
            self.camera.pos[0] = self.state[0] - a
            self.camera.pos[1] = self.state[1] + e
            self.display()
        

    def mouseMoveEvent(self,e):
        if self.dynamic:
            self.dyna(e.x(),e.y())
        
    def mousePressEvent(self,e):
        self.statex = e.x()
        self.statey = e.y()
        if e.button() == Qt.LeftButton:
            self.dynamic = "rotate"
            self.state = self.camera.pos[0:2]
        elif e.button() == Qt.MidButton:
            self.dynamic = "pan"
        elif e.button() == Qt.RightButton:
            self.dynamic = "combizoom"
            self.state = [self.camera.distance(),self.camera.fovy]
        
    def mouseReleaseEvent(self,e):
        self.dynamic = None

    def save(self,fn,fmt='PNG'):
        """Save the current rendering as an image file."""
        self.makeCurrent()
        glFinish()
        qim = self.grabFrameBuffer()
        qim.save(fn,fmt)

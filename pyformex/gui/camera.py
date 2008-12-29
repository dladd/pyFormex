#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Mon Dec 29 15:32:01 2008
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
"""camera 0.1 (C) Benedict Verhegghe"""

import sys

from coords import tand
#from formex import Formex

import numpy
import distutils.version
Version=distutils.version.LooseVersion
if Version(numpy.__version__) < Version('0.9.8'):
    inverse = numpy.linalg.inverse
else:
    inverse = numpy.linalg.linalg.inv
multiply = numpy.dot
array = numpy.array

import copy

import OpenGL.GL as GL
import OpenGL.GLU as GLU

def printModelviewMatrix(s="%s"):
    print s % GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX)



class ViewAngles(dict):
    """A dict to keep named camera angle settings.

    This class keeps a dictionary of named angle settings. Each value is
    a tuple of (longitude, latitude, twist) camera angles.
    This is a static class which should not need to be instantiated.

    There are seven predefined values: six for looking along global
    coordinate axes, one isometric view.
    """

    def __init__(self,data = { 'front': (0.,0.,0.),
                          'back': (180.,0.,0.),
                          'right': (90.,0.,0.),
                          'left': (270.,0.,0.),
                          'top': (0.,90.,0.),
                          'bottom': (0.,-90.,0.),
                          'iso': (45.,45.,0.),
                          }):
        dict.__init__(self,data)
        

    def get(self,name):
        """Get the angles for a named view.

        Returns a tuple of angles (longitude, latitude, twist) if the
        named view was defined, or None otherwise
        """
        return dict.get(self,name,None)


view_angles = ViewAngles()


## ! For developers: the information in this module is not fully correct
## ! We now store the rotation of the camera as a combined rotation matrix,
##   not by the individual rotation angles.

class Camera:
    """This class defines a camera for OpenGL rendering.

    It provides functions for manipulating the camera position, the viewing
    direction and the lens parameters.

    The camera viewing line can be defined by two points : the position of
    the camera and the center of the scene the camera is looking at.
    To enable continuous camera rotations however, it is essential that the
    camera angles are stored as such, and not be calculated from the camera
    position and the center point, because the transformation from cartesian
    to spherical coordinates is not unique.
    Furthermore, to enable smooth mouse-controlled camera rotation based on
    the current camera angles, it is essential to store the camera angles as
    the combined rotation matrix, not as the individual angles.
    
    Therefore we store the camera position/direction as follows:
        ctr: [ x,y,z ] : the reference point of the camera: this is always
              a point on the viewing axis. Usualy, it is the center point of
              the scene we are looking at.

        rot: 
        twist : rotation angle around the camera's viewing axis
        
    The default camera is at [0,0,0] and looking in the -z direction.
    Near and far clipping planes are by
    default set to 0.1, resp 10 times the camera distance.

    Some camera terminology:
    Position (eye) : position of the camera
    Scene center (ctr) : the point the camera is looking at.
    Up Vector : a vector pointing up from the camera.
    Viewing direction (rotx,roty,rotz)
    Lens angle (fovy)
    Aspect ratio (aspect)
    Clip (front/back)
    Perspective/Orthogonal

    We assume that matrixmode is always MODELVIEW.
    For other operations we explicitely switch before and afterwards back
    to MODELVIEW.
    """

    def __init__(self,center=[0.,0.,0.], long=0., lat=0., twist=0., dist=0.):
        """Create a new camera at position (0,0,0) looking along the -z axis"""
        self.setCenter(*center)
        self.setRotation(long,lat,twist)
        self.setDist(dist)
        self.setLens(45.,4./3.)
        self.setClip(0.1,10.)
        self.setPerspective(True)
        self.viewChanged = True

    # Use only these access functions to make implementation changes easier
        
    def getCenter(self):
        """Return the camera reference point (the scene center)."""
        return self.ctr
    def getRot(self):
        """Return the camera rotation matrix."""
        return self.rot
    def getDist(self):
        """Return the camera distance."""
        return self.dist


    def setCenter(self,x,y,z):
        """Set the center of the camera in global cartesian coordinates."""
        self.ctr = [x,y,z]
        self.viewChanged = True


    def setAngles(self,angles):
        """Set the rotation angles.

        angles is either:
            - a tuple of angles (long,lat,twist)
            - a named view corresponding to angles in view_angles
            - None
        """
        if type(angles) is str:
            angles = view_angles.get(angles)
        if angles is None:
            return
        self.setRotation(*angles)
            

    def setRotation(self,long,lat,twist=0):
        """Set the rotation matrix of the camera from three angles."""
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glRotatef(-twist % 360, 0.0, 0.0, 1.0)
        GL.glRotatef(lat % 360, 1.0, 0.0, 0.0)
        GL.glRotatef(-long % 360, 0.0, 1.0, 0.0)
        self.rot = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX)
        self.viewChanged = True

    def setDist(self,dist):
        """Set the distance."""
        self.dist = dist
        self.viewChanged = True

    def report(self):
        """Return a report of the current camera settings."""
        return """Camera Settings:
  Center: %s
  Rotation Matrix: %s
  Distance: %s
""" % (self.ctr,self.rot,self.dist)
        
    def dolly(self,val):
        """Move the camera eye towards/away from the scene center.

        This has the effect of zooming. A value > 1 zooms out,
        a value < 1 zooms in. The resulting enlargement of the view
        will approximately be 1/val.
        A zero value will move the camera to the center of the scene.
        The front and back clipping planes may need adjustment after
        a dolly operation.
        """
        self.setDist(self.getDist() * val)
        self.viewChanged = True
        
    def pan(self,val,axis=0):
        """Rotate the camera around axis through its eye. 

        The camera is rotated around an axis through the eye point.
        For axes 0 and 1, this will move the center, creating a panning
        effect. The default axis is parallel to the y-axis, resulting in
        horizontal panning. For vertical panning (axis=1) a convenience
        alias tilt is created.
        For axis = 2 the operation is equivalent to the rotate operation.
        """
        if axis==0 or axis ==1:
            pos = self.getPosition()
            self.eye[axis] = (self.eye[axis] + val) % 360
            center = diff(pos,sphericalToCartesian(self.eye))
            self.setCenter(*center)
        elif axis==2:
            self.twist = (self.twist + val) % 360
        self.viewChanged = True

    def tilt(self,val):
        """Rotate the camera up/down around its own horizontal axis.

        The camera is rotated around and perpendicular to the plane of the
        y-axis and the viewing axis. This has the effect of a vertical pan.
        A positive value tilts the camera up, shifting the scene down.
        The value is specified in degrees.
        """
        self.pan(val,1)
        self.viewChanged = True

    def move(self,dx,dy,dz):
        """Move the camera over translation (dx,dy,dz) in global coordinates.

        The center of the camera is moved over the specified translation
        vector. This has the effect of moving the scene in opposite direction.
        """
        x,y,z = self.ctr
        self.setCenter(x+dx,y+dy,z+dz)

##    def truck(self,dx,dy,dz):
##        """Move the camera translation vector in local coordinates.

##        This has the effect of moving the scene in opposite direction.
##        Positive coordinates mean:
##          first  coordinate : truck right,
##          second coordinate : pedestal up,
##          third  coordinate : dolly out.
##        """
##        #pos = self.getPosition()
##        ang = self.getAngles()
##        tr = [dx,dy,dz]
##        for i in [1,0,2]:
##            r = rotationMatrix(i,ang[i])
##            tr = multiply(tr, r)
##        self.move(*tr)
##        self.viewChanged = True


        
    def setMatrix(self):
        """Set the ModelView matrix from camera parameters.

        These are the transformations applied on the model space.
        Rotations and translations need be taken negatively.
        """
        # The operations on the model space
        # arguments should be taken negative and applied in backwards order
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        #printModelviewMatrix("Identity:\n%s")
        # translate over camera distance
        GL.glTranslate(0,0,-self.dist)
        #printModelviewMatrix("Camera distance:\n%s")
        # rotate
        GL.glMultMatrixf(self.rot)
        #printModelviewMatrix("Rotation:\n%s")
        # translate to center
        dx,dy,dz = self.getCenter()
        GL.glTranslatef(-dx,-dy,-dz)
        #printModelviewMatrix("Translation:\n%s")


    def lookAt(self,eye,center,up):
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GLU.gluLookAt(*numpy.concatenate([eye,center,up]))
        self.saveMatrix()

    def rotate(self,val,vx,vy,vz):
        """Rotate the camera around current camera axes."""
        GL.glMatrixMode(GL.GL_MODELVIEW)
        self.saveMatrix()
        GL.glLoadIdentity()
        GL.glTranslatef(0,0,-self.dist)
        GL.glRotatef(val,vx,vy,vz)
        GL.glMultMatrixf(self.rot)
        dx,dy,dz = self.getCenter()
        GL.glTranslatef(-dx,-dy,-dz)
        self.saveMatrix()

    def saveMatrix (self):
        """Save the ModelView matrix."""
        self.m = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX)
        self.rot = copy.deepcopy(self.m)
        self.trl = copy.deepcopy(self.rot[3,0:3])
        #print "Translation: %s" % self.trl
        self.rot[3,0:3] = [0.,0.,0.]

    def loadMatrix (self):
        """Load the saved ModelView matrix."""
        GL.glMatrixMode(GL.GL_MODELVIEW)
        if self.viewChanged:
            self.setMatrix()
            self.saveMatrix()
            self.viewChanged = False
        else:
            GL.glLoadMatrixf(self.m)
 
    def translate(self,vx,vy,vz,local=True):
        if local:
            vx,vy,vz = self.toWorld([vx,vy,vz,1])
        self.move(-vx,-vy,-vz)
      
    def transform(self,v):
        """Transform a vertex using the currently saved Modelview matrix."""
        if len(v) == 3:
            v = v + [ 1. ]
        v = multiply([v],self.m)[0]
        return [ a/v[3] for a in v[0:3] ]

    def toWorld(self,v,trl=False):
        """Transform a vertex from camera to world coordinates.

        The specified vector can have 3 or 4 (homogoneous) components.
        This uses the currently saved rotation matrix.
        """
        a = inverse(array(self.rot))
        if len(v) == 3:
            v = v + [ 1. ]
        v = multiply(array(v),a)
        return v[0:3] / v[3]

       
    
    # Camera Lens Setting.
    #
    # These include :
    #   - the vertical lens opening angle (fovy),
    #   - the aspect ratio (aspect = width/height)
    #   - the front and back clipping planes (near,far)
    #
    # These functions do not auto-reload the projection matrix, so you
    # do not need to make the GL-environment current before using them.
    # The client has to explicitely call the loadProjection() method to
    # make the settings active 
    # These functions will flag a change in the camera settings, which
    # can be tested by your display() function to know if it has to reload
    # the projection matrix.

    def setLens(self,fovy=None,aspect=None):
        """Set the field of view of the camera.

        We set the field of view by the vertical opening angle fovy
        and the aspect ratio (width/height) of the viewing volume.
        A parameter that is not specified is left unchanged.
        """
        if fovy: self.fovy = min(abs(fovy),180)
        if aspect: self.aspect = abs(aspect)
        self.lensChanged = True
        
    def setClip(self,near,far):
        """Set the near and far clipping planes"""
        if near > 0 and near < far:
            self.near,self.far = near,far
            self.lensChanged = True
        else:
            print "Error: Invalid Near/Far clipping values""" 
        self.lensChanged = True
        
    def setClipRel(self,near,far):
        """Set the near and far clipping planes"""
        if near > 0 and near < far:
            self.near,self.far = near,far
            self.lensChanged = True
        else:
            print "Error: Invalid Near/Far clipping values""" 

    def setPerspective(self,on=True):
        """Set perspective on or off"""
        self.perspective = on
        self.lensChanged = True

    def zoom(self,val=0.5):
        """Zoom in/out by shrinking/enlarging the camera view angle.

        The zoom factor is relative to the current setting.
        Use setFovy() to specify an absolute setting.
        """
        if val>0:
            self.fovy *= val
        self.lensChanged = True

    def loadProjection(self,force=False,pick=None):
        """Load the projection/perspective matrix.

        The caller will have to setup the correct GL environment beforehand.
        No need to set matrix mode though. This function will switch to
        GL_PROJECTION mode before loading the matrix, and go back to
        GL_MODELVIEW mode on exit.

        A pick region can be defined to use the camera in picking mode.
        pick defines the picking region center and size (x,y,w,h).

        This function does it best at autodetecting changes in the lens
        settings, and will only reload the matrix if such changes are
        detected. You can optionally force loading the matrix.
        """
        if self.lensChanged or force:
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            if pick:
                #print 'PICK: %s' % str(pick)
                GLU.gluPickMatrix(*pick)
            if self.perspective:
                GLU.gluPerspective(self.fovy,self.aspect,self.near,self.far)
            else:
                #print "FOVY: %s" % self.fovy
                top = tand(self.fovy*0.5) * self.dist
                bottom = -top
                right = top * self.aspect
                left = bottom * self.aspect
                #print "Ortho %s" % [left,right,bottom,top,self.near,self.far]
                GL.glOrtho(left,right,bottom,top,self.near,self.far)
            GL.glMatrixMode(GL.GL_MODELVIEW)     


if __name__ == "__main__":
    
    from OpenGL.GLUT import *
   
    def init():
        GL.glClearColor (0.0, 0.0, 0.0, 0.0)
        GL.glShadeModel (GL.GL_FLAT)

    def display():
        global cam
        GL.glClear (GL.GL_COLOR_BUFFER_BIT)
        GL.glColor3f (1.0, 1.0, 1.0)
        GL.glLoadIdentity ()             # clear the matrix
        cam.loadMatrix()
        glutWireCube (1.0)
        GL.glFlush ()

    def reshape (w, h):
        GL.glViewport (0, 0, w, h)
        GL.glMatrixMode (GL.GL_PROJECTION)
        GL.glLoadIdentity ()
        GL.glFrustum (-1.0, 1.0, -1.0, 1.0, 1.5, 20.0)
        GL.glMatrixMode (GL.GL_MODELVIEW)

    def keyboard(key, x, y):
        global cam
        if key == 27:
            sys.exit()
        elif key == 'd':
            cam.dolly(1.1)
        elif key == 'D':
            cam.dolly(0.9)
        elif key == 'r':
            cam.rotate(5.)
        elif key == 'R':
            cam.rotate(-5.)
        elif key == 's':
            cam.rotate(5.,1)
        elif key == 'S':
            cam.rotate(-5.,1)
        elif key == 'w':
            cam.rotate(5.,2)
        elif key == 'W':
            cam.rotate(-5.,2)
        elif key == 'p':
            cam.pan(5.)
        elif key == 'P':
            cam.pan(-5.)
        elif key == 't':
            cam.tilt(5.)
        elif key == 'T':
            cam.tilt(-5.)
        elif key == 'h':
            cam.move(0.2,0.,0.)
        elif key == 'H':
            cam.move(-0.2,0.,0.)
        elif key == 'v':
            cam.move(0.,0.2,0.)
        elif key == 'V':
            cam.move(0.,-0.2,0.)
        elif key == '+':
            cam.zoom(0.8)
        elif key == '-':
            cam.zoom(1.25)
##         elif key == 'x':
##             cam.truck([0.5,0.,0.])
##         elif key == 'X':
##             cam.truck([-0.5,0.,0.])
##         elif key == 'y':
##             cam.truck([0.,0.5,0.])
##         elif key == 'Y':
##             cam.truck([0.,-0.5,0.])
##         elif key == 'z':
##             cam.truck([0.,0.,0.5])
##         elif key == 'Z':
##             cam.truck([0.,0.,-0.5])
        elif key == 'o':
            cam.setPerspective(not cam.perspective)
            cam.loadProjection
        else:
            print key
        display()
            

    def main():
        global cam
        glutInit(sys.argv)
        glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB)
        glutInitWindowSize (500, 500) 
        #glutInitWindowPosition (100, 100)
        glutCreateWindow (sys.argv[0])
        init ()
        
        cam = Camera(center=[0.,0.,0.],position=[0.,0.,5.])
        cam.setLens(45.,1.)

        glutDisplayFunc(display) 
        glutReshapeFunc(reshape)
        glutKeyboardFunc(keyboard)
        glutMainLoop()
        return 0

    main()

#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""camera 0.1 (C) Benedict Verhegghe"""

from numpy import *

inverse = linalg.linalg.inv
multiply = dot

def tand(arg):
    """Return the tan of an angle in degrees."""
    return tan(arg*pi/180.)

import copy

import OpenGL.GL as GL
import OpenGL.GLU as GLU




def printModelviewMatrix(s="%s"):
    print(s % GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX))



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

    def __init__(self,center=[0.,0.,0.], long=0., lat=0., twist=0., dist=1.):
        """Create a new camera at position (0,0,0) looking along the -z axis"""
        self.locked = False
        self.setCenter(*center)
        self.setRotation(long,lat,twist)
        self.setDist(dist)
        self.setLens(45.,4./3.)
        self.setClip(0.1,10.)
        self.area = None
        self.resetArea()
        self.keep_aspect = True
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

    def lock(self,onoff=True):
        self.locked = onoff
        print "Camera locked is %s" % self.locked


    def setCenter(self,x,y,z):
        """Set the center of the camera in global cartesian coordinates."""
        if not self.locked:
            self.ctr = [x,y,z]
            self.viewChanged = True


    def setAngles(self,angles):
        """Set the rotation angles.

        angles is either:
            - a tuple of angles (long,lat,twist)
            - a named view corresponding to angles in view_angles
            - None
        """
        if not self.locked:
            if type(angles) is str:
                angles = view_angles.get(angles)
            if angles is None:
                return
            self.setRotation(*angles)
            

    def setRotation(self,long,lat,twist=0):
        """Set the rotation matrix of the camera from three angles."""
        if not self.locked:
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            GL.glRotatef(-twist % 360, 0.0, 0.0, 1.0)
            GL.glRotatef(lat % 360, 1.0, 0.0, 0.0)
            GL.glRotatef(-long % 360, 0.0, 1.0, 0.0)
            self.rot = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX)
            self.viewChanged = True


    def setDist(self,dist):
        """Set the distance."""
        if not self.locked:
            if dist > 0.0 and dist != inf:
                self.dist = dist
                self.viewChanged = True


    def report(self):
        """Return a report of the current camera settings."""
        return """Camera Settings:
  Center: %s
  Distance: %s
  Rotation Matrix:
  %s
  Field of View y: %s
  Aspect Ratio: %s
  Area: %s, %s
  Near/Far Clip: %s, %s
""" % (self.ctr,self.dist,self.rot,self.fovy,self.aspect,self.area[0],self.area[1],self.near,self.far)

        
    def dolly(self,val):
        """Move the camera eye towards/away from the scene center.

        This has the effect of zooming. A value > 1 zooms out,
        a value < 1 zooms in. The resulting enlargement of the view
        will approximately be 1/val.
        A zero value will move the camera to the center of the scene.
        The front and back clipping planes may need adjustment after
        a dolly operation.
        """
        if not self.locked:
            self.setDist(self.getDist() * val)
            #print("DIST %s" % self.dist)
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
        if not self.locked:
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
        if not self.locked:
            self.pan(val,1)
            self.viewChanged = True


    def move(self,dx,dy,dz):
        """Move the camera over translation (dx,dy,dz) in global coordinates.

        The center of the camera is moved over the specified translation
        vector. This has the effect of moving the scene in opposite direction.
        """
        if not self.locked:
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
        if not self.locked:
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
        if not self.locked:
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            GLU.gluLookAt(*concatenate([eye,center,up]))
            self.saveMatrix()


    def rotate(self,val,vx,vy,vz):
        """Rotate the camera around current camera axes."""
        if not self.locked:
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
        if not self.locked:
            self.m = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX)
            self.rot = copy.deepcopy(self.m)
            self.trl = copy.deepcopy(self.rot[3,0:3])
            #print("Translation: %s" % self.trl)
            self.rot[3,0:3] = [0.,0.,0.]
            #print "Rotation: %s" % self.rot


    def loadMatrix (self):
        """Load the saved ModelView matrix."""
        if not self.locked:
            GL.glMatrixMode(GL.GL_MODELVIEW)
            if self.viewChanged:
                self.setMatrix()
                self.saveMatrix()
                self.viewChanged = False
            else:
                GL.glLoadMatrixf(self.m)


    def loadCurrentRotation (self):
        """Load the current ModelView matrix with translations canceled out."""
        rot = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX)
        rot[3,0:3] = [0.,0.,0.]
        GL.glLoadMatrixf(rot)

 
    def translate(self,vx,vy,vz,local=True):
        if not self.locked:
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


    def resetArea(self):
        """Set maximal camera area.

        Resets the camera window area to its maximum values corresponding
        to the fovy setting, symmetrical about the camera axes. 
        """
        self.setArea(0.,0.,1.,1.,False)


    def setArea(self,hmin,vmin,hmax,vmax,relative=True,center=False,clip=True):
        """Set the viewable area of the camera."""
        area = array([hmin,vmin,hmax,vmax])
        if clip:
            area = area.clip(0.,1.)
        if area[0] < area[2] and area[1] < area[3]:
            area = area.reshape(2,2)
            mean = (area[1]+area[0]) * 0.5
            diff = (area[1]-area[0]) * 0.5
            
            if relative:
                if center:
                    mean = zeros(2)
                if self.keep_aspect:
                    aspect = diff[0] / diff[1]
                    if aspect > 1.0:
                        diff[1] = diff[0] #/ self.aspect
                        # no aspect factor: this is relative!!!
                    else:
                        diff[0] = diff[1] #* self.aspect
                    area[0] = mean-diff
                    area[1] = mean+diff
                #print("RELATIVE AREA %s" % (area))
                area = (1.-area) * self.area[0] + area * self.area[1]

            #print("OLD ZOOM AREA %s (aspect %s)" % (self.area,self.aspect))
            #print("NEW ZOOM AREA %s" % (area))
                                                       
            self.area = area
            self.lensChanged = True



    def zoomArea(self,val=0.5,area=None):
        """Zoom in/out by shrinking/enlarging the camera view area.

        The zoom factor is relative to the current setting.
        Values smaller than 1.0 zoom in, larger values zoom out.
        """
        if val>0:
            #val = (1.-val) * 0.5
            #self.setArea(val,val,1.-val,1.-val,center=center)
            if area is None:
                area = self.area
            #print("ZOOM AREA %s (%s)" % (area.tolist(),val))
            mean = (area[1]+area[0]) * 0.5
            diff = (area[1]-area[0]) * 0.5 * val
            area[0] = mean-diff
            area[1] = mean+diff
            self.area = area
            #print("CAMERA AREA %s" % self.area.tolist())
            self.lensChanged = True

            
    def transArea(self,dx,dy):
        """Pan by moving the vamera area.

        dx and dy are relative movements in fractions of the
        current area size.
        """
        #print("TRANSAREA %s,%s" % (dx,dy))
        area = self.area
        diff = (area[1]-area[0]) * array([dx,dy])
        area += diff
        self.area = area
        self.lensChanged = True
            

        
    def setClip(self,near,far):
        """Set the near and far clipping planes"""
        if near > 0 and near < far:
            self.near,self.far = near,far
            self.lensChanged = True
        else:
            print("Error: Invalid Near/Far clipping values")

        
    ## def setClipRel(self,near,far):
    ##     """Set the near and far clipping planes"""
    ##     if near > 0 and near < far:
    ##         self.near,self.far = near,far
    ##         self.lensChanged = True
    ##     else:
    ##         print("Error: Invalid Near/Far clipping values")

    def setPerspective(self,on=True):
        """Set perspective on or off"""
        self.perspective = on
        self.lensChanged = True


    ## def zoom(self,val=0.5):
    ##     """Zoom in/out by shrinking/enlarging the camera view angle.

    ##     The zoom factor is relative to the current setting.
    ##     Use setFovy() to specify an absolute setting.
    ##     """
    ##     if val>0:
    ##         self.fovy *= val
    ##     self.lensChanged = True


    def loadProjection(self,force=False,pick=None,keepmode=False):
        """Load the projection/perspective matrix.

        The caller will have to setup the correct GL environment beforehand.
        No need to set matrix mode though. This function will switch to
        GL_PROJECTION mode before loading the matrix

        !! CHANGED: does not switch back to GL_MODELVIEW mode!

        A pick region can be defined to use the camera in picking mode.
        pick defines the picking region center and size (x,y,w,h).

        This function does it best at autodetecting changes in the lens
        settings, and will only reload the matrix if such changes are
        detected. You can optionally force loading the matrix.
        """
        GL.glMatrixMode(GL.GL_PROJECTION)
        if self.lensChanged or force:
            GL.glLoadIdentity()
            if pick:
                GLU.gluPickMatrix(*pick)
                
            fv = tand(self.fovy*0.5)
            if self.perspective:
                fv *= self.near
            else:
                fv *= self.dist
            fh = fv * self.aspect
            x0,x1 = 2*self.area - 1.0
            frustum = (fh*x0[0],fh*x1[0],fv*x0[1],fv*x1[1],self.near,self.far)
            if self.perspective:
                GL.glFrustum(*frustum)
            else:
                GL.glOrtho(*frustum)
        if not keepmode:
            GL.glMatrixMode(GL.GL_MODELVIEW)     



if __name__ == "__main__":
    
    from OpenGL.GLUT import *
    import sys
   
    def init():
        GL.glClearColor (0.0, 0.0, 0.0, 0.0)
        GL.glShadeModel (GL.GL_FLAT)

    def display():
        global cam
        GL.glClear (GL.GL_COLOR_BUFFER_BIT)
        GL.glColor3f (1.0, 1.0, 1.0)
        GL.glLoadIdentity ()             # clear the matrix
        cam.loadProjection()
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
        else:
            print(key)
        display()
            

    def main():
        global cam
        glutInit(sys.argv)
        glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB)
        glutInitWindowSize (500, 500) 
        #glutInitWindowPosition (100, 100)
        glutCreateWindow (sys.argv[0])
        init ()
        
        cam = Camera(center=[0.,0.,0.],dist=5.)
        cam.setLens(45.,1.)

        glutDisplayFunc(display) 
        glutReshapeFunc(reshape)
        glutKeyboardFunc(keyboard)
        glutMainLoop()
        return 0

    main()

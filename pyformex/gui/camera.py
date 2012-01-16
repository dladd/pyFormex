#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""OpenGL camera handling"""

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

built_in_views = {
    'front': (0.,0.,0.),
    'back': (180.,0.,0.),
    'right': (90.,0.,0.),
    'left': (270.,0.,0.),
    'top': (0.,90.,0.),
    'bottom': (0.,-90.,0.),
    'iso0': (45.,45.,0.),
    'iso1': (45.,135.,0.),
    'iso2': (45.,225.,0.),
    'iso3': (45.,315.,0.),
    'iso4': (-45.,45.,0.),
    'iso5': (-45.,135.,0.),
    'iso6': (-45.,225.,0.),
    'iso7': (-45.,315.,0.),
    }

class ViewAngles(dict):
    """A dict to keep named camera angle settings.

    This class keeps a dictionary of named angle settings. Each value is
    a tuple of (longitude, latitude, twist) camera angles.
    This is a static class which should not need to be instantiated.

    There are seven predefined values: six for looking along global
    coordinate axes, one isometric view.
    """

    def __init__(self,data = built_in_views):
       dict.__init__(self,data)
       self['iso'] = self['iso0']
        

    def get(self,name):
        """Get the angles for a named view.

        Returns a tuple of angles (longitude, latitude, twist) if the
        named view was defined, or None otherwise
        """
        return dict.get(self,name,None)


view_angles = ViewAngles()


class Camera(object):
    """A camera for OpenGL rendering.

    The Camera class holds all the camera related settings related to
    the rendering of a scene in OpenGL. These include camera position,
    the viewing direction of the camera, and the lens parameters (opening
    angle, front and back clipping planes).
    This class also provides convenient methods to change the settings so as
    to get smooth camera manipulation.

    Camera position and orientation:

        The camera viewing line is defined by two points: the position of
        the camera and the center of the scene the camera is looking at.
        We use the center of the scene as the origin of a local coordinate
        system to define the camera position. For convenience, this could be
        stored in spherical coordinates, as a distance value and two angles:
        longitude and latitude. Furthermore, the camera can also rotate around
        its viewing line. We can define this by a third angle, the twist.
        From these four values, the needed translation vector and rotation
        matrix for the scene rendering may be calculated. 

        Inversely however, we can not compute a unique set of angles from
        a given rotation matrix (this is known as 'gimball lock').
        As a result, continuous (smooth) camera rotation by e.g. mouse control
        requires that the camera orientation be stored as the full rotation
        matrix, rather than as three angles. Therefore we store the camera
        position and orientation as follows:
    
        - `ctr`: `[ x,y,z ]` : the reference point of the camera:
          this is always a point on the viewing axis. Usually, it is set to
          the center of the scene you are looking at.
        - `dist`: distance of the camera to the reference point. 
        - `rot`: a 3x3 rotation matrix, rotating the global coordinate system
          thus that the z-direction is oriented from center to camera.

        These values have influence on the ModelView matrix.
       
    Camera lens settings:

        The lens parameters define the volume that is seen by the camera.
        It is described by the following parameters:

        - `fovy`: the vertical lens opening angle (Field Of View Y),
        - `aspect`: the aspect ratio (width/height) of the lens. The product
          `fovy * aspect` is the horizontal field of view.
        - `near, far`: the position of the front and back clipping planes.
          They are given as distances from the camera and should both be
          strictly positive. Anything that is closer to the camera than
          the `near` plane or further away than the `far` plane, will not be
          shown on the canvas.

        Camera methods that change these values will not directly change
        the ModelView matrix. The :meth:`loadModelView` method has to be called
        explicitely to make the settings active.

        These values have influence on the Projection matrix.

    Methods that change the camera position, orientation or lens parameters
    will not directly change the related ModelView or Projection matrix.
    They will just flag a change in the camera settings. The changes are
    only activated by a call to the :meth:`loadModelView` or
    :meth:`loadProjection` method, which will test the flags to see whether
    the corresponding matrix needs a rebuild.
        
    The default camera is at distance 1.0 of the center point [0.,0.,0.] and
    looking in the -z direction.
    Near and far clipping planes are by default set to 0.1, resp 10 times
    the camera distance.
    """

    # DEVELOPERS:
    #    The camera class assumes that matrixmode is always ModelView on entry.
    #    For operations in other modes, an explicit switch before the operations
    #    and afterwards back to ModelView should be performed.


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
        self.tracking = False
        self.m = self.p = self.v = None

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
        """Lock/unlock a camera.

        When a camera is locked, its position and lens parameters can not be
        changed.
        This can e.g. be used in multiple viewports layouts to create fixed
        views from different angles.
        """
        self.locked = bool(onoff)


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


    def lookAt(self,eye,center,up):
        if not self.locked:
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            GLU.gluLookAt(*concatenate([eye,center,up]))
            self.saveModelView()


    def rotate(self,val,vx,vy,vz):
        """Rotate the camera around current camera axes."""
        if not self.locked:
            GL.glMatrixMode(GL.GL_MODELVIEW)
            self.saveModelView()
            GL.glLoadIdentity()
            GL.glTranslatef(0,0,-self.dist)
            GL.glRotatef(val,vx,vy,vz)
            GL.glMultMatrixf(self.rot)
            dx,dy,dz = self.getCenter()
            GL.glTranslatef(-dx,-dy,-dz)
            self.saveModelView()


    def saveModelView (self):
        """Save the ModelView matrix."""
        if not self.locked:
            self.m = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX)
            self.rot = copy.deepcopy(self.m)
            self.trl = copy.deepcopy(self.rot[3,0:3])
            #print("Translation: %s" % self.trl)
            self.rot[3,0:3] = [0.,0.,0.]
            #print "Rotation: %s" % self.rot

        
    def setModelView(self):
        """Set the ModelView matrix from camera parameters.

        """
        if not self.locked:
            # The camera operations are applied on the model space
            # Arguments should be taken negative and applied in backwards order
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            # translate over camera distance
            GL.glTranslate(0,0,-self.dist)
            # rotate according to current rotation matrix
            GL.glMultMatrixf(self.rot)
            # translate to center
            dx,dy,dz = self.getCenter()
            GL.glTranslatef(-dx,-dy,-dz)


    def loadModelView (self,m=None):
        """Load the ModelView matrix.

        There are thrre uses of this function:

        - Without argument and if the viewing parameters have not changed
          since the last save of the ModelView matrix, this will just reload
          the ModelView matrix from the saved value.

        - If an argument is supplied, it should be a legal ModelView matrix
          and that matrix will be loaded (and saved) as the new ModelView
          matrix.

        - Else, a new ModelView matrix is set up from the camera parameters,
          and it is loaded and saved.

        In the latter two cases, the new ModelView matrix is saved, and if
        a camera attribute `modelview_callback` has been set, a call to
        this function is done, passing the camera instance as parameter.
        """
        if not self.locked:
            GL.glMatrixMode(GL.GL_MODELVIEW)

            if m is not None or self.viewChanged:
                if m is not None:
                    GL.glLoadMatrixf(m)
                else:
                    self.setModelView()
                self.saveModelView()
                try:
                    self.modelview_callback(self)
                except:
                    pass
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
            try:
                self.projection_callback(self)
            except:
                pass
        if not keepmode:
            GL.glMatrixMode(GL.GL_MODELVIEW)     


    #### global manipulation ###################

    def set3DMatrices(self):
        self.loadProjection()
        self.loadModelView()
        # this is saved by loadModelView
        #self.m = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
        ##!!! self.p and self.v should be saved as we do with self.m
        self.p = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
        self.v = GL.glGetIntegerv(GL.GL_VIEWPORT)


    def project(self,x,y,z):
        "Map the object coordinates (x,y,z) to window coordinates."""
        self.set3DMatrices()
        return GLU.gluProject(x,y,z,self.m,self.p,self.v)


    def unProject(self,x,y,z):
        "Map the window coordinates (x,y,z) to object coordinates."""
        self.set3DMatrices()
        return GLU.gluUnProject(x,y,z,self.m,self.p,self.v)


    def setTracking(self,onoff=True):
        """Enable/disable coordinate tracking using the camera"""
        if onoff:
            self.tracking = True
            self.set3DMatrices()
        else:
            self.tracking = False

#############################################################################

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
        cam.loadModelView()
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

# End

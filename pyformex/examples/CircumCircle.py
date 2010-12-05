#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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
"""CircumCircle

level = 'beginner'
topics = ['geometry']
techniques = ['function','import','mpattern','dialog','viewport']
"""
import simple
from examples.Cube import cube_tri
from plugins.geomtools import *

## /// <summary>
## /// Calculates the intersection point between two lines, assuming that there is such a point.
## /// </summary>
## /// <param name="originD">The origin of the first line</param>
## /// <param name="directionD">The direction of the first line.</param>
## /// <param name="originE">The origin of the second line.</param>
## /// <param name="directionE">The direction of the second line.</param>
## /// <returns>The point at which the two lines intersect.</returns>
## Vector3 LineLineIntersection(Vector3 originD, Vector3 directionD, Vector3 originE, Vector3 directionE) {
##   directionD.Normalize();
##   directionE.Normalize();
##   var N = Vector3.Cross(directionD, directionE);
##   var SR = originD - originE;
##   var absX = Math.Abs(N.X);
##   var absY = Math.Abs(N.Y);
##   var absZ = Math.Abs(N.Z);
##   float t;
##   if (absZ > absX && absZ > absY) {
##     t = (SR.X*directionE.Y - SR.Y*directionE.X)/N.Z;
##   } else if (absX > absY) {
##     t = (SR.Y*directionE.Z - SR.Z*directionE.Y)/N.X;
##   } else {
##     t = (SR.Z*directionE.X - SR.X*directionE.Z)/N.Y;
##   }
##   return originD - t*directionD;
## }

## /// <summary>
## /// Calculates the distance between a point and a line.
## /// </summary>
## /// <param name="P">The point.</param>
## /// <param name="S">The origin of the line.</param>
## /// <param name="D">The direction of the line.</param>
## /// <returns>
## /// The distance of the point to the line.
## /// </returns>
## float PointLineDistance(Vector3 P, Vector3 S, Vector3 D) {
##   D.Normalize();
##   var SP = P - S;
##   return Vector3.Distance(SP, Vector3.Dot(SP, D)*D);
## }


# Circumcircle
## // lines from a to b and a to c
## var AB = B - A;
## var AC = C - A;

## // perpendicular vector on triangle
## var N = Vector3.Normalize(Vector3.Cross(AB, AC));

## // find the points halfway on AB and AC
## var halfAB = A + AB*0.5f;
## var halfAC = A + AC*0.5f;

## // build vectors perpendicular to ab and ac
## var perpAB = Vector3.Cross(AB, N);
## var perpAC = Vector3.Cross(AC, N);

## // find intersection between the two lines
## // D: halfAB + t*perpAB
## // E: halfAC + s*perpAC
## var center = LineLineIntersection(halfAB, perpAB, halfAC, perpAC);
## // the radius is the distance between center and any point
## // distance(A, B) = length(A-B)
## var radius = Vector3.Distance(center, A);



def draw_circles(circles,color=red):
    for r,c,n in circles:
        C = simple.circle(r=r,n=n,c=c)
        draw(C,color=color)


def drawCircles(F,func,color=red):
    r,c,n = func(F.coords)
    draw(c,color=color)
    draw_circles(zip(r,c,n),color=color)
    
    
layout(2)
wireframe()

# draw in viewport 0
viewport(0)
view('front')
clear()
rtri = Formex(mpattern('16-32')).scale([1.5,1,0])
F = rtri + rtri.shear(0,1,-0.5).trl(0,-4.0) + rtri.shear(0,1,0.75).trl(0,3.0)
draw(F)

drawCircles(F,triangleCircumCircle,color=red)
zoomAll()   
drawCircles(F,triangleInCircle,color=blue)
drawCircles(F,triangleBoundingCircle,color=black)
zoomAll()   


# draw in viewport 1
viewport(1)
view('iso')
clear()
F,c = cube_tri()
draw(F)
drawCircles(F,triangleInCircle)
zoomAll()   

if not ack("Keep both viewports ?"):
    print "Removing a viewport"
    # remove last viewport
    removeViewport()

# End


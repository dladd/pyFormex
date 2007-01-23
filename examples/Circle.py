#!/usr/bin/env pyformex
# $Id: Circle.py 154 2006-11-03 19:08:25Z bverheg $

from simple import circle

def length(A,axis=-1):
    """Returns the length of the vectors of A in the direction of axis.

    The default axis is the last.
    """
    return sqrt((A*A).sum(axis))

def rotationAngle(A,B,rad=False):
    """Return rotation vector and angle for rotation of A to B.

    A and B are (n,3)-shaped arrays where each line represents a vector.
    This function computes the rotation from each vector of A to the
    corresponding vector of B. Broadcasting is done if one of A or B has
    only one row.
    The return value is a tuple of an (n,) shaped array with rotation angles
    (by default in degrees) and an (n,3)-shaped array with unit vectors
    along the rotation axis.
    If rad==True, the returned angles are in radians.
    """
    N = cross(A,B)
    L = length(N)
    S = L / (length(A)*length(B))
    ANG = arcsin(S.clip(min=-1.0,max=1.0))
    N = N/column_stack([L])
    if not rad:
        ANG *= 180./pi
    return (ANG,N)

# Test
linewidth(1)
drawtimeout = 1
for i in [3,4,5,6,8,12,20,60,180]:
    #print "%s points" % i
    clear()
    draw(circle(360./i,360./i),bbox=None)
    clear()
    draw(circle(360./i,2*360./i),bbox=None)
    clear()
    draw(circle(360./i,360./i,180.),bbox=None)

# Example of the use
clear()
n = 40
h = 0.5
line = Formex(pattern('1'*n)).scale(2./n).translate([-1.,0.,0.])
curve = line.bump(1,[0.,h,0.],lambda x: 1.-x**2)
curve.setProp(1)
draw(line)
draw(curve)


# Create circles in the middle of each segment, with normal along the segment
# begin and end points of segment
A = curve.f[:,0,:]
B = curve.f[:,1,:]
# midpoint and segment director
C = 0.5*(B+A)
D = B-A
# vector initially normal to circle defined above
nuc = array([0.,0.,1.])
# rotation angles and vectors
ang,rot = rotationAngle(nuc,D)
# diameters varying linearly with the |x| coordinate
diam = 0.1*h*(2.-abs(C[:,0]))
# finally, here are the circles:
circles = [ circle().scale(d).rotate(a,r).translate(c) for d,r,a,c in zip(diam,rot,ang,C) ]
F = Formex.concatenate(circles).setProp(3) 
draw(F)

# And now something more fancy: connect 1 out of 10 points of the circles
conn=range(0,180,15)
G = Formex.concatenate([ connect([c1.select(conn),c2.select(conn)]) for c1,c2 in zip(circles[:-1],circles[1:]) ])
draw(G)


# Fly through
if ack("Are you ready for a flight through the tube?"):
    flyAlong(curve,sleeptime=0.1)

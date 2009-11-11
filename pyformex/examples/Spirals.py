#!/usr/bin/env pyformex --gui
"""Spirals

level = 'normal'
topics = ['geometry']
techniques = ['curve','transform']
"""

from plugins import curve

m = 100 # number of cells along spiral
a = 2. # number of 360 degree turns

linewidth(2)
clear()

F = Formex(origin()) # base pattern, here a point
F = F.replic(m,1.,0)
s = a*2*pi/m
F = F.scale(s)
draw(F)

def spiral(X,dir=[0,1,2],rfunc=lambda x:1,zfunc=lambda x:1):
    """Perform a spiral transformation on a coordinate array"""
    print X.shape
    theta = X[...,dir[0]]
    r = rfunc(theta) + X[...,dir[1]]
    x = r * cos(theta)
    y = r * sin(theta)
    z = zfunc(theta) + X[...,dir[2]]
    X = hstack([x,y,z]).reshape(X.shape)
    print X.shape
    return Coords(X)


nwires=6

phi = 30.
alpha2 = 70.
c = 1.
a = c*tand(phi)
b = tand(phi) / tand(alpha2)
 

print "a = %s, b = %s, c = %s" % (a,b,c)
print c*b/a
print tand(45.)
print arctan(c*b/a) / Deg

zf = lambda x : c * exp(b*x)
rf = lambda x : a * exp(b*x)


S = spiral(F.f,[0,1,2],rf)#.rosette(nwires,360./nwires)

PL = curve.PolyLine(S[:,0,:])
clear()
draw(PL,color=red)
draw(PL.coords,color=red)
#drawNumbers(PL.coords)


if ack("Spread point evenly?"):
    at = PL.atLength(PL.nparts)
    X = PL.pointsAt(at)
    PL = curve.PolyLine(X)
    clear()
    draw(PL,color=blue)
    draw(PL.coords,color=blue)

sweep = ask("Sweep cross section",['None','line','surface'])
if sweep == 'line':
    CS = Formex(pattern('123'))  # circumference of a square
elif sweep == 'surface':
    CS = Formex(mpattern('123'))  # a square surface
else:
    exit()

# Use a Mesh, because that already has a 'sweep' function
CS = CS.toMesh()
structure = CS.sweep(PL,normal=2)
clear()
draw(structure)

# End


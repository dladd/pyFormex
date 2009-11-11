#!/usr/bin/env pyformex --gui
"""Spirals

level = 'normal'
topics = ['geometry']
techniques = ['curve','sweep','mesh']
"""

from plugins import curve

linewidth(2)
clear()

res = askItems([
    ('nmod',200,{'text':'Number of cells along spiral'}),
    ('turns',2.25,{'text':'Number of 360 degree turns'}),
    ('spiral3d',0.0,{'text':'Out of plane factor'}),
    ('nwires',2,{'text':'Number of spirals'}),
    ])

globals().update(res)

F = Formex(origin()).replic(nmod,1.,0).scale(turns*2*pi/nmod)

def spiral(X,dir=[0,1,2],rfunc=lambda x:1,zfunc=lambda x:0):
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

def drawSpiralCurves(PL,nwires,color1,color2=None):
    if color2 is None:
        color2 = color1
    draw(PL,color=color1)
    if nwires <= 1:
        draw(PL.coords,color=color2)
    else:
        draw(Formex(PL.coords).rosette(nwires,360./nwires),color=color2)


phi = 30.
alpha2 = 70.
c = 1.
a = c*tand(phi)
b = tand(phi) / tand(alpha2)
 

print "a = %s, b = %s, c = %s" % (a,b,c)
print c*b/a
print tand(45.)
print arctan(c*b/a) / Deg

rf = lambda x : a * exp(b*x)
if spiral3d:
    zf = lambda x : spiral3d * exp(b*x)
else:
    zf = lambda x : 0.0

S = spiral(F.f,[0,1,2],rf,zf)

PL = curve.PolyLine(S[:,0,:])

clear()
drawSpiralCurves(PL,nwires,red,blue)

if ack("Spread point evenly?"):
    at = PL.atLength(PL.nparts)
    X = PL.pointsAt(at)
    PL = curve.PolyLine(X)
    clear()
    drawSpiralCurves(PL,nwires,blue,red)


sweep = ask("Sweep cross section",['None','line','surface'])
if sweep == 'line':
    CS = Formex(pattern('1653')).rotate(90)  # circumference of a square
elif sweep == 'surface':
    CS = Formex(mpattern('123'))  # a square surface
else:
    exit()

# Use a Mesh, because that already has a 'sweep' function
CS = CS.swapAxes(0,2).scale(0.5).toMesh()
structure = CS.sweep(PL,normal=0,upvector=None,avgdir=True)
clear()
draw(structure,color=yellow)

if nwires > 1:
    structure = structure.toFormex().rosette(nwires,360./nwires).toMesh()
    draw(structure,color='orange')
    

# End


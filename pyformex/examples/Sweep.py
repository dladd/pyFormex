#!/usr/bin/env pyformex --gui
"""Spirals

level = 'normal'
topics = ['geometry']
techniques = ['curve','sweep','mesh']
"""

from plugins import curve
import simple

linewidth(2)
clear()

spiral_data = [
    ('nmod',200,{'text':'Number of cells along spiral'}),
    ('spread',False,{'text':'Spread points evenly along spiral'}),
    ('turns',2.25,{'text':'Number of 360 degree turns'}),
    ('spiral3d',0.0,{'text':'Out of plane factor'}),
    ('nwires',2,{'text':'Number of spirals'}),
    ('sweep',False,{'text':'Sweep a cross section along the spiral'}),
    ]


# get the existing patterns from simple module
cross_sections_2d = simple.Pattern
# remove the non-plane patterns
#del cross_sections_2d['cube']
#del cross_sections_2d['star3d']
# add some more patterns
cross_sections_2d['channel'] = '1223'
cross_sections_2d['sigma'] = '16253'
cross_sections_2d['H-beam'] = '11/322/311'
# define some plane surface patterns
cross_sections_3d = {
    'filled_square':'123',
    'filled_octagon':'15263748',
    }


sweep_data = [
    ('cross_section',None,'select',{'text':'Shape of cross section','choices':cross_sections_2d.keys()+cross_sections_3d.keys()}),
    ('cross_rotate',0.,{'text':'Cross section rotation angle before sweeping'}),
    ]


input_data = {
    'Spiral Data' : spiral_data,
    'Sweep Data' : sweep_data,
}

#import gui.widgets
#dialog = widgets.InputDialog(input_data)
#res = dialog.getResult()

res = askItems(input_data)
if not res:
    exit()

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


def createCrossSection():
    if cross_section in cross_section_2d:
        CS = Formex(pattern(cross_section_2d[cross_section]))
    elif cross_section in cross_section_3d:
        CS = Formex(mpattern(cross_section_3d[cross_section]))
    if cross_rotate :
        CS = CS.rotate(cross_rotate)
    return CS
    

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

if spread:
    at = PL.atLength(PL.nparts)
    X = PL.pointsAt(at)
    PL = curve.PolyLine(X)
    clear()
    drawSpiralCurves(PL,nwires,blue,red)


if not sweep:
    exit()
    
CS = createCrossSection()
draw(CS)
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


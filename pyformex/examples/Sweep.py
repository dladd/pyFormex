#!/usr/bin/env pyformex --gui
"""Spirals

level = 'normal'
topics = ['geometry','curves']
techniques = ['curve','sweep','mesh']
"""

from plugins import curve
import simple
import re

linewidth(2)
clear()

rfuncs = {
    'constant': 'a',
    'linear (Archimedes)': 'a*x+b',
    'quadratic' : 'a*x**2+b*x+c',
    'exponential (equi-angular)' : 'a*exp(b*x)',
    'custom' : 'a*sqrt(x)',
}

spiral_data = [
    ('nmod',100,{'text':'Number of cells along spiral'}),
    ('turns',2.5,{'text':'Number of 360 degree turns'}),
#    ('rfunc',None,{'text':'Spiral function','choices':rfuncs}),
    ('spiral3d',0.0,{'text':'Out of plane factor'}),
    ('spread',False,{'text':'Spread points evenly along spiral'}),
    ('nwires',1,{'text':'Number of spirals'}),
    ('sweep',False,{'text':'Sweep a cross section along the spiral'}),
    ]


# get the plane line patterns from simple module
cross_sections_2d = {}
for cs in simple.Pattern:
    if re.search('[a-zA-Z]',cs) is None:
        cross_sections_2d[cs] = simple.Pattern[cs]
# add some more patterns
cross_sections_2d.update({
    'channel' : '1223',
    'sigma' : '16253',
    'H-beam' : '11/322/311',
    })
# define some plane surface patterns
cross_sections_3d = {
    'filled_square':'123',
    'filled_triangle':'12',
    }


sweep_data = [
    ('cross_section',None,'select',{'text':'Shape of cross section','choices':cross_sections_2d.keys()+cross_sections_3d.keys()}),
    ('cross_rotate',0.,{'text':'Cross section rotation angle before sweeping'}),
    ('cross_scale',0.,{'text':'Cross section scaling factor'}),
    ]


input_data = {
    'Spiral Data' : spiral_data,
    'Sweep Data' : sweep_data,
}

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
    if cross_section in cross_sections_2d:
        CS = Formex(pattern(cross_sections_2d[cross_section]))
    elif cross_section in cross_sections_3d:
        CS = Formex(mpattern(cross_sections_3d[cross_section]))
    if cross_rotate :
        CS = CS.rotate(cross_rotate)
    if cross_scale:
        CS = CS.scale(cross_scale)
    # Return a Mesh, because that has a 'sweep' function
    CS = CS.swapAxes(0,2).toMesh()
    return CS


def createSpiralCurve(turns,nmod):
    F = Formex(origin()).replic(nmod,1.,0).scale(turns*2*pi/nmod)

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

    return PL
    


def show():
    """Accept the data and draw according to them"""
    clear()
    dialog.acceptData()
    res = dialog.results
    globals().update(res)

    PL = createSpiralCurve(turns,nmod)
    drawSpiralCurves(PL,nwires,red,blue)

    if spread:
        at = PL.atLength(PL.nparts)
        X = PL.pointsAt(at)
        PL = curve.PolyLine(X)
        clear()
        drawSpiralCurves(PL,nwires,blue,red)


    if not sweep:
        return

    CS = createCrossSection()
    draw(CS)

    structure = CS.sweep(PL,normal=0,upvector=None,avgdir=True)
    clear()
    draw(structure,color=yellow)

    if nwires > 1:
        structure = structure.toFormex().rosette(nwires,360./nwires).toMesh()
        smoothwire()
        draw(structure,color='orange')


def close():
    global dialog
    if dialog:
        dialog.close()
        dialog = None


def timeOut():
    """What to do on a InputDialog timeout event.

    As a policy, all pyFormex examples should behave well on a
    dialog timeout.
    Most users can simply ignore this.
    """
    show()
    close()
    
# Create the modeless dialog widget
dialog = widgets.InputDialog(input_data,caption='Sweep Dialog',actions = [('Close',close),('Show',show)],default='Show')
# The examples style requires a timeout action
dialog.timeout = timeOut
# Show the dialog and let the user have fun
dialog.show()

# End


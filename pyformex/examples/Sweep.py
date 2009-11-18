#!/usr/bin/env pyformex --gui
"""Spirals

level = 'normal'
topics = ['geometry','curves']
techniques = ['curve','sweep','mesh']
"""


def vectorRotation(vec1,vec2,upvec=[0.,0.,1.]):
    """Return axis and angle to rotate vectors in a parallel to b

    vectors in a and b should be unit vectors.
    The returned axis is the cross product of a and b. If the vectors
    are already parallel, a random vector normal to a is returned.
    """
    u = normalize(vec1)
    u1 = normalize(vec2)
    v = normalize(upvec)
    v1 = v
    w = cross(u,v)
    w1 = cross(u1,v)
    wa = where(length(w) == 0.)[0]
    wa1 = where(length(w1) == 0.)[0]
    print u
    print u1
    print v
    print v1
    print w
    print w1
    print len(wa)
    print len(wa1)
    if len(wa) > 0 or len(wa1) > 0:
        print wa,wa1
        raise
    ## U = column_stack(
    ## C = linalg.solve(U,B)


vec1 = array([
    [ 0.06151652 , 0.97150612 , 0.22889221],
    [-0.0847166  , 0.97068906 , 0.22491293],
    [-0.36283585 , 0.9072879  , 0.21255307],
    [-0.60093606 , 0.7734409  , 0.20165572],
    [-0.78529388 , 0.58856845 , 0.19209583],
    [-0.90958768 , 0.37270108 , 0.18369579],
    [-0.97370327 , 0.14436606 , 0.17623945],
    [-0.98221517 ,-0.08080836 , 0.16947962],
    [-0.94275808 ,-0.29084417 , 0.16314666],
    [-0.907462   ,-0.3887904  , 0.15923157]])
vec2 = array([[1.,0.,0.]])

vectorRotation(vec1,vec2)
exit()


from plugins import curve
import simple
import re

linewidth(2)
clear()

rfuncs = [
    'linear (Archimedes)',
    'quadratic',
    'exponential (equi-angular)',
    'constant',
#    'custom',
]

spiral_data = [
    ['nmod',100,{'text':'Number of cells along spiral'}],
    ['turns',2.5,{'text':'Number of 360 degree turns'}],
    ['rfunc',None,{'text':'Spiral function','choices':rfuncs}],
    ['coeffs',(1.,0.5,0.2),{'text':'Coefficients in the spiral function'}],
    ['spiral3d',0.0,{'text':'Out of plane factor'}],
    ['spread',False,{'text':'Spread points evenly along spiral'}],
    ['nwires',1,{'text':'Number of spirals'}],
    ['sweep',False,{'text':'Sweep a cross section along the spiral'}],
    ]


# get the plane line patterns from simple module
cross_sections_2d = {}
for cs in simple.Pattern:
    if re.search('[a-zA-Z]',simple.Pattern[cs]) is None:
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
    ['cross_section',None,'select',{'text':'Shape of cross section','choices':cross_sections_2d.keys()+cross_sections_3d.keys()}],
    ['cross_rotate',0.,{'text':'Cross section rotation angle before sweeping'}],
    ['cross_upvector','None',{'text':'Cross section vector that keeps its orientation'}],
    ['cross_scale',0.,{'text':'Cross section scaling factor'}],
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
    # Convert to Formex, because that has a rosette() method
    PL = PL.toFormex()
    if nwires > 1:
        PL = PL.rosette(nwires,360./nwires)
    draw(PL,color=color1)
    draw(PL.points(),color=color2)


def createCrossSection():
    if cross_section in cross_sections_2d:
        CS = Formex(pattern(cross_sections_2d[cross_section]))
    elif cross_section in cross_sections_3d:
        CS = Formex(mpattern(cross_sections_3d[cross_section]))
    if cross_rotate :
        CS = CS.rotate(cross_rotate)
    if cross_scale:
        CS = CS.scale(cross_scale)
    # Convert to Mesh, because that has a sweep() method
    CS = CS.swapAxes(0,2).toMesh()
    return CS


def createSpiralCurve(turns,nmod):
    F = Formex(origin()).replic(nmod,1.,0).scale(turns*2*pi/nmod)
    a,b,c = coeffs
    rfunc_defs = {
        'constant':                    lambda x: a,
        'linear (Archimedes)':         lambda x: a + b*x,
        'quadratic' :                  lambda x: a + b*x + c*x*x,
        'exponential (equi-angular)' : lambda x: a + b * exp(c*x),
#        'custom' :                     lambda x: a + b * sqrt(c*x),
    }

    rf = rfunc_defs[rfunc]
    if spiral3d:
        zf = lambda x : spiral3d * rf(x)
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

    structure = CS.sweep(PL,normal=[1.,0.,0.],upvector=eval(cross_upvector),avgdir=True)
    clear()
    draw(structure,color='red')

    if nwires > 1:
        structure = structure.toFormex().rosette(nwires,360./nwires).toMesh()
        smoothwire()
        draw(structure,color='orange')


def close():
    global dialog
    GD.PF['Sweep_data'] = dialog.results
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


# Update the data items from saved values
saved_data = GD.PF.get('Sweep_data',{})
print saved_data
print input_data
widgets.updateDialogItems(input_data,GD.PF.get('Sweep_data',{}))
print input_data
# Create the modeless dialog widget
dialog = widgets.InputDialog(input_data,caption='Sweep Dialog',actions = [('Close',close),('Show',show)],default='Show')
# The examples style requires a timeout action
dialog.timeout = timeOut
# Show the dialog and let the user have fun
dialog.show()

# End


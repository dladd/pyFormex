#!/usr/bin/env python pyformex.py

"""TestDraw

Example for testing the low level drawing functions

level = 'normal'
topics = ['geometry','mesh','drawing','color',]
techniques = ['widgets','modeless dialog','random']
"""

from numpy.random import rand
    
setDrawOptions({'clear':True, 'bbox':'auto'})
linewidth(2) # The linewidth option is not working nyet

geom_mode = [ 'Formex','Mesh' ]
plexitude = [ 1,2,3,4,5,6,8 ]
element_type = [ 'auto', 'tet4', 'wedge6', 'hex8' ]
color_mode = [ 'none', 'single', 'element', 'vertex' ]

# The points used for a single element of plexitude 1..8
Points = {
    1: [[0.,0.,0.]],
    2: [[0.,0.,0.],[1.,0.,0.]],
    3: [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],
    4: [[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]],
    5: [[0.,0.,0.],[1.,0.,0.],[1.5,0.5,0.],[1.,1.,0.],[0.,1.,0.]],
    6: [[0.,0.,0.],[1.,0.,0.],[1.5,0.5,0.],[1.,1.,0.],[0.,1.,0.],[-0.2,0.5,0.]],
    'tet4': [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
    'wedge6': [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]],
    'hex8': [[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.]],
    }


def select_geom(geom,nplex,eltype):
    """Construct the geometry"""
    try:
        nplex = int(eltype[-1])
    except:
        if not nplex in Points.keys():
            nplex = max([i for i in plexitude if i in Points.keys()])
        eltype = None
    if eltype is None:
        x = Points[nplex]
    else:
        x = Points[eltype]
    F = Formex([x],eltype=eltype).replic2(2,2,2.,2.)
    if geom == 'Formex':
        return F
    else:
        return F.toMesh()
    

def select_color(F,color):
    """Create a set of colors for object F"""
    if color == 'single':
        shape = (1,3)
    elif color == 'element':
        shape = (F.nelems(),3)
    elif color == 'vertex':
        shape = (F.nelems(),F.nplex(),3)
    else:
        return None
    return rand(*shape)

geom = 'Formex'
nplex = 3
eltype = 'auto'
color = 'element'
pos = None
items = [('Geometry Model',geom,'radio',geom_mode),
         ('Plexitude',nplex,'select',plexitude),
         ('Element Type',eltype,'select',element_type),
         ('Color Mode',color,'select',color_mode),
         ]

dialog = None

def show():
    """Accept the data and draw according to them"""
    dialog.acceptData()
    res = dialog.results
    globals().update(dict(
        geom = res[items[0][0]],
        nplex = int(res[items[1][0]]),
        eltype = res[items[2][0]],
        color = res[items[3][0]],
        ))

    G = select_geom(geom,nplex,eltype)
    print "GEOM: nelems=%s, nplex=%s" % (G.nelems(),G.nplex())
    C = select_color(G,color)
    print "COLORS: shape=%s" % str(C.shape)
    draw(G,color=C,clear=True)


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
    showAll()
    close()

# Create the non-modal dialog widget and show it
dialog = widgets.InputDialog(items,caption='Drawing parameters',actions = [('Close',close),('Show',show)],default='Show')
dialog.timeout = timeOut
dialog.show()
        
# End

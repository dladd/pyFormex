#!/usr/bin/env pyformex --gui
# $Id$

import globaldata as GD
#from plugins.postproc import *
from gui.colorscale import ColorScale,ColorLegend
from gui import decors,canvas,widgets
from gui.draw import *
import utils

import commands

from numpy import *
from formex import *
from gui.draw import *
from gui.colors import *


DB = None
R = None

def setDB(db):
    global DB
    DB = db
    

def niceNumber(f,approx=floor):
    """Returns a nice number close to but not smaller than f."""
    n = int(approx(log10(f)))
    m = int(str(f)[0])
    return m*10**n


def frameScale(nframes=10,cycle='up',shape='linear'):
    """Return a sequence of scale values between -1 and +1.

    nframes is the number of steps between 0 and |1| values.

    cycle determines how subsequent cycles occur:
      'up' : ramping up
      'updown': ramping up and down
      'revert': ramping up and down then reverse up and down

    shape determines the shape of the amplitude curve:
      'linear': linear scaling
      'sine': sinusoidal scaling
    """
    s = arange(nframes+1)
    if cycle in [ 'updown', 'revert' ]:
        s = concatenate([s, fliplr(s[:-1].reshape((1,-1)))[0]])
    if cycle in [ 'revert' ]: 
        s = concatenate([s, -fliplr(s[:-1].reshape((1,-1)))[0]])
    return s.astype(float)/nframes

#############################################################
# Do something with the data
# These function should be moved to a more general postprocessor
#

def postABQ():
    """Translate an Abaqus .fil file in a postproc script."""
    types = [ 'Abaqus results file (*.fil)' ]
    fn = askFilename(GD.cfg['workdir'],types,exist=True)
    if fn:
        chdir(fn)
        name,ext = os.path.splitext(fn)
        post = name+'_post.py'
        cmd = "%s/postabq/postabq %s > %s" % (GD.cfg['pyformexdir'],fn,post)
        print cmd
        sta,out = utils.runCommand(cmd)
        if sta:
            GD.message(out)
        

def importDB():
    types = [ 'Postproc scripts (*_post.py)' ]
    fn = askFilename(GD.cfg['workdir'],types,exist=True)
    if fn:
        chdir(fn)
        play(fn)
        if GD.PF.has_key('DB'):
            setDB(GD.PF['DB'])
            GD.message(DB.about['heading'])


def showModel(nodes=True,elems=True):
    if nodes:
        Fn = Formex(DB.nodes)
        draw(Fn)
    if elems:
        Fe = [ Formex(DB.nodes[elems],i+1) for i,elems in enumerate(DB.elems.itervalues()) ]
        draw(Fe)
    zoomAll()


def selectStepInc():
    res = askItems([('Step',DB.step,'select',DB.res.keys())])
    if res:
        step = int(res['Step'])
        res = askItems([('Increment',None,'select',DB.res[step].keys())])
        if res:
            inc = int(res['Increment'])
    GD.message("Step %s; Increment %s;" % (step,inc))
    DB.setStepInc(step,inc)
        

def showResults(nodes,elems,displ,text,val,showref=False,dscale=100.,
                count=1,sleeptime=-1.):
    """Display a constant or linear field on triangular elements.

    nodes is an array with nodal coordinates
    elems is a single element group or a list of elem groups
    displ are the displacements at the nodes
    val are the scalar values at the nodes

    If dscale is a list of values, the results will be drawn with
    subsequent deformation scales, with a sleeptime intermission,
    and the whole cycle will be repeated count times.
    """
    clear()
    
    if type(elems) != list:
        elems = [ elems ]

    # draw undeformed structure
    if showref:
        ref = [ Formex(nodes[el]) for el in elems ]
        draw(ref,bbox=None,color='green',linewidth=1,mode='wireframe')

    # compute the colors according to the values
    if val is not None:
        # create a colorscale and draw the colorlegend
        vmin,vmax = val.min(),val.max()
        if vmin*vmax < 0.0:
            vmid = 0.0
        else:
            vmid = 0.5*(vmin+vmax)
        CS = ColorScale([blue,green,red],vmin,vmax,vmid,1.,1.)
##         CS = ColorScale([green,None,magenta],0.,1.,None,0.5,None)
        cval = array(map(CS.color,val))
        CL = ColorLegend(CS,100)
        CLA = decors.ColorLegend(CL,10,20,30,200) 
        GD.canvas.addDecoration(CLA)

    # the supplied text
    if text:
        drawtext(text,150,30,'tr24')

    smooth()
    lights(False)

    # create the frames while displaying them
    dscale = array(dscale)
    frames = []   # a place to store the drawn frames
    for dsc in dscale.flat:

        dnodes = nodes + dsc * displ
        deformed = [ Formex(dnodes[el]) for el in elems ]

        # We store the changing parts of the display, so that we can
        # easily remove/redisplay them
        if val is None:
            F = [ draw(df,color='blue',view='__last__',wait=None) for df in deformed ]
        else:
            F = [ draw(df,color=cval[el],view='__last__',wait=None) for df,el in zip(deformed,elems) ]
        T = drawtext('Deformation scale = %s' % dsc,150,10,'tr18')

        # remove the last frame
        # This is a clever trick: we remove the old drawings only after
        # displaying new ones. This makes the animation a lot smoother
        # (though the code is less clear and compact).
        if len(frames) > 0:
            for Fi in frames[-1][0]:
                GD.canvas.removeActor(Fi)
            GD.canvas.removeDecoration(frames[-1][1])
        # add the latest frame to the stored list of frames
        frames.append((F,T))
        if sleeptime > 0.:
            sleep(sleeptime)

    # display the remaining cycles
    count -= 1
    FA,TA = frames[-1]
    while count > 0:
        count -= 1

        for F,T in frames:
            # It would be interesting if addactor would add/remove a list
            # of actors
            for Fi in F:
                GD.canvas.addActor(Fi)
            GD.canvas.addDecoration(T)
            for Fi in FA:
                GD.canvas.removeActor(Fi)
            GD.canvas.removeDecoration(TA)
            GD.canvas.display()
            GD.canvas.update()
            FA,TA = F,T
            if sleeptime > 0.:
                sleep(sleeptime)


def postProc():
    """Show results from the analysis."""
    
    results = [
        ('','None'),
        ('U','Displacement'),
        ('U0','X-Displacement'),
        ('U1','Y-Displacement'),
        ('U2','Z-Displacement'),
        ('S0','X-Normal Stress'),
        ('S1','Y-Normal Stress'),
        ('S2','Z-Normal Stress'),
        ('S3','XY-Shear Stress'),
        ('S4','XZ-Shear Stress'),
        ('S5','YZ-Shear Stress'),
        ('Computed','Distance from a point'),
        ]
    # split in two lists
    res_keys = [ c[0] for c in results ]
    res_desc = [ c[1] for c in results ]

    # Ask the user which results he wants
    print DB.elems.keys()
    print res_desc
    res = askItems([('Element Group','All','select',['All',]+DB.elems.keys()),
                    ('Type of result',None,'select',res_desc),
                    ('Load case',0),
                    ('Autocalculate deformation scale',True),
                    ('Deformation scale',100.),
                    ('Show undeformed configuration',False),
                    ('Animate results',False),
                    ('Amplitude shape','linear','select',['linear','sine']),
                    ('Animation cycle','updown','select',['up','updown','revert']),
                    ('Number of cycles',5),
                    ('Number of frames',10),
                    ('Animation sleeptime',0.1),
                    ])
    if not res:
        return None

    # Show results
    nodes = DB.nodes
    elgrp = res['Element Group']
    if elgrp == 'All':
        elems = DB.elems.values()
    else:
        elems = [ DB.elems[elgrp] ]
    resindex = res_desc.index(res['Type of result'])
    loadcase = res['Load case']
    autoscale = res['Autocalculate deformation scale']
    dscale = res['Deformation scale']
    showref = res['Show undeformed configuration']
    animate = res['Animate results']
    shape = res['Amplitude shape']
    cycle = res['Animation cycle']
    count = res['Number of cycles']
    nframes = res['Number of frames']
    sleeptime = res['Animation sleeptime']

    displ = DB.getres('U')
    if displ is not None:
        displ = displ[:,0:3]
        if autoscale:
            siz0 = Coords(nodes).sizes()
            siz1 = Coords(displ).sizes()
            w = where(siz0 > 0.0)[0]
            dscale = niceNumber(1./(siz1[w]/siz0[w]).max())

    if animate:
        dscale = dscale * frameScale(nframes,cycle=cycle,shape=shape) 

    # Get the scalar element result values from the results.
    txt = 'No Results'
    val = None
    if resindex > 0:
        key = res_keys[resindex]
        print "RESULT KEY = %s" % key
        if key == 'Computed':
            if askPoint():
                val = Coords(nodes).distanceFromPoint(point)
        else:
            val = DB.getres(key)
    if val is not None:
        txt = res_desc[resindex]
    showResults(nodes,elems,displ,txt,val,showref,dscale,count,sleeptime)
    return val


def DistanceFromPoint(nodes,pt):
    """Show distance from origin rendered on the domain of triangles"""
    val = Fn.distanceFromPoint(pt)
##     nodes = DB.nodes
##     displ = zeros(nodes.shape)
##     text = "Distance from point %s" % pt
##     showResults(nodes,elems,displ,text,val,showref=False,dscale=100.,
##                 count=1,sleeptime=-1.)

        
point = [3.,2.,0.]

def askPoint():
    global point
    res = askItems([('Point',point)])
    if res:
        point = res['Point']
        return point
    else:
        return None
    

################### menu #################

def create_menu():
    """Create the Postproc menu."""
    MenuData = [
        ("&Open Postproc Database",importDB),
        ("&Translate Abaqus .fil results file",postABQ),
        ("---",None),
        ("Show Geometry",showModel),
        ("Select Step/Inc",selectStepInc),
        ("Show Results",postProc),
        ("---",None),
        ("&Reload menu",reload_menu),
        ("&Close menu",close_menu),
        ]
    return widgets.Menu('Postproc',items=MenuData,parent=GD.gui.menu,before='help')
  
def show_menu():
    """Show the Postproc menu."""
    if not GD.gui.menu.item('Postproc'):
        create_menu()

def close_menu():
    """Close the Postproc menu."""
    m = GD.gui.menu.item('Postproc')
    if m :
        m.remove()

def reload_menu():
    """Reload the Postproc menu."""
    close_menu()
    show_menu()


if __name__ == "draw":
    reload_menu()
    
elif __name__ == "__main__":
    print __doc__

# End


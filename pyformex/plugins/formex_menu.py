#!/usr/bin/env python pyformex
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""formex_menu.py

This is a pyFormex plugin. It is not intended to be executed as a script,
but to be loaded as a plugin.
"""

import pyformex as GD
from gui import actors
from gui.draw import *
from formex import *


from plugins import objects,surface,inertia,partition,sectionize
from pyformex.arraytools import niceLogSize

import commands, os, timer


##################### select, read and write ##########################

selection = objects.DrawableObjects(clas=Formex)


setSelection = selection.set
drawSelection = selection.draw


def read_Formex(fn):
    GD.message("Reading file %s" % fn)
    t = timer.Timer()
    F = Formex.read(fn)
    nelems,nplex = F.f.shape[0:2]
    GD.message("Read %d elems of plexitude %d in %s seconds" % (nelems,nplex, t.seconds()))
    return F


def readSelection(select=True,draw=True,multi=True):
    """Read a Formex (or list) from asked file name(s).

    If select is True (default), this becomes the current selection.
    If select and draw are True (default), the selection is drawn.
    """
    types = [ 'Formex Files (*.formex)', 'All Files (*)' ]
    fn = askFilename(GD.cfg['workdir'],types,multi=multi)
    if fn:
        if not multi:
            fn = [ fn ]
        chdir(fn[0])
        names = map(utils.projectName,fn)
        GD.GUI.setBusy()
        F = map(read_Formex,fn)
        GD.GUI.setBusy(False)
        export(dict(zip(names,F)))
        if select:
            GD.message("Set selection to %s" % str(names))
            selection.set(names)
            if draw:
                selection.draw()
    return fn
    

def printSize():
    for s in selection.names:
        S = named(s)
        GD.message("Formex %s has shape %s" % (s,S.shape()))


def writeSelection():
    """Writes the currently selected Formex to .formex file."""
    F = selection.check(single=True)
    if F:
        name = selection.names[0]
        fn = askNewFilename(os.path.join(GD.cfg['workdir'],"%s.formex" % name),
                         filter=['(*.formex)','*'])
        if fn:
            if not fn.endswith('.formex'):
                fn += '.formex'
            GD.message("Writing Formex '%s' to file '%s'" % (name,fn))
            chdir(fn)
            F.write(fn)


################### Change attributes of Formex #######################


def shrink():
    """Toggle the shrink mode"""
    if selection.shrink is None:
        selection.shrink = 0.8
    else:
        selection.shrink = None
    selection.draw()


#################### CoordPlanes ####################################

_bbox = None

def showBbox():
    """Draw the bbox on the current selection."""
    global _bbox
    FL = selection.check()
    if FL:
        bb = bbox(FL)
        GD.message("Bbox of selection: %s" % bb)
        nx = array([4,4,4])
        _bbox = actors.CoordPlaneActor(nx=nx,ox=bb[0],dx=(bb[1]-bb[0])/nx)
        GD.canvas.addAnnotation(_bbox)
        GD.canvas.update()

def removeBbox():
    """Remove the bbox of the current selection."""
    global _bbox
    if _bbox:
        GD.canvas.removeAnnotation(_bbox)
        _bbox = None


#################### Axes ####################################

def unitAxes():
    """Create a set of three axes."""
    Hx = Formex(pattern('1'),5).translate([-0.5,0.0,0.0])
    Hy = Hx.rotate(90)
    Hz = Hx.rotate(-90,1)
    Hx.setProp(4)
    Hy.setProp(5)
    Hz.setProp(6)
    return Formex.concatenate([Hx,Hy,Hz])    


def showPrincipal():
    """Show the principal axes."""
    F = selection.check(single=True)
    if not F:
        return
    # compute the axes
    C,I = inertia.inertia(F.f)
    GD.message("Center of gravity: %s" % C)
    GD.message("Inertia tensor: %s" % I)
    Iprin,Iaxes = inertia.principal(I)
    GD.message("Principal Values: %s" % Iprin)
    GD.message("Principal Directions: %s" % Iaxes)
    data = (C,I,Iprin,Iaxes)
    # now display the axes
    siz = F.dsize()
    H = unitAxes().scale(1.1*siz).affine(Iaxes.transpose(),C)
    A = 0.1*siz * Iaxes.transpose()
    G = Formex([[C,C+Ax] for Ax in A],3)
    draw([G,H])
    export({'principalAxes':H,'_principal_data_':data})
    return data


def rotatePrincipal():
    """Rotate the selection according to the last shown principal axes."""
    try:
        data = named('_principal_data_')
    except:
        data = showPrincipal() 
    FL = selection.check()
    if FL:
        ctr = data[0]
        rot = data[3]
        selection.changeValues([ F.trl(-ctr).rot(rot).trl(ctr) for F in FL ])
        selection.drawChanges()


def transformPrincipal():
    """Transform the selection according to the last shown principal axes.

    This is analog to rotatePrincipal, but positions the Formex at its center.
    """
    try:
        data = named('_principal_data_')
    except:
        data = showPrincipal() 
    FL = selection.check()
    if FL:
        ctr = data[0]
        rot = data[3]
        selection.changeValues([ F.trl(-ctr).rot(rot) for F in FL ])
        selection.drawChanges()



################### Perform operations on Formex #######################
    

def scaleSelection():
    """Scale the selection."""
    FL = selection.check()
    if FL:
        res = askItems([['scale',1.0]],
                       caption = 'Scale Factor')
        if res:
            scale = float(res['scale'])
            selection.changeValues([ F.scale(scale) for F in FL ])
            selection.drawChanges()

            
def scale3Selection():
    """Scale the selection with 3 scale values."""
    FL = selection.check()
    if FL:
        res = askItems([['x-scale',1.0],['y-scale',1.0],['z-scale',1.0]],
                       caption = 'Scaling Factors')
        if res:
            scale = map(float,[res['%c-scale'%c] for c in 'xyz'])
            selection.changeValues([ F.scale(scale) for F in FL ])
            selection.drawChanges()


def translateSelection():
    """Translate the selection."""
    FL = selection.check()
    if FL:
        res = askItems([['direction',0],['distance','1.0']],
                       caption = 'Translation Parameters')
        if res:
            dir = int(res['direction'])
            dist = float(res['distance'])
            selection.changeValues([ F.translate(dir,dist) for F in FL ])
            selection.drawChanges()


def centerSelection():
    """Center the selection."""
    FL = selection.check()
    if FL:
        selection.changeValues([ F.translate(-F.center()) for F in FL ])
        selection.drawChanges()


def rotateSelection():
    """Rotate the selection."""
    FL = selection.check()
    if FL:
        res = askItems([['axis',2],['angle','90.0']])
        if res:
            axis = int(res['axis'])
            angle = float(res['angle'])
            selection.changeValues([ F.rotate(angle,axis) for F in FL ])
            selection.drawChanges()


def rotateAround():
    """Rotate the selection."""
    FL = selection.check()
    if FL:
        res = askItems([['axis',2],['angle','90.0'],['around','[0.0,0.0,0.0]']])
        if res:
            axis = int(res['axis'])
            angle = float(res['angle'])
            around = eval(res['around'])
            GD.debug('around = %s'%around)
            selection.changeValues([ F.rotate(angle,axis,around) for F in FL ])
            selection.drawChanges()


def rollAxes():
    """Rotate the selection."""
    FL = selection.check()
    if FL:
        selection.changeValues([ F.rollAxes() for F in FL ])
        selection.drawChanges()
            
        
def clipSelection():
    """Clip the selection."""
    FL = selection.check()
    if FL:
        res = askItems([['axis',0],['begin',0.0],['end',1.0],['nodes','all','select',['all','any','none']]],caption='Clipping Parameters')
        if res:
            bb = bbox(FL)
            axis = int(res['axis'])
            xmi = bb[0][axis]
            xma = bb[1][axis]
            dx = xma-xmi
            xc1 = xmi + float(res['begin']) * dx
            xc2 = xmi + float(res['end']) * dx
            selection.changeValues([ F.clip(F.test(nodes=res['nodes'],dir=axis,min=xc1,max=xc2)) for F in FL ])
            selection.drawChanges()
        

def cutSelection():
    """Cut the selection with a plane."""
    FL = selection.check()
    FLnot = [ F for F in FL if F.nplex() not in [2,3] ]
    if FLnot:
        warning("Currently I can only cut Formices with plexitude 2 or 3.\nPlease change your selection.")
        return
    
    dsize = bbox(FL).dsize()
    if dsize > 0.:
        esize = 10 ** (niceLogSize(dsize)-5)
    else:
        esize = 1.e-5
    
    res = askItems([['Point',(0.0,0.0,0.0)],
                    ['Normal',(1.0,0.0,0.0)],
                    ['New props',[1,2,2,3,4,5,6]],
                    ['Side','positive', 'radio', ['positive','negative','both']],
                    ['Tolerance',esize],
                    ],caption = 'Define the cutting plane')
    if res:
        P = res['Point']
        N = res['Normal']
        atol = res['Tolerance']
        p = res['New props']
        side = res['Side']
        if side == 'both':
            G = [F.cutWithPlane(P,N,side=side,atol=atol,newprops=p) for F in FL]
            #print G[0][0].p
            draw(G[0])
            G_pos = [ g[0] for g in G ]
            G_neg = [ g[1] for g in G ]
            export(dict([('%s/pos' % n,g) for n,g in zip(selection,G_pos)]))
            export(dict([('%s/neg' % n,g) for n,g in zip(selection,G_neg)]))
            selection.set(['%s/pos' % n for n in selection] + ['%s/neg' % n for n in selection])
            selection.draw()
        else:
            selection.changeValues([ F.cutWithPlane(P,N,side=side,atol=atol,newprops=p) for F in FL ])
            selection.drawChanges()


def concatenateSelection():
    """Concatenate the selection."""
    FL = selection.check()
    if FL:
        plexitude = array([ F.nplex() for F in FL ])
        if plexitude.min() == plexitude.max():
            res = askItems([['name','combined']],'Name for the concatenation')
            if res:
                name = res['name']
                export({name:Formex.concatenate(FL)})
                selection.set(name)
                selection.draw()
        else:
            warning('You can only concatenate Formices with the same plexitude!')
    

def partitionSelection():
    """Partition the selection."""
    F = selection.check(single=True)
    if not F:
        return

    name = selection[0]
    GD.message("Partitioning Formex '%s'" % name)
    cuts = partition.partition(F)
    GD.message("Subsequent cutting planes: %s" % cuts)
    if ack('Save cutting plane data?'):
        types = [ 'Text Files (*.txt)', 'All Files (*)' ]
        fn = askNewFilename(GD.cfg['workdir'],types)
        if fn:
            chdir(fn)
            fil = file(fn,'w')
            fil.write("%s\n" % cuts)
            fil.close()
    

def createParts():
    """Create parts of the current selection, based on property values."""
    F = selection.check(single=True)
    if not F:
        return

    name = selection[0]
    partition.splitProp(F,name)


def sectionizeSelection():
    """Sectionize the selection."""
    F = selection.check(single=True)
    if not F:
        return

    name = selection[0]
    GD.message("Sectionizing Formex '%s'" % name)
    ns,th,segments = sectionize.createSegments(F)
    if not ns:
        return
    
    sections,ctr,diam = sectionize.sectionize(F,segments,th)
    #GD.message("Centers: %s" % ctr)
    #GD.message("Diameters: %s" % diam)
    if ack('Save section data?'):
        types = [ 'Text Files (*.txt)', 'All Files (*)' ]
        fn = askNewFilename(GD.cfg['workdir'],types)
        if fn:
            chdir(fn)
            fil = file(fn,'w')
            fil.write("%s\n" % ctr)
            fil.write("%s\n" % diam)
            fil.close()
    if ack('Draw circles?'):
        circles = sectionize.drawCircles(sections,ctr,diam)
        ctrline = sectionize.connectPoints(ctr)
        if ack('Draw circles on Formex ?'):
            sectionize.drawAllCircles(F,circles)
        circles = Formex.concatenate(circles)
        circles.setProp(3)
        ctrline.setProp(1)
        draw(ctrline,color='red')
        export({'circles':circles,'ctrline':ctrline,'flypath':ctrline})
        if ack('Fly through the Formex ?'):
            flyAlong(ctrline)
##        if ack('Fly through in smooth mode ?'):
##            smooth()
##            flytruCircles(ctr)
    selection.draw()



################### menu #################

def create_menu():
    """Create the Formex menu."""
    MenuData = [
        ("&Read Formex Files",readSelection),
        ("&Select",selection.ask),
        ("&Draw Selection",selection.draw),
        ("&Forget Selection",selection.forget),
        ("&Save Selection as Formex",writeSelection),
        ("---",None),
        ("Print &Information",
         [('&Data Size',printSize),
          ('&Bounding Box',selection.printbbox),
          ]),
        ("&Set Property",selection.setProperty),
        ("&Shrink",shrink),
        ("Toggle &Annotations",
         [("&Names",selection.toggleNames,dict(checkable=True)),
          ("&Numbers",selection.toggleNumbers,dict(checkable=True)),
          ('&Toggle Bbox',selection.toggleBbox,dict(checkable=True)),
          ]),
        ("---",None),
        ("&Bbox",
         [('&Show Bbox Planes',showBbox),
          ('&Remove Bbox Planes',removeBbox),
          ]),
        ("&Transform",
         [("&Scale Selection",scaleSelection),
          ("&Scale non-uniformly",scale3Selection),
          ("&Translate",translateSelection),
          ("&Center",centerSelection),
          ("&Rotate",rotateSelection),
          ("&Rotate Around",rotateAround),
          ("&Roll Axes",rollAxes),
          ]),
        ("&Clip/Cut",
         [("&Clip",clipSelection),
          ("&Cut With Plane",cutSelection),
          ]),
        ("&Undo Last Changes",selection.undoChanges),
        ("---",None),
        ("Show &Principal Axes",showPrincipal),
        ("Rotate to &Principal Axes",rotatePrincipal),
        ("Transform to &Principal Axes",transformPrincipal),
        ("---",None),
        ("&Concatenate Selection",concatenateSelection),
        ("&Partition Selection",partitionSelection),
        ("&Create Parts",createParts),
        ("&Sectionize Selection",sectionizeSelection),
        ("---",None),
        ("&Fly",flyAlong),
        ("---",None),
        ("&Reload menu",reload_menu),
        ("&Close",close_menu),
        ]
    return widgets.Menu('Formex',items=MenuData,parent=GD.GUI.menu,before='help')

    
def show_menu():
    """Show the Tools menu."""
    if not GD.GUI.menu.item('Formex'):
        create_menu()


def close_menu():
    """Close the Tools menu."""
    m = GD.GUI.menu.item('Formex')
    if m :
        m.remove()
      

def reload_menu():
    """Reload the Postproc menu."""
    close_menu()
#    reload(plugins.formex_menu)
    show_menu()


if __name__ == "draw":
    reload_menu()
    
elif __name__ == "__main__":
    print __doc__

# End


#!/usr/bin/env python pyformex
# $Id$
##
## This file is part of pyFormex 0.7 Release Fri Apr  4 18:41:11 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""formex_menu.py

This is a pyFormex plugin. It is not intended to be executed as a script,
but to be loaded as a plugin.
"""

import globaldata as GD
from gui import actors
from gui.draw import *
from formex import *
from plugins import objects,surface,inertia,partition,sectionize

import commands, os, timer


##################### select, read and write ##########################

selection = objects.DrawableObjects(clas=Formex)


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
    fn = askFilename(GD.cfg['workdir'],types,exist=True,multi=multi)
    if not multi:
        fn = [ fn ]
    if fn:
        chdir(fn[0])
        names = map(utils.projectName,fn)
        GD.gui.setBusy()
        F = map(read_Formex,fn)
        GD.gui.setBusy(False)
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
        fn = askFilename(GD.cfg['workdir'],file="%s.formex" % name,
                         filter=['(*.formex)','*'],exist=False)
        if fn:
            GD.message("Writing Formex '%s' to file '%s'" % (name,fn))
            chdir(fn)
            F.write(fn)


## def writeSelectionSTL():
##     """Writes the currently selected Formices to .stl files."""
##     F = selection.check(single=True)
##     if F:
##         name = selection.names[0]
##         fn = askFilename(GD.cfg['workdir'],file="%s.stl" % name,
##                          filter=['(*.stl)','*'],exist=False)
##         if fn:
##             print "Writing Formex '%s' to file '%s'" % (name,fn)
##             print named(name).bbox()
##             chdir(fn)
##             surface.write_stla(fn,named(name).f)



################### Change attributes of Formex #######################



def shrink():
    """Toggle the shrink mode"""
    if selection.shrink is None:
        selection.shrink = 0.8
    else:
        selection.shrink = None
    selection.draw()
    


#################### CoordPlanes ####################################

bboxB = None

def showBboxB():
    """Draw the bbox on the current selection."""
    global bboxB
    FL = selection.check()
    if FL:
        bb = bbox(FL)
        GD.message("Bbox of selection: %s" % bb)
        nx = array([4,4,4])
        bboxB = actors.CoordPlaneActor(nx=nx,ox=bb[0],dx=(bb[1]-bb[0])/nx)
        GD.canvas.addActor(bboxB)
        GD.canvas.update()

def removeBboxB():
    """Remove the bbox of the current selection."""
    global bboxB
    if bboxB:
        undraw(bboxB)
        bboxB = None


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
    siz = F.diagonal()
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
        

def cutAtPlane():
    """Cut the selection with a plane."""
    FL = selection.check()
    FLnot = [ F for F in FL if F.nplex() > 3 ]
    if FLnot:
        warning("Currently I can only cut Formices with plexitude <= 3.\nPlease change your selection.")
        return
    res = askItems([['Point',(0.0,0.0,0.0)],
                     ['Normal',(0.0,0.0,1.0)],
                     ['Tolerance',0.],
                     ['New props',[1,2,2,3,4,5,6]],
                     ['Side','positive', 'radio', ['positive','negative','both']]],
                     caption = 'Define the cutting plane')
    if res:
        P = res['Point']
        N = res['Normal']
        atol = res['Tolerance']
        p = res['New props']
        side = res['Side']
        if side == 'both':
            G = [F.cutAtPlane(P,N,p,side,atol) for F in FL]
            G_pos = []
            G_neg  =[]
            for F in G:
                G_pos.append(F[0])
                G_neg.append(F[1])
            export(dict([('%s/pos' % n,g) for n,g in zip(selection,G_pos)]))
            export(dict([('%s/neg' % n,g) for n,g in zip(selection,G_neg)]))
            selection.set(['%s/pos' % n for n in selection] + ['%s/neg' % n for n in selection])
            selection.draw()
        else:
            selection.changeValues([ F.cutAtPlane(P,N,atol,p,side) for F in FL ])
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
        fn = askFilename(GD.cfg['workdir'],types,exist=False)
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
        fn = askFilename(GD.cfg['workdir'],types,exist=False)
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


def flyThru():
    """Fly through the structure along the flypath."""
    path = named('flypath')
    if path:
        flyAlong(path)
    else:
        warning("You have to define a flypath first!")
   


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
          ('&Bounding Box',selection.printBbox),
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
         [('&Show Bbox Planes',showBboxB),
          ('&Remove Bbox Planes',removeBboxB),
          ]),
        ("&Transform",
         [("&Scale Selection",scaleSelection),
          ("&Non-uniformly Scale Selection",scale3Selection),
          ("&Translate Selection",translateSelection),
          ("&Center Selection",centerSelection),
          ("&Rotate Selection",rotateSelection),
          ("&Rotate Selection Around",rotateAround),
          ("&Roll Axes",rollAxes),
          ("&Clip Selection",clipSelection),
          ("&Cut at Plane",cutAtPlane),
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
        ("&Fly",flyThru),
        ("---",None),
        ("&Close",close_menu),
        ]
    return widgets.Menu('Formex',items=MenuData,parent=GD.gui.menu,before='help')

    
def show_menu():
    """Show the Tools menu."""
    if not GD.gui.menu.item('Formex'):
        create_menu()


def close_menu():
    """Close the Tools menu."""
    m = GD.gui.menu.item('Formex')
    if m :
        m.remove()
      

if __name__ == "draw":
    # If executed as a pyformex script
    close_menu()
    show_menu()
    
elif __name__ == "__main__":
    print __doc__

# End


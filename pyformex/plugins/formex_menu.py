# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""formex_menu.py

This is a pyFormex plugin. It is not intended to be executed as a script,
but to be loaded as a plugin.
"""

import pyformex as pf
from formex import *
import utils
from odict import ODict

from gui import actors
from gui import menu
from gui.draw import *

from plugins import objects,trisurface,partition,sectionize

import commands, os, timer


##################### select, read and write ##########################

selection = pf.GUI.selection['formex']

setSelection = selection.set
drawSelection = selection.draw


## def readSelection(select=True,draw=True,multi=True):
##     """Read a Formex (or list) from asked file name(s).

##     If select is True (default), this becomes the current selection.
##     If select and draw are True (default), the selection is drawn.
##     """
##     types = utils.fileDescription(['pgf','all'])
##     fn = askFilename(pf.cfg['workdir'],types,multi=multi)
##     if fn:
##         if not multi:
##             fn = [ fn ]
##         chdir(fn[0])
##         res = ODict()
##         pf.GUI.setBusy()
##         for f in fn:
##             res.update(readGeomFile(f))
##         pf.GUI.setBusy(False)
##         export(res)
##         if select:
##             oknames = [ k for k in res if isinstance(res[k],Formex) ]
##             selection.set(oknames)
##             pf.message("Set Formex selection to %s" % oknames)
##             if draw:
##                 selection.draw()
##     return fn


## def writeSelection():
##     """Writes the currently selected Formices to a Geometry File."""
##     F = selection.check()
##     if F:
##         types = utils.fileDescription(['pgf','all'])
##         name = selection.names[0]
##         fn = askNewFilename(os.path.join(pf.cfg['workdir'],"%s.pgf" % name),
##                             filter=types)
##         if fn:
##             if not (fn.endswith('.pgf') or fn.endswith('.formex')):
##                 fn += '.pgf'
##             pf.message("Creating pyFormex Geometry File '%s'" % fn)
##             chdir(fn)
##             selection.writeToFile(fn)
##             pf.message("Contents: %s" % selection.names)
    

## def printSize():
##     for s in selection.names:
##         S = named(s)
##         pf.message("Formex %s has shape %s" % (s,S.shape()))


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
        pf.message("Bbox of selection: %s" % bb)
        nx = array([4,4,4])
        _bbox = actors.CoordPlaneActor(nx=nx,ox=bb[0],dx=(bb[1]-bb[0])/nx)
        pf.canvas.addAnnotation(_bbox)
        pf.canvas.update()

def removeBbox():
    """Remove the bbox of the current selection."""
    global _bbox
    if _bbox:
        pf.canvas.removeAnnotation(_bbox)
        _bbox = None


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
            pf.debug('around = %s'%around)
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
    pf.message("Partitioning Formex '%s'" % name)
    cuts = partition.partition(F)
    pf.message("Subsequent cutting planes: %s" % cuts)
    if ack('Save cutting plane data?'):
        types = [ 'Text Files (*.txt)', 'All Files (*)' ]
        fn = askNewFilename(pf.cfg['workdir'],types)
        if fn:
            chdir(fn)
            fil = open(fn,'w')
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
    pf.message("Sectionizing Formex '%s'" % name)
    ns,th,segments = sectionize.createSegments(F)
    if not ns:
        return
    
    sections,ctr,diam = sectionize.sectionize(F,segments,th)
    #pf.message("Centers: %s" % ctr)
    #pf.message("Diameters: %s" % diam)
    if ack('Save section data?'):
        types = [ 'Text Files (*.txt)', 'All Files (*)' ]
        fn = askNewFilename(pf.cfg['workdir'],types)
        if fn:
            chdir(fn)
            fil = open(fn,'w')
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

def fly():
    path = named('flypath')
    if path is not None:
        flyAlong(path)
    else:
        warning("You should define the flypath first")


################### menu #################

_menu = 'Formex'

def create_menu():
    """Create the Formex menu."""
    MenuData = [
        ("&Shrink",shrink),
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
        ("&Concatenate Selection",concatenateSelection),
        ("&Partition Selection",partitionSelection),
        ("&Create Parts",createParts),
        ("&Sectionize Selection",sectionizeSelection),
        ("---",None),
        ("&Fly",fly),
        ("---",None),
        ("&Reload menu",reload_menu),
        ("&Close",close_menu),
        ]
    return menu.Menu(_menu,items=MenuData,parent=pf.GUI.menu,before='help')

    
def show_menu():
    """Show the Tools menu."""
    if not pf.GUI.menu.item(_menu):
        create_menu()


def close_menu():
    """Close the Tools menu."""
    m = pf.GUI.menu.item(_menu)
    if m :
        m.remove()
      

def reload_menu():
    """Reload the Postproc menu."""
    close_menu()
    show_menu()


####################################################################
######### What to do when the script is executed ###################

if __name__ == "draw":
    reload_menu()

# End


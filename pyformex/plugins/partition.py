#!/usr/bin/env python pyformex.py
# $Id$
##
## This file is part of pyFormex 0.5 Release Fri Aug 10 12:04:07 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

import globaldata as GD
from gui import decors,colors
from gui.camera import inverse
from gui.draw import *
from formex import *
from plugins import surface

VA=None

def prepare(V):
    """Prepare the surface for slicing operation."""
    global VA
    V = V.translate(-V.center())
    P = V.center()
    print "Initial P = %s" % P
    VA = draw(V,bbox=None,color='black')
    area,norm = surface.areaNormals(V.f)
    N = norm[0]
    return V,P,N


def testview(F,V,P):
    global VA,waiting
    p = array(GD.canvas.camera.getCenter())
    waiting = True
    while waiting:
        GD.app.processEvents()
    p -= array(GD.canvas.camera.getCenter())
    print "TRANSLATE: %s" % p
    m = GD.canvas.camera.getRot()
    P += p
    print "TOTAL TRANSLATE: %s" % P
    V = V.affine(inverse(m[0:3,0:3])).translate(-P)
    print V.center()
    print F.center()
    undraw(VA)
    VA = draw(V)
    area,norm = surface.areaNormals(V.f)
    N = norm[0]
    return P,N


def colorCut(F,P,N,prop):
    """Color a Formex in two by a plane (P,N)"""
    print F.bbox()
    print P
    print N
    print prop
    dist = F.distanceFromPlane(P,N)
    print dist
    right = any(dist>0.0,axis=1)
    print right
    F.p[right] = prop
    nright = right.sum()
    nleft = F.nelems() - nright
    print "Left part has %s elements, right part has %s elements" % (nleft,nright)
    return F


def splitProp(F,name):
    """Partition a Formex according to its prop values.

    Returns a dict with the partitions, named like name-prop and exports
    these named Formex instances.
    It the Formex has no props, the whole Formex is given the name.
    """
    if F.p is None:
        d = { name:F }
    else:
        d = dict([['%s-%s' % (name,p),F.withProp(p)] for p in F.propSet()])
    export(d)
    return d


waiting = True

def wakeup():
    global waiting
    waiting = False


def partition(Fin,prop=0):
    """Interactively partition a Formex.

    By default, the parts will get properties 0,1,...
    If prop >= 0, the parts will get incremental props starting from prop.

    Returns the cutplanes in an array with shape (ncuts,2,3), where
      (i,0,:) is a point in the plane i and
      (i,1,:) is the normal vector on the plane i .
    
    As a side effect, the properties of the input Formex will be changed
    to flag the parts between successive cut planes by incrementing
    property values.
    If you wish to restore the original properties, you should copy them
    (or the input Formex) before calling this function.
    """
    global FA,VA,waiting

    # start color
    keepprops = prop
    if prop is None or prop < 0:
        prop = 0

    # Make sure the inital Formex is centered
    initial_trl = -Fin.center()
    F = Fin.translate(initial_trl)

    # Store the inital properties and make all properties equal to start value
    initial_prop = F.p
    F.setProp(prop)

    # draw it
    linewidth(1)
    perspective(False)
    clear()
    FA = draw(F,view='front')

    # create a centered cross plane, large enough
    bb = F.bbox()
    siz = F.sizes().max()
    V = Formex([[[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]],
                [[1.,1.,0.],[0.,1.,0.],[0.,0.,0.]]],
               0)
    V = V.translate(-V.center()).rotate(90,1).scale(siz)

    cut_planes = []

    QtCore.QObject.connect(GD.canvas,QtCore.SIGNAL("Wakeup"),wakeup)

    linewidth(2)
    w,h = GD.canvas.width(),GD.canvas.height()
    fgcolor('magenta')
    SD = decors.Line(w/2,0,w/2,h)
    decorate(SD)
    
    fgcolor(colors.black)

    V,P,N = prepare(V)
    while True:
        res = ask("",["Adjust Cut","Keep Cut", "Finish"])
        if res == "Adjust Cut":
            P,N = testview(F,V,P)
            print "Plane: point %s, normal %s" % (P,N)
        elif res == "Keep Cut":
            undraw(FA)
            #undraw(VA)
            cut_planes.append((P,N))
            prop += 1
            F = colorCut(F,-P,N,prop)
            FA = draw(F)
            undraw(VA)
            V,P,N = prepare(V)
        else:
            break
        
    QtCore.QObject.disconnect(GD.canvas,QtCore.SIGNAL("Wakeup"),wakeup)
    clear()
    draw(F)
    Fin.setProp(F.p)
    return array(cut_planes)

   
def savePartitions(F):
    print "Current dir is %s" % os.getcwd()
    if ack("Save the partitioned Formex?"):
        writeFormex(F,'part.fmx')
        clear()

    if ack("Reread/draw the partitioned Formex?"):
        F = readFormex('part.fmx')
        draw(F)
    

    d = splitProp(F,'part')
    export(d)

    if ack("Save the partitions separately?"):
        for (k,v) in d.iteritems():
            writeFormex(v,"%s.fmx"%k)
   

# End

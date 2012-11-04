# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
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
"""Partitioning tools

"""
from __future__ import print_function

import pyformex as pf
from gui import decors,colors
from gui.camera import inverse
from gui.draw import *
from formex import *
import geomtools

VA=None

def prepare(V):
    """Prepare the surface for slicing operation."""
    global VA
    V = V.translate(-V.center())
    P = V.center()
    print("Initial P = %s" % P)
    VA = draw(V,bbox=None,color='black')
    area,norm = geomtools.areaNormals(V.coords)
    N = norm[0]
    return V,P,N


def testview(F,V,P):
    global VA,waiting
    p = array(pf.canvas.camera.getCenter())
    waiting = True
    while waiting:
        pf.app.processEvents()
    p -= array(pf.canvas.camera.getCenter())
    print("TRANSLATE: %s" % p)
    m = pf.canvas.camera.getRot()
    P += p
    print("TOTAL TRANSLATE: %s" % P)
    V = V.affine(inverse(m[0:3,0:3])).translate(-P)
    print(V.center())
    print(F.center())
    undraw(VA)
    VA = draw(V)
    area,norm = geomtools.areaNormals(V.coords)
    N = norm[0]
    return P,N


def colorCut(F,P,N,prop):
    """Color a Formex in two by a plane (P,N)"""
    print(F.bbox())
    print(P)
    print(N)
    print(prop)
    dist = F.distanceFromPlane(P,N)
    print(dist)
    right = any(dist>0.0,axis=1)
    print(right)
    F.prop[right] = prop
    nright = right.sum()
    nleft = F.nelems() - nright
    print("Left part has %s elements, right part has %s elements" % (nleft,nright))
    return F


def splitProp(F,name):
    """Partition a Formex according to its prop values.

    Returns a dict with the partitions, named like name-prop and exports
    these named Formex instances.
    It the Formex has no props, the whole Formex is given the name.
    """
    if F.prop is None:
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
    initial_prop = F.prop
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

    QtCore.QObject.connect(pf.canvas,QtCore.SIGNAL("Wakeup"),wakeup)

    linewidth(2)
    w,h = pf.canvas.width(),pf.canvas.height()
    fgcolor('magenta')
    SD = decors.Line(w/2,0,w/2,h)
    decorate(SD)
    
    fgcolor(colors.black)

    V,P,N = prepare(V)
    while True:
        res = ask("",["Adjust Cut","Keep Cut", "Finish"])
        if res == "Adjust Cut":
            P,N = testview(F,V,P)
            print("Plane: point %s, normal %s" % (P,N))
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
        
    QtCore.QObject.disconnect(pf.canvas,QtCore.SIGNAL("Wakeup"),wakeup)
    clear()
    draw(F)
    Fin.setProp(F.prop)
    return array(cut_planes)

   
def savePartitions(F):
    print("Current dir is %s" % os.getcwd())
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

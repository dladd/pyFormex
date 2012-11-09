# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

"""tools.py

Graphic Tools for pyFormex.
"""
from __future__ import print_function

import pyformex as pf
from coords import *
from collection import Collection
from gui.actors import GeomActor
from mesh import Mesh
from formex import Formex
from plugins.trisurface import TriSurface
from plugins.nurbs import NurbsCurve,NurbsSurface

class Plane(object):

    def __init__(self,points,normal=None,size=((1.0,1.0),(1.0,1.0))):
        pts = Coords(points)
        if pts.shape == (3,) and normal is not None:
            P = pts
            n = Coords(normal)
            if n.shape != (3,):
                raise ValueError,"normal does not have correct shape"
        elif pts.shape == (3,3,):
            P = pts.centroid()
            n = cross(pts[1]-pts[0],pts[2]-pts[0])
        else:
            raise ValueError,"point(s) does not have correct shape"
        size = asarray(size)
        s = Coords([insert(size[0],0,0.,-1),insert(size[1],0,0.,-1)])
        self.P = P
        self.n = n
        self.s = s


    def point(self):
        return self.P

    def normal(self):
        return self.n
    
    def size(self):
        return self.s
    
    def bbox(self):
        return self.P.bbox()

    def __str__(self):
        return 'P:%s n:%s s:%s' % (list(self.P),list(self.n), (list(self.s[0]),list(self.s[1])))


    def actor(self,**kargs):
        from gui import actors
        actor = actors.PlaneActor(size=self.s,**kargs)
        actor = actors.RotatedActor(actor,self.n,**kargs)
        actor = actors.TranslatedActor(actor,self.P,**kargs)
        return actor


################# Report information about picked objects ################

def report(K):
    if K is not None and hasattr(K,'obj_type'):
        print(K.obj_type)
        if K.obj_type == 'actor':
            return reportActors(K)
        elif K.obj_type == 'element':
            return reportElements(K)
        elif K.obj_type == 'point':
            return reportPoints(K)
        elif K.obj_type == 'edge':
            return reportEdges(K)
        elif K.obj_type == 'partition':
            return reportPartitions(K)
    return ''


def reportActors(K):
    s = "Actor report\n"
    v = K.get(-1,[])
    s += "Actors %s\n" % v
    for k in v:
        A = pf.canvas.actors[k]
        t = A.getType()
        s += "  Actor %s (type %s)\n" % (k,t)
    return s


def reportElements(K):
    s = "Element report\n"
    for k in K.keys():
        v = K[k]
        A = pf.canvas.actors[k]
        t = A.getType()
        s += "Actor %s (type %s); Elements %s\n" % (k,t,v)
        if t == Formex:
            e = A.coords
        elif t == TriSurface or  t == Mesh :
            e = A.elems
        for p in v:
            s += "  Element %s: %s\n" % (p,e[p])
    return s


def reportPoints(K):
    s = "Point report\n"
    for k in K.keys():
        v = K[k]
        A = pf.canvas.actors[k]
        s += "Actor %s (type %s); Points %s\n" % (k,A.getType(),v)
        x = A.vertices()
        for p in v:
            s += "  Point %s: %s\n" % (p,x[p]) 
    return s


def reportEdges(K):
    s = "Edge report\n"
    for k in K.keys():
        v = K[k]
        A = pf.canvas.actors[k]
        s += "Actor %s (type %s); Edges %s\n" % (k,A.getType(),v)
        e = A.edges()
        for p in v:
            s += "  Edge %s: %s\n" % (p,e[p]) 


def reportPartitions(K):
    s = "Partition report\n"
    for k in K.keys():
        P = K[k][0]
        A = pf.canvas.actors[k]
        t = A.getType()
        for l in P.keys():
            v = P[l]
            s += "Actor %s (type %s); Partition %s; Elements %s\n" % (k,t,l,v)
            if t == 'Formex':
                e = A
            elif t == 'TriSurface':
                e = A.getElems()
            for p in v:
                s += "  Element %s: %s\n" % (p,e[p])
    return s


def reportDistances(K):
    if K is None or not hasattr(K,'obj_type') or K.obj_type != 'point':
        return ''
    s = "Distance report\n"
    x = Coords.concatenate(getCollection(K))
    s += "First point: %s %s\n" % (0,x[0])
    d = x.distanceFromPoint(x[0])
    for i,p in enumerate(zip(x,d)):
        s += "Distance from point: %s %s: %s\n" % (i,p[0],p[1])
    return s


def reportAngles(K):
    if K is None or not hasattr(K,'obj_type') or K.obj_type != 'element':
        return ''
    s = "Angle report:\n"
    for F in getCollection(K):
        if isinstance(F,Mesh):
            F=F.toFormex()
        if isinstance(F,Formex):
            x = F.coords
            if len(x)!=2:
                raise ValueError,"You didn't select 2 elements"
            v = x[:,1,:] - x[:,0,:]
            v = normalize(v)
            cosa = dotpr(v[0],v[1])
            #print(cosa)
            a = arccosd(cosa)
            s += "  a = %s" % a
        else:
            raise TypeError,"Angle measurement only possible with Formex or Mesh"
    return s

    
def getObjectItems(obj,items,mode):
    """Get the specified items from object."""
    if mode == 'actor':
        return [ obj[i].object for i in items if hasattr(obj[i],'object') ]
    elif mode in ['element','partition']:
        if hasattr(obj,'object') and hasattr(obj.object,'select'):
            return obj.object.select(items)        
    elif mode == 'point':
        if hasattr(obj,'vertices'):
            return obj.vertices()[items]
    return None


def getCollection(K):
    """Returns a collection."""
    if K.obj_type == 'actor':
        return [ pf.canvas.actors[int(i)].object for i in K.get(-1,[]) if hasattr(pf.canvas.actors[int(i)],'object') ]
    elif K.obj_type in ['element','point']:
        return [ getObjectItems(pf.canvas.actors[k],K[k],K.obj_type) for k in K.keys() ]
    elif K.obj_type == 'partition':
        return [getObjectItems(pf.canvas.actors[k],K[k][0][prop],K.obj_type) for k in K.keys() for prop in K[k][0].keys()]
    else:
        return None

   
def growCollection(K,**kargs):
    """Grow the collection with n frontal rings.

    K should be a collection of elements.
    This should work on any objects that have a growSelection method.
    """
    if K.obj_type == 'element':
        for k in K.keys():
            o = pf.canvas.actors[k]
            if hasattr(o,'growSelection'):
                K[k] = o.growSelection(K[k],**kargs)


def partitionCollection(K):
    """Partition the collection according to node adjacency.
    
    The actor numbers will be connected to a collection of property numbers,
    e.g. 0 [1 [4,12] 2 [6,20]], where 0 is the actor number, 1 and 2 are the
    property numbers and 4, 12, 6 and 20 are the element numbers.
    """
    sel = getCollection(K)
    if len(sel) == 0:
        print("Nothing to partition!")
        return
    if K.obj_type == 'actor':
        actor_numbers = K.get(-1,[])
        K.clear()
        for i in actor_numbers:
            K.add(range(sel[int(i)].nelems()),i)
    prop = 1
    j = 0
    for i in K.keys():
        p = sel[j].partitionByConnection() + prop
        print("Actor %s partitioned in %s parts" % (i,p.max()-p.min()+1))
        C = Collection()
        C.set(transpose(asarray([p,K[i]])))
        K[i] = C
        prop += p.max()-p.min()+1
        j += 1
    K.setType('partition')


def getPartition(K,prop):
    """ Remove all partitions with property not in prop."""
    for k in K.keys():
        for p in K[k][0].keys():
            if not p in prop:
                K[k][0].remove(K[k][0][p],p)


def exportObjects(obj,name,single=False):
    """Export a list of objects under the given name.

    If obj is a list, and single=True, each element of the list is exported
    as a single item. The items will be given the names name-0, name-1, etc.
    Else, the obj is exported as is under the name.
    """
    if single and type(obj) == list:
        export(dict([ ("name-%s"%i,v) for i,v in enumerate(obj)]))
    else:
        export({name:obj})

                
# End

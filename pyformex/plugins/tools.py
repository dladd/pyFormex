# $Id$

"""tools.py

Graphic Tools for pyFormex.
"""

from coords import *
from numtools import Collection
from numpy import ones

class Plane(object):

    def __init__(self,point,normal,size=(1.0,1.0)):
        P = Coords(point)
        n = Coords(normal)
        s = Coords((0.0,size[0],size[1]))

        if P.shape != (3,) or n.shape != (3,):
            raise ValueError,"point or normal does not have correct shape"

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
        return 'P:%s n:%s' % (list(self.P),tuple(self.n)) 


################# Report information about picked objects ################

def report(K):
    if K is not None and hasattr(K,'obj_type'):
        print K.obj_type
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
        A = GD.canvas.actors[k]
        t = A.atype()
        s += "  Actor %s (type %s)\n" % (k,t)
    return s


def reportElements(K):
    s = "Element report\n"
    for k in K.keys():
        v = K[k]
        A = GD.canvas.actors[k]
        t = A.atype()
        s += "Actor %s (type %s); Elements %s\n" % (k,t,v)
        if t == 'Formex':
            e = A
        elif t == 'TriSurface':
            e = A.getElems()
        for p in v:
            s += "  Element %s: %s\n" % (p,e[p])
    return s


def reportPoints(K):
    s = "Point report\n"
    for k in K.keys():
        v = K[k]
        A = GD.canvas.actors[k]
        s += "Actor %s (type %s); Points %s\n" % (k,A.atype(),v)
        x = A.vertices()
        for p in v:
            s += "  Point %s: %s\n" % (p,x[p]) 
    return s


def reportEdges(K):
    s = "Edge report\n"
    for k in K.keys():
        v = K[k]
        A = GD.canvas.actors[k]
        s += "Actor %s (type %s); Edges %s\n" % (k,A.atype(),v)
        e = A.edges()
        for p in v:
            s += "  Edge %s: %s\n" % (p,e[p]) 


def reportPartitions(K):
    s = "Partition report\n"
    for k in K.keys():
        P = K[k][0]
        A = GD.canvas.actors[k]
        t = A.atype()
        for l in P.keys():
            v = P[l]
            s += "Actor %s (type %s); Partition %s; Elements %s\n" % (k,t,l,v)
            if t == 'Formex':
                e = A
            elif t == 'Surface':
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


    
def getObjectItems(obj,items,mode):
    """Get the specified items from object."""
    if mode == 'actor':
        return [ obj[i] for i in items ]
    elif mode in ['element','partition']:
        if hasattr(obj,'select'):
            return obj.select(items)
    elif mode == 'point':
        if hasattr(obj,'vertices'):
            return obj.vertices()[items]
    return None


def getCollection(K):
    """Returns a collection."""
    if K.obj_type == 'actor':
        return [ GD.canvas.actors[i] for i in K.get(-1,[]) ]
    elif K.obj_type in ['element','point']:
        return [ getObjectItems(GD.canvas.actors[k],K[k],K.obj_type) for k in K.keys() ]
    elif K.obj_type == 'partition':
        return [[getObjectItems(GD.canvas.actors[k],K[k][0][prop],K.obj_type) for prop in K[k][0].keys()] for k in K.keys()]
    else:
        return None

   
def setpropCollection(K,prop):
    """Set the property of a collection.

    prop should be a single non-negative integer value or None.
    If None is given, the prop attribute will be removed from the objects
    in collection even the non-selected items.
    If a selected object does not have a setProp method, it is ignored.
    """
    if K.obj_type == 'actor':
        obj = [ GD.canvas.actors[i] for i in K.get(-1,[]) ]
        for o in obj:
            if hasattr(o,'setProp'):
                o.setProp(prop)
            
    elif K.obj_type in ['element','point']:
        for k in K.keys():
            o = GD.canvas.actors[k]
            if prop is None:
                o.setProp(prop)
            elif hasattr(o,'setProp'):
                if not hasattr(o,'p') or o.p is None:
                    o.setProp(0)
                o.p[K[k]] = prop
                print o.p
                o.setColor(o.p)
                o.redraw()

   
def growCollection(K,**kargs):
    """Grow the collection with n frontal rings.

    K should be a collection of elements.
    This currently only works on surfaces. Objects that do not have a
    nodeFront() generator function are 
    """
    if K.obj_type == 'element':
        for k in K.keys():
            o = GD.canvas.actors[k]
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
        print "Nothing to partition!"
        return
    if K.obj_type == 'actor':
        actor_numbers = K.get(-1,[])
        K.clear()
        for i in actor_numbers:
            K.add(range(sel[i].nelems()),i)
    prop = 1
    j = 0
    for i in K.keys():
        p = sel[j].partitionByConnection() + prop
        print "Actor %s partitioned in %s parts" % (i,p.max()-p.min()+1)
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

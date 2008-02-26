# $Id$

"""tools.py

Graphic Tools for pyFormex.
"""

from coords import *

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
        if K.obj_type == 'element':
            return reportElements(K)
        elif K.obj_type == 'point':
            return reportPoints(K)
    return ''


def reportElements(K):
    s = "Element report\n"
    for k in K.keys():
        v = K[k]
        A = GD.canvas.actors[k]
        t = A.atype()
        s += "Actor %s (type %s); Elements %s\n" % (k,t,v)
        if t == 'Formex':
            e = A
        elif t == 'Surface':
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


def getObjectItems(obj,items,mode):
    """Get the specified items from object."""
    if mode == 'actor':
        return [ obj[i] for i in items ]
    elif mode == 'element':
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

   
def growCollection(K,n=1):
    """Grow the collection with n frontal rings.

    K should be a collection of elements.
    This currently only works on surfaces. Objects that do not have a
    nodeFront() generator function are 
    """
    if K.obj_type == 'element':
        for k in K.keys():
            o = GD.canvas.actors[k]
            if hasattr(o,'nodeFront'):
                p = o.walkNodeFront(nsteps=n+1,startat=K[k])
                K[k] = where(p>=0)[0]
                #o.setProp(0)
                #o.p[K[k]] = 1

    
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

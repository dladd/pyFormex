#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Finite Element Methods.

"""

from coords import *
from mydict import Dict
import numpy

def mergeNodes(nodes):
    """Merge all the nodes of a list of node sets.

    Each item in nodes is a Coords array.
    The return value is a tuple with:
     - the coordinates of all unique nodes,
     - a list of indices translating the old node numbers to the new.
    """
    coords = Coords(numpy.concatenate([x for x in nodes],axis=0))
    coords,index = coords.fuse()
    n = numpy.array([0] + [ x.npoints() for x in nodes ]).cumsum()
    ind = [ index[f:t+1] for f,t in zip(n[:-1],n[1:]) ]
    return coords,ind


def mergeModels(femodels):
    """Merge all the nodes of a list of FE models.

    Each item in femodels is a (coords,elems) tuple.
    The return value is a tuple with:
     - the coordinates of all unique nodes,
     - the index translating old node numbers to new,
     - a list of elems corresponding to the input list,
       but with numbers referring to the new coordinates.
    """
    nodes = [ x for x,e in femodels ]
    coords = Coords(numpy.concatenate(nodes,axis=0))
    coords,index = coords.fuse()
    n = numpy.array([0] + [ x.npoints() for x in nodes ]).cumsum()
    ind = [ index[f:t+1] for f,t in zip(n[:-1],n[1:]) ]
    elems = [ e for x,e in femodels ]
    return coords,[i[e] for i,e in zip(ind,elems)]
              


######################## Finite Element Model ##########################

class Model(Dict):
    """Contains all FE model data."""
    
    def __init__(self,nodes,elems):
        """Create new model data.

        nodes is an array with nodal coordinates
        elems is either a single element connectivity array, or a list of such.
        In a simple case, nodes and elems can be the arrays obtained by 
            nodes, elems = F.feModel()
        This is however limited to a model where all elements have the same
        number of nodes. Then you can use the list of elems arrays. The 'fe'
        plugin has a helper function to create this list. E.g., if FL is a
        list of Formices (possibly with different plexitude), then
          fe.mergeModels([Fi.feModel() for Fi in FL])
        will return the (nodes,elems) tuple to create the Model.

        """
        if not type(elems) == list:
            elems = [ elems ]
        self.nodes = asarray(nodes)
        self.elems = map(asarray,elems)
        nelems = [elems.shape[0] for elems in self.elems]
        self.celems = cumsum([0]+nelems)
        GD.message("Number of nodes: %s" % self.nodes.shape[0])
        GD.message("Number of elements: %s" % self.celems[-1])
        GD.message("Number of element groups: %s" % len(nelems))
        GD.message("Number of elements per group: %s" % nelems)


    def splitElems(self,set):
        """Splits a set of element numbers over the element groups.

        Returns two lists of element sets, the first in global numbering,
        the second in group numbering.
        Each item contains the element numbers from the given set that
        belong to the corresponding group.
        """
        set = unique1d(set)
        split = []
        n = 0
        for e in self.celems[1:]:
            i = set.searchsorted(e)
            split.append(set[n:i])
            n = i

        return split,[ asarray(s) - ofs for s,ofs in zip(split,self.celems) ]
        
 
    def getElems(self,sets):
        """Return the definitions of the elements in sets.

        sets should be a list of element sets with length equal to the
        number of element groups. Each set contains element numbers local
        to that group.
        
        As the elements can be grouped according to plexitude,
        this function returns a list of element arrays matching
        the element groups in self.elems. Some of these arrays may
        be empty.

        It also provide the global and group element numbers, since they
        had to be calculated anyway.
        """
        return [ e[s] for e,s in zip(self.elems,sets) ]



if __name__ == "__main__":

    print __doc__


# End

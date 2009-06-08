# $Id$
##
##  This file is part of pyFormex 0.8 Release Mon Jun  8 11:56:55 2009
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
"""Finite Element Models in pyFormex.

Finite element models are geometrical models that consist of a unique
set of nodal coordinates and one of more sets of elements.
"""

from coords import *
from connectivity import *
from mydict import Dict
from numpy import *


def mergeNodes(nodes):
    """Merge all the nodes of a list of node sets.

    Each item in nodes is a Coords array.
    The return value is a tuple with:
     - the coordinates of all unique nodes,
     - a list of indices translating the old node numbers to the new.
    """
    coords = Coords(concatenate([x for x in nodes],axis=0))
    coords,index = coords.fuse()
    n = array([0] + [ x.npoints() for x in nodes ]).cumsum()
    ind = [ index[f:t+1] for f,t in zip(n[:-1],n[1:]) ]
    return coords,ind


def mergeModels(femodels):
    """Merge all the nodes of a list of FE models.

    Each item in femodels is a (coords,elems) tuple.
    The return value is a tuple with:
     - the coordinates of all unique nodes,
     - a list of elems corresponding to the input list,
       but with numbers referring to the new coordinates.
    """
    nodes = [ x for x,e in femodels ]
    coords = Coords(concatenate(nodes,axis=0))
    coords,index = coords.fuse()
    n = array([0] + [ x.npoints() for x in nodes ]).cumsum()
    ind = [ index[f:t+1] for f,t in zip(n[:-1],n[1:]) ]
    elems = [ e for x,e in femodels ]
    return coords,[i[e] for i,e in zip(ind,elems)]
              

def checkUniqueNumbers(nrs,nmin=0,nmax=None,error=None):
    """Check that an array contains a set of unique integers in range.

    nrs is an integer array with any shape.
    All integers should be unique and in the range(nmin,nmax).
    Beware: this means that    nmin <= i < nmax  !
    Default nmax is unlimited. Set nmin to None to
    error is the value to return if the tests are not passed.
    By default, a ValueError is raised.
    On success, None is returned
    """
    nrs = asarray(nrs)
    new = unique1d(nrs)
    if new.size != nrs.size or \
           (nmin is not None and new.min() < nmin) or \
           (nmax is not None and new.max() > nmax):
        if error is None:
            raise ValueError,"Values not unique or not in range"
        else:
            return error
    return None


######################## Finite Element Model ##########################

class Model(Dict):
    """Contains all FE model data."""
    
    def __init__(self,coords,elems):
        """Create new model data.

        coords is an array with nodal coordinates
        elems is either a single element connectivity array, or a list of such.
        In a simple case, coords and elems can be the arrays obtained by 
            coords, elems = F.feModel()
        This is however limited to a model where all elements have the same
        number of nodes. Then you can use the list of elems arrays. The 'fe'
        plugin has a helper function to create this list. E.g., if FL is a
        list of Formices (possibly with different plexitude), then
          fe.mergeModels([Fi.feModel() for Fi in FL])
        will return the (coords,elems) tuple to create the Model.

        The model can have node and element property numbers.
        """
        Dict.__init__(self)
        if not type(elems) == list:
            elems = [ elems ]
        self.coords = Coords(coords)
        self.elems = [ Connectivity(e) for e in elems ]
        nelems = [ e.nelems() for e in self.elems ]
        nplex = [ e.nplex() for e in self.elems ]
        self.celems = cumsum([0]+nelems)
        GD.message("Number of nodes: %s" % self.coords.shape[0])
        GD.message("Number of elements: %s" % self.celems[-1])
        GD.message("Number of element groups: %s" % len(nelems))
        GD.message("Number of elements per group: %s" % nelems)
        GD.message("Plexitude of each group: %s" % nplex)


    def nnodes(self):
        return self.coords.shape[0]

    def nelems(self):
        return self.celems[-1]

    def mplex(self):
        """Return the maximum plexitude of the model."""
        return max([e.nplex() for e in self.elems])


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


    def elemNrs(group,set):
        """Return the global element numbers for elements set in group"""
        return self.celems[group] + set

 
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
        
 
    def renumber(self,old=None,new=None):
        """Renumber a set of nodes.

        old and new are equally sized lists with unique node numbers, each
        smaller that the number of nodes in the model.
        The old numbers will be renumbered to the new numbers.
        If one of the lists is None, a range with the length of the
        other is used.
        If the lists are shorter than the number of nodes, the remaining
        nodes will be numbered in an unspecified order.
        If both lists are None, the nodes are renumbered randomly.

        This function returns a tuple (old,new) with the full renumbering
        vectors used. The first gives the old node numbers of the current
        numbers, the second gives the new numbers cooresponding with the
        old ones.
        """
        nnodes = self.nnodes()
        if old is None and new is None:
            old = unique1d(random.randint(0,nnodes-1,nnodes))
            new = unique1d(random.randint(0,nnodes-1,nnodes))
            nn = max(old.size,new.size)
            old = old[:nn]
            new = new[:nn]
        elif old is None:
            new = asarray(new).reshape(-1)
            checkUniqueNumbers(new,0,nnodes)
            old = arange(new.size)
        elif new is None:
            old = asarray(old).reshape(-1)
            checkUniqueNumbers(old,0,nnodes)
            new = arange(old.size)

        all = arange(nnodes)
        old = concatenate([old,setdiff1d(all,old)])
        new = concatenate([new,setdiff1d(all,new)])
        #print "old:\n",old
        #print "new:\n",new
        oldnew = old[new]
        newold = argsort(oldnew)
        #print "oldnew:\n",oldnew
        #print "newold:\n",newold
        self.coords = self.coords[oldnew]
        self.elems = [ Connectivity(newold[e]) for e in self.elems ]
        return oldnew,newold

        
def mergedModel(*args):
    """Returns the fe Model obtained from merging individual models.

    The input arguments are (coords,elems) tuples.
    The return value is a merged fe Model.
    """
    return Model(*mergeModels(args))


if __name__ == "__main__":

    print __doc__

# End

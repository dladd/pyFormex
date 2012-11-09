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
"""Finite Element Models in pyFormex.

Finite element models are geometrical models that consist of a unique
set of nodal coordinates and one of more sets of elements.
"""
from __future__ import print_function
import pyformex as pf

from coords import *
from connectivity import *
from geometry import Geometry
#from numpy import *
from mesh import Mesh,mergeMeshes
from utils import deprecation
import warnings


######################## Finite Element Model ##########################

class Model(Geometry):
    """Contains all FE model data."""
    
    _set_coords = Geometry._set_coords_inplace
    
    def __init__(self,coords=None,elems=None,meshes=None):
        """Create new model data.

        coords is an array with nodal coordinates
        elems is either a single element connectivity array, or a list of such.
        In a simple case, coords and elems can be the arrays obtained by 
        ``coords, elems = F.feModel()``.
        This is however limited to a model where all elements have the same
        number of nodes. Then you can use the list of elems arrays. The 'fe'
        plugin has a helper function to create this list. E.g., if ``FL`` is a
        list of Formices (possibly with different plexitude), then
        ``fe.mergeModels([Fi.feModel() for Fi in FL])``
        will return the (coords,elems) tuple to create the Model.

        The model can have node and element property numbers.
        """
        if meshes is not None:
            if coords is not None or elems is not None:
                raise ValueError,"You can not use nodes/elems together with meshes"
            for M in meshes:
                if M.prop is None:
                    M.setProp(0)
            self.coords,self.elems = mergeMeshes(meshes)
            self.prop = concatenate([M.prop for M in meshes])
        else:
            if not type(elems) == list:
                elems = [ elems ]
            self.coords = Coords(coords)
            self.elems = [ Connectivity(e) for e in elems ]
            self.prop = None
            meshes = self.meshes()
        nnodes = [ m.nnodes() for m in meshes ]
        nelems = [ m.nelems() for m in meshes ]
        nplex = [ m.nplex() for m in meshes ]
        self.cnodes = cumsum([0]+nnodes)
        self.celems = cumsum([0]+nelems)
        pf.message("Finite Element Model")
        pf.message("Number of nodes: %s" % self.coords.shape[0])
        pf.message("Number of elements: %s" % self.celems[-1])
        pf.message("Number of element groups: %s" % len(nelems))
        #pf.message("Number of nodes per group: %s" % nnodes)
        pf.message("Number of elements per group: %s" % nelems)
        pf.message("Plexitude of each group: %s" % nplex)


    def meshes(self):
        """Return the parts as a list of meshes"""
        return [ Mesh(self.coords,e) for e in self.elems ]

    def nnodes(self):
        """Return the number of nodes in the model."""
        return self.coords.shape[0]

    def nelems(self):
        """Return the number of elements in the model."""
        return self.celems[-1]

    def ngroups(self):
        """Return the number of element groups in the model."""
        return len(self.elems)

    def mplex(self):
        """Return the maximum plexitude of the model."""
        return max([e.nplex() for e in self.elems])


    def splitElems(self,elems):
        """Splits a set of element numbers over the element groups.

        Returns two lists of element sets, the first in global numbering,
        the second in group numbering.
        Each item contains the element numbers from the given set that
        belong to the corresponding group.
        """
        elems = unique(elems)
        split = []
        n = 0
        for e in self.celems[1:]:
            i = elems.searchsorted(e)
            split.append(elems[n:i])
            n = i

        return split,[ asarray(s) - ofs for s,ofs in zip(split,self.celems) ]


    def elemNrs(self,group,elems=None):
        """Return the global element numbers for elements set in group"""
        if elems is None:
            elems = range(len(self.elems[group]))
        return self.celems[group] + elems

 
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
            old = unique(random.randint(0,nnodes-1,nnodes))
            new = unique(random.randint(0,nnodes-1,nnodes))
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
        oldnew = old[new]
        newold = argsort(oldnew)
        self.coords = self.coords[oldnew]
        self.elems = [ Connectivity(newold[e],eltype=e.eltype) for e in self.elems ]
        return oldnew,newold


class FEModel(Geometry):
    """A Finite Element Model.

    This class is intended to collect all data concerning a Finite Element
    Model. In due time it may replace the Model class. Currently it only
    holds geometrical data, but will probably be expanded later to include
    a property database holding material data, boundary conditions, loading
    conditions and simulation step data.

    While the Model class stores the geometry  in a single coords block
    and multiple elems blocks, the new FEModel class uses a list of Meshes.
    The Meshes do not have to be compact though, and thus all Meshes in the
    FEModel could used the same coords block, resulting in an equivalent
    model as the old Model class. But the Meshes may also use different
    coords blocks, allowing to accomodate better to versatile applications.

    
    """
    
    _set_coords = Geometry._set_coords_inplace
    
    def __init__(self,meshes):
        """Create a new FEModel."""
        
        if not type(meshes) == list:
            meshes = [ meshes ]

        for m in meshes:
            if not isinstance(m,Mesh):
                raise ValueError,"Expected a Mesh or a list thereof."

        nnodes = [ m.nnodes() for m in meshes ]
        nelems = [ m.nelems() for m in meshes ]
        nplex = [ m.nplex() for m in meshes ]
        self.meshes = meshes
        self.cnodes = cumsum([0]+nnodes)
        self.celems = cumsum([0]+nelems)
        pf.message("Finite Element Model")
        pf.message("Number of nodes: %s" % self.coords.shape[0])
        pf.message("Number of elements: %s" % self.celems[-1])
        pf.message("Number of element groups: %s" % len(nelems))
        #pf.message("Number of nodes per group: %s" % nnodes)
        pf.message("Number of elements per group: %s" % nelems)
        pf.message("Plexitude of each group: %s" % nplex)

        
def mergedModel(meshes,**kargs):
    """Returns the fe Model obtained from merging individual meshes.

    The input arguments are (coords,elems) tuples.
    The return value is a merged fe Model.
    """
    return Model(*mergeMeshes(meshes,**kargs))


if __name__ == "__main__":

    print(__doc__)

# End

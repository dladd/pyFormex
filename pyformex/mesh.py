# $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

"""Mesh representation of geometrical data.

A Mesh representation describes geometrical entities by a tuple of
(coords,elems), where coords is an array with coordinates of points
and elems is an array describing the geometrical entities by a list of
point numbers.

On large data sets, this model can be more efficient then the Formex model,
both in memory and cpu usage, but it does not allow the simple geometry
creation functions of the Formex.
"""

from coords import *


######################## Mesh geometry model ##########################

class Mesh(Dict):
    """A class representing (discrete) geometrical entities.

    The Mesh representation describes geometrical entities by a tuple of
    (coords,elems), where coords is an array with coordinates of points
    and elems is an array describing the geometrical entities by a list of
    point numbers.
    
    coords is a (npoints,3) array of float values, each line of which
    represents a single point in space, addresses by its row index.
    
    elems is a (nelems,nplex) array of integer values, such that each row
    describes a single geometrical entity, defined by the nplex points
    defined by rows of coords with indices equal to to values in elems.

    By default all geometrical entities are of the same plexitude.
    It is possible to relax this requirement and allow the inclusion of
    elements with a lower plexitude by setting some vertex indices to a
    negative value.

    Besides the geometrical data, the object can also hold an array of
    node and element property numbers. These could e.g. be used to give
    the nodes/elements another numbering scheme than the internal one.
    """

    
    def __init__(self,coords,elems,nprop=None,eprop=None,strict=True):
        """Create new mesh geometry from given coords and elems arrays.

        This will check that coords and elems arrays have proper shape
        and data type, and that the numbers in elems are within the row
        range of the coords array.

        The model can have node and element property numbers.
        """
        self.coords = checkArray(coords,shape=(-1,3),kind='f')
        self.elems = checkArray(elems,shape=(-1,-1),kind='i')


    def nnodes(self):
        return self.nodes.shape[0]

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
        print "old:\n",old
        print "new:\n",new
        oldnew = old[new]
        newold = argsort(oldnew)
        print "oldnew:\n",oldnew
        print "newold:\n",newold
        self.nodes = self.nodes[oldnew]
        self.elems = [ newold[e] for e in self.elems ]
        return oldnew,newold

        
def mergedModel(*args):
    """Returns the fe Model obtained from merging individual models.

    The input arguments are (nodes,elems) tuples.
    The return value is a merged fe Model.
    """
    return Model(*mergeModels(args))



# End

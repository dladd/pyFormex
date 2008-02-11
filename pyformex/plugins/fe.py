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

import numpy

def mergeNodes(nodes):
    """Merge all the nodes of a list of node sets.

    Each item in nodes is a Coords array.
    The return value is a tuple with:
     - the coordinates of all unique nodes,
     - a list of indices translating the old node numbers to the new.
    """
    coords = numpy.concatenate([x for x in nodes],axis=0)
    coords,index = coords.fuse()
    n = numpy.array([0] + [ x.npoints for x in nodes ])
    print n
    print n.cumsum()
    return coords,index


def mergeModels(femodels):
    """Merge all the nodes of a list of FE models.

    Each item in femodels is a (coords,elems) tuple.
    The return value is a tuple with:
     - the coordinates of all unique nodes,
     - the index translating old node numbers to new,
     - a list of elems corresponding to the input list,
       but with numbers referring to the new coordinates.
    """
    coords = numpy.concatenate([x for x,e in femodels],axis=0)
    coords,index = coords.fuse()
    return coords,index,[index[e] for x,e in femodels]
              



if __name__ == "__main__":

    print __doc__


# End

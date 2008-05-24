# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
#
"""Python equivalents of the functions in lib.misc"""

import globaldata as GD
from numpy import *

# Default is to try using the compiled library
if GD.options.uselib is None:
    GD.options.uselib = True

# Try to load the library
success = False
if GD.options.uselib:
    try:
        from lib.misc import *
        GD.debug("Succesfully loaded the pyFormex compiled library")
        success = True
    except ImportError:
        GD.debug("Error while loading the pyFormex compiled library")
        GD.debug("Reverting to scripted versions")

if not success:
    GD.debug("Using the (slower) Python implementations")
    print "OVERRIDING WITH SLOW IMPLEMENTATION"

    def nodalSum(val,elems,nodes,work,avg):
        """Compute the nodal sum of values defined on elements.

        val   : (nelems,nplex,nval) values at points of elements.
        elems : (nelems,nplex) nodal ids of points of elements.
        work  : a work space (unused) 
        nodes : (nnod) unique nodal ids in elems.

        The return value is a (nelems,nplex,nval) array where each value is
        replaced with the sum of its values at that node.
        If avg=True, the values are replaced with the average instead.

        The summation is done inplace!
        """
        print "NODAL SUM"
        for i in nodes:
            wi = where(elems==i)
            vi = val[wi]
            if avg:
                vi = vi.sum(axis=0)/vi.shape[0]
            else:
                vi = vi.sum(axis=0)
            val[wi] = vi

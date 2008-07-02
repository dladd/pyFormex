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

import pyformex as GD
from formex import *
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

    def nodalSum(val,elems,work,avg):
        """Compute the nodal sum of values defined on elements.

        val   : (nelems,nplex,nval) values at points of elements.
        elems : (nelems,nplex) nodal ids of points of elements.
        work  : a work space (unused) 

        On return each value is replaced with the sum of values at that node.
        If avg=True, the values are replaced with the average instead.

        The summation is done inplace, so there is no return value!
        """
        nodes = unique1d(elems)
        for i in nodes:
            wi = where(elems==i)
            vi = val[wi]
            if avg:
                vi = vi.sum(axis=0)/vi.shape[0]
            else:
                vi = vi.sum(axis=0)
            val[wi] = vi



    def average_close(a,mode='direction',tol=0.001):
        """Average values from an array according to some specification.

        The default is to have a direction that is nearly the same.
        a is a 2-dim array
        """
        if a.ndim != 2:
            raise ValueError,"array should be 2-dimensional!"
        print a
        tol = 1.0 - tol
        n = normalize(a)
        print n
        nrow = a.shape[0]
        cnt = zeros(nrow,dtype=int32)
        print cnt
        while cnt.min() == 0:
            w = where(cnt==0)
            nw = n[w]
            #print "w=%s" % w
            #print "nw=%s"%nw
            d = dotpr(nw[0],nw)
            #print "dotpr=%s" % d
            #print tol
            wok = where(d >= tol)
            #print "wok=%s"%wok
            wi = w[0][wok[0]]
            #print "wi=%s"%wi
            nwi = len(wi)
            #print nwi,wi
            cnt[wi] = nwi
            a[wi] = a[wi].sum(axis=0)
            #print cnt
            #print a
            #print "DONE"
            #return
        print a
        print cnt

    def nodalSum2(val,elems):
        """Compute the nodal sum of values defined on elements.

        val   : (nelems,nplex,nval) values at points of elements.
        elems : (nelems,nplex) nodal ids of points of elements.
        work  : a work space (unused) 

        The return value is a tuple of two arrays:
        res:
        cnt
        On return each value is replaced with the sum of values at that node.
        """
        res = zeros(val.shape,dtype=val.dtype)
        cnt = zeros(val.shape,dtype=int32)
        nodes = unique1d(elems)
        for i in nodes:
            wi = where(elems==i)
            print wi
            vi = val[wi]
            print vi.shape
            average_close(vi)
            ri = res[wi]
            ni = cnt[wi]
            vi = vi.sum(axis=0)
            val[wi] = vi
        return res,cnt

# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
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
"""Operations on triangulated surfaces using GTS functions.

This module provides access to GTS from insisde pyFormex.
"""
from __future__ import print_function

import pyformex as pf
from coords import *
import utils
import os
from plugins.trisurface import TriSurface,read_gts_intersectioncurve

utils.hasExternal('gts')
utils.hasModule('pygts')
#
# gts commands used:
#   in Debian package: stl2gts gts2stl gtscheck
#   not in Debian package: gtssplit gtscoarsen gtsrefine gtssmooth
#

def boolean(self,surf,op,check=False,verbose=False):
    """Perform a boolean operation with another surface.

    Boolean operations between surfaces are a basic operation in
    free surface modeling. Both surfaces should be closed orientable
    non-intersecting manifolds.
    Use the :meth:`check` method to find out.

    The boolean operations are set operations on the enclosed volumes:
    union('+'), difference('-') or intersection('*').

    Parameters:

    - `surf`: a closed manifold surface
    - `op`: boolean operation: one of '+', '-' or '*'.
    - `check`: boolean: check that the surfaces are not self-intersecting;
      if one of them is, the set of self-intersecting faces is written
      (as a GtsSurface) on standard output
    - `verbose`: boolean: print statistics about the surface

    Returns: a closed manifold TriSurface
    """
    return self.gtsset(surf,op,filt = '| gts2stl',ext='.stl',check=check,verbose=verbose)


def intersection(self,surf,check=False,verbose=False):
    """Return the intersection curve of two surfaces.

    Boolean operations between surfaces are a basic operation in
    free surface modeling. Both surfaces should be closed orientable
    non-intersecting manifolds.
    Use the :meth:`check` method to find out.

    Parameters:

    - `surf`: a closed manifold surface
    - `check`: boolean: check that the surfaces are not self-intersecting;
      if one of them is, the set of self-intersecting faces is written
      (as a GtsSurface) on standard output
    - `verbose`: boolean: print statistics about the surface

    Returns: a list of intersection curves.
    """
    return self.gtsset(surf,op='*',ext='.list',curve=True,check=check,verbose=verbose)


def gtsset(self,surf,op,filt='',ext='.tmp',curve=False,check=False,verbose=False):
    """_Perform the boolean/intersection methods.

    See the boolean/intersection methods for more info.
    Parameters not explained there:

    - filt: a filter command to be executed on the gtsset output
    - ext: extension of the result file
    - curve: if True, an intersection curve is computed, else the surface.

    Returns the name of the (temporary) results file.
    """
    op = {'+':'union', '-':'diff', '*':'inter'}[op]
    options = ''
    if curve:
        options += '-i'
    if check:
        options += ' -s'
    if not verbose:
        options += ' -v'
    tmp = utils.tempFile(suffix='.gts').name
    tmp1 = utils.tempFile(suffix='.gts').name
    tmp2 = utils.tempFile(suffix=ext).name
    pf.message("Writing temp file %s" % tmp)
    self.write(tmp,'gts')
    pf.message("Writing temp file %s" % tmp1)
    surf.write(tmp1,'gts')
    pf.message("Performing boolean operation")
    cmd = "gtsset %s %s %s %s %s > %s" % (options,op,tmp,tmp1,filt,tmp2)
    sta,out = utils.runCommand(cmd)
    os.remove(tmp)
    os.remove(tmp1)
    if sta or verbose:
        pf.message(out)
    pf.message("Reading result from %s" % tmp2)
    if curve:
        res = read_gts_intersectioncurve(tmp2)
    else:
        res = TriSurface.read(tmp2)        
    os.remove(tmp2)
    return res


def gtsinside(self,pts,dir=0):
    """_Test whether points are inside the surface.

    pts is a plex-1 Formex.
    dir is the shooting direction.

    Returns a list of point numbers that are inside.
    This is not intended to be used directly. Use inside instead
    """
    import os
    print(os.environ)
    S = self.rollAxes(dir)
    P = pts.rollAxes(dir)
    tmp = utils.tempFile(suffix='.gts').name
    tmp1 = utils.tempFile(suffix='.dta').name
    tmp2 = utils.tempFile(suffix='.out').name
    pf.message("Writing temp file %s" % tmp)
    S.write(tmp,'gts')
    pf.message("Writing temp file %s" % tmp1)
    f = open(tmp1,'w')
    P.coords.tofile(f,sep=' ')
    f.write('\n')
    f.close()
    pf.message("Performing inside testing")
    cmd = "gtsinside %s %s > %s" % (tmp,tmp1,tmp2)
    sta,out = utils.runCommand(cmd)
    os.remove(tmp)
    os.remove(tmp1)
    if sta:
        pf.message("An error occurred during the testing.\nSee file %s for more details." % tmp2)
        return None
    pf.message("Reading results from %s" % tmp2)
    ind = fromfile(tmp2,sep=' ',dtype=Int)
    return ind


def inside(self,pts):
    """Test which of the points pts are inside the surface.

    Parameters:

    - `pts`: a (usually 1-plex) Formex or a data structure that can be used
      to initialize a Formex.

    Returns an integer array with the indices of the points that are
    inside the surface. The indices refer to the onedimensional list
    of points as obtained from pts.points().
    """
    from formex import Formex
    if not isinstance(pts,Formex):
        pts = Formex(pts)
    pts = Formex(pts).points()

    # determine bbox of common space of surface and points
    bb = bboxIntersection(self,pts)
    if (bb[0] > bb[1]).any():
        # No bbox intersection: no points inside
        return array([],dtype=Int)

    # Limit the points to the common part
    # Add point numbers as property, to allow return of original numbers
    pts.setProp(arange(pts.nelems())) 
    pts = pts.clip(testBbox(pts,bb))

    # Apply the gtsinside shooting algorithm in three directions
    ins = zeros((pts.nelems(),3),dtype=bool)
    for i in range(3):
        dirs = roll(arange(3),-i)[1:]
        # clip the surface perpendicular to the shooting direction
        S = self.clip(testBbox(self,bb,dirs))
        # find inside points shooting in direction i
        ok = gtsinside(S,pts,dir=i)
        ins[ok,i] = True

    ok = where(ins.sum(axis=-1) > 1)[0]
    return pts.prop[ok]



def install_more_trisurface_methods():

    ###################################################
    ## install these functions as TriSurface methods ## 

    from trisurface import TriSurface
    TriSurface.boolean = boolean
    TriSurface.intersection = intersection
    TriSurface.gtsset = gtsset
    TriSurface.inside = inside


## try:
##     import gts
##     print "You have pygts installed: pyFormex will try to use it instead of gts commands"
## except:
##     print "Oops, looks like something went wrong in pyformex_gts"


# End

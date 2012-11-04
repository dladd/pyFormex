# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Inside

This example shows how to find out if points are inside a closed surface.

"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['surface']
_techniques = ['inside']

from gui.draw import *
import simple
import timer

filename = os.path.join(getcfg('datadir'),'horse.off')


def selectSurfaceFile(fn):
    fn = askFilename(fn,filter=utils.fileDescription('surface'))
    return fn


def getData():
    """Ask input data from the user."""
    res = askItems(
        [ _G('Surface', [
            _I('surface','file',choices=['file','sphere']),
            _I('filename',filename,text='Image file',itemtype='button',func=selectSurfaceFile),
            _I('grade',8),
            _I('refine',0),
            ]),
          _G('Points', [
              _I('points','grid',choices=['grid','random']),
              _I('npts',[30,30,30],itemtype='ivector'),
              _I('scale',[1.,1.,1.],itemptype='point'),
              _I('trl',[0.,0.,0.],itemptype='point'),
              ]),
          ],
        enablers = [
            ( 'surface','file','filename', ),
            ( 'surface','sphere','grade', ),
            ],
        )
    if res:
        globals().update(res)

    return res


def create():
    """Create a closed surface and a set of points."""
    nx,ny,nz = npts

    # Create surface
    if surface == 'file':
        S = TriSurface.read(filename).centered()
    elif surface == 'sphere':
        S = simple.sphere(ndiv=grade)

    if refine > S.nedges():
        S = S.refine(refine)
    
    draw(S, color='red')

    if not S.isClosedManifold():
        warning("This is not a closed manifold surface. Try another.")
        return None,None
    
    # Create points

    if points == 'grid':
        P = simple.regularGrid([-1.,-1.,-1.],[1., 1., 1.],[nx-1,ny-1,nz-1])
    else:
        P = random.rand(nx*ny*nz*3)

    sc = array(scale)
    siz = array(S.sizes())
    tr = array(trl)
    P = Formex(P.reshape(-1, 3)).resized(sc*siz).centered().translate(tr*siz)
    draw(P, marksize=1, color='black')
    zoomAll()

    return S,P

    
def testInside(S,P):
    """Test which of the points P are inside surface S"""

    print("Testing %s points against %s faces" % (P.nelems(),S.nelems()))

    bb = bboxIntersection(S,P)
    drawBbox(bb,color=array(red),linewidth=2)

    t = timer.Timer()
    ind = S.inside(P)
    print("gtsinside: %s points / %s faces: found %s inside points in %s seconds" % (P.nelems(),S.nelems(),len(ind),t.seconds()))

    if len(ind) > 0:
        draw(P[ind],color=green,marksize=3,ontop=True,nolight=True,bbox='last')


def run():
    resetAll()
    clear()
    smooth()

    chdir(__file__)

    if getData():
        S,P = create()
        if S:
            testInside(S,P)
    

if __name__ == 'draw':
    run()
# End



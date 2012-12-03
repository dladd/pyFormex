# $Id$ *** pyformex ***
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

"""Voxelize

This example illustrates the use of the gtsinside program to create a
voxelization of a closed surface.
"""
from __future__ import print_function
_status = 'checked'
_level = 'advanced'
_topics = ['surface']
_techniques = ['voxelize','image']

from gui.draw import *
from plugins.imagearray import *
from plugins.trisurface import TriSurface
import simple


def showGreyImage(a):
    F = Formex('4:0123').rep([a.shape[1],a.shape[0]],[0,1],[1.,1.]).setProp(a)
    return draw(F)

def saveBinaryImage(a,f):
    c = flipud(a)
    c = dstack([c,c,c])
    im = numpy2qimage(c)
    im.save(f)


def run():
    reset()
    smooth()
    lights(True)

    S = TriSurface.read(getcfg('datadir')+'/horse.off')
    SA = draw(S)

    bb = S.bbox()
    bb1 = [ 1.1*bb[0]-0.1*bb[1], 1.1*bb[1]-0.1*bb[0]]
    print(bb)
    print(bb1)

    res = askItems([
        _I('Resolution',100),
        ])
    if not res:
        return
    
    nmax = res['Resolution']
    sz = bb1[1]-bb1[0]
    step = sz.max() / (nmax-1)
    n = (sz / step).astype(Int)
    print(n)
    P = Formex(simple.regularGrid(bb1[0],bb1[0]+n*step,n).reshape(-1,3))
    draw(P, marksize=1, color='black')
    #drawNumbers(P)
    zoomAll()
    ind = S.inside(P)
    vox = zeros(n+1,dtype=uint8)
    print(vox.shape)
    vox1 = vox.reshape(-1)
    print(vox1.shape,ind.max())
    vox1[ind] = 1
    print(vox.max())
    P.setProp(vox1)
    draw(P, marksize=8)

    dirname = askDirname()
    chdir(dirname)
    # Create output file
    if not checkWorkdir():
        print("Could not open a directory for writing. I have to stop here")
        return
    
    fs = utils.NameSequence('horse','.png')
    clear()
    flat()
    A = None
    for frame in vox:
        B = showGreyImage(frame)
        saveBinaryImage(frame*255,fs.next())
        undraw(A)
        A = B

# The following is to make it work as a script
if __name__ == 'draw':
    run()


# End

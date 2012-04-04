# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
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
"""Boolean

Perform boolean operations on surfaces
"""
_status = 'checked'
_level = 'normal'
_topics = ['surface']
_techniques = ['boolean','intersection']

from gui.draw import *
from simple import cylinder
from connectivity import connectedLineElems
from plugins.trisurface import TriSurface,fillBorder



def run():
    global F,G
    clear()
    smooth()
    view('iso')
    F = cylinder(L=8.,D=2.,nt=36,nl=20,diag='u').centered()
    F = TriSurface(F).setProp(3).close(method='planar').fixNormals()
    G = F.rotate(90.,0).trl(0,1.).setProp(1)
    export({'F':F,'G':G})
    draw([F,G])

    res = askItems(
        [ _I('op',text='Operation',choices=[
            '+ (Union)',
            '- (Difference)',
            '* Intersection',
            'Intersection Curve',
            ],itemtype='vradio'),
          _I('verbose',False,text='Show stats'),
        ])
    
    if not res:
        return
    op = res['op'][0]
    verbose = res['verbose']
    if op in '+-*':
        I = F.boolean(G,op,verbose=verbose)
    else:
        I = F.intersection(G,verbose=verbose)
    clear()
    draw(I)

    if op in '+-*':
        return

    else:
        if ack('Create a surface inside the curve ?'):
            I = I.toMesh()
            e = connectedLineElems(I.elems)
            I = Mesh(I.coords,connectedLineElems(I.elems)[0])
            clear()
            draw(I,color=red,linewidth=3)
            S = fillBorder(I,method='planar')
            draw(S)


if __name__ == 'draw':
    run()
# End

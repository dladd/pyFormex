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
"""Mobius Ring

"""
from __future__ import print_function
_status = 'checked'
_level = 'advanced'
_topics = ['geometry','surface']
_techniques = ['dialog', 'animation', 'color']

from gui.draw import *

def run():
    reset()
    smoothwire()

    res = askItems([
        ('w',2,{'text':'width'}),
        ('l',30,{'text':'length'}),
        ('n',1,{'text':'number of turns'}),
        ])
    if not res:
        return

    globals().update(res)

    C = Formex('l:1234')
    cell = connect([C,C,C,C],bias=[0,1,2,3])
    strip = cell.replic2(l,w,1.,1.).translate(1,-0.5*w)
    TA = draw(strip,color='orange',bkcolor='red')

    sleep(1)

    nsteps = 40
    step = n*180./nsteps/l
    for i in arange(nsteps+1):
        a = i*step
        torded = strip.map(lambda x,y,z: [x,y*cosd(x*a),y*sind(x*a)])
        TB = draw(torded,color='orange',bkcolor='red')
        undraw(TA)
        TA = TB

    sleep(1)
    #TA = None
    nsteps = 60
    step = 360./nsteps
    for i in arange(1,nsteps+1):
        ring = torded.trl(2,l*nsteps/pi/i).scale([i*step/l,1.,1.]).trl(0,-90).cylindrical(dir=[2,0,1])
        TB = draw(ring,color='orange',bkcolor='red')
        undraw(TA)
        TA = TB

    sleep(1)
    nsteps = 80
    step = 720./nsteps
    for i in arange(1,nsteps+1):
        mobius = ring.rotate(i*step,1)
        TB = draw(mobius,color='orange',bkcolor='red',bbox='last')
        undraw(TA)
        TA = TB


if __name__ == 'draw':
    run()
# End

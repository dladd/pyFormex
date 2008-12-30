#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Tue Dec 30 20:45:35 2008
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""Horse

level = 'normal'
topics = ['surface']
techniques = ['animation','colors'] 

This script reads horse.formex, transforms it into a surface,
loads the surface plugin and cuts the horse in a number of surfaces.
"""

from plugins.surface import TriSurface

reset()
wireframe()
chdir(GD.cfg['curfile'])

x = 20
y = GD.canvas.height()-20

def say(text):
    global y
    drawtext(text,x,y)
    y -=20

GD.message('Click Step to continue')

say('A Horse Story...')
y -= 10
F = Formex.read(GD.cfg['pyformexdir']+'/examples/horse.formex')
A = draw(F)
pause()

say('It\'s rather sad, but')
smooth()
pause()


say('the horse was badly cut;')
T = F.cutAtPlane([0.,0.,0.],[-1.,0.,0.])
undraw(A)
A = draw(T)
pause()


say('to keep it stable,')
undraw(A)
A = draw(T.rotate(-80))
pause()


say('the doctors were able')
undraw(A)
A = draw(T)
pause()


say('to add a mirrored clone:')
T += T.reflect(0)
undraw(A)
A = draw(T)
pause()

say('A method as yet unknown!')
colors = 0.1 * random.random((10,3))
for color in colors:
    B = draw(T,color=color)
    undraw(A)
    A = B
    sleep(0.5)

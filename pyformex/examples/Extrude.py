#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

"""Extrude

level = 'beginner'
topics = ['formex']
techniques = ['extrude']

"""
clear()
smooth()
M = Formex(mpattern('123')).replic2(60,30).toMesh()#.setProp(5)
draw(M,color=yellow)
#drawNumbers(M)
exit()

smoothwire()
view('iso')
delay(1)

a = Formex([0.,0.,0.])
draw(a,color='black')


b = a.extrude(8,1.,1)
draw(b,color='red')


c = b.extrude(8,1.,0)
draw(c,color='blue')


d = c.extrude(7,-1.,2)
draw(d,color='yellow')

# End

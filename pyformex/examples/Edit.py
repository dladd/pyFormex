#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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
"""Edit Global Variables

level = 'advanced'
topics = ['editing']
techniques = ['persistence','interactive']
"""
clear()
alfabet = {
    'A': '22144/61'
    }

 
def plotChar(char):
    try:
        draw(Formex(pattern(alfabet[char])))
    except:
        raise ValueError,"Can not plot character %s" % char


plotChar('A')

data = dict(
    a = 1,
    b = 1.5,
    c = [0.,0.,0.],
    d = 'red',
    f = Formex(pattern('121212'))
    )
export(data)
globals().update(data)
draw(f)
               

# End

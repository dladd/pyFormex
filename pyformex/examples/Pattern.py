#!/usr/bin/env pyformex --gui
# $Id: Pattern.py 85 2006-04-02 12:36:40Z bverheg $
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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
"""Pattern

level = 'beginner'
topics = ['geometry']
techniques = ['colors']

"""

import simple


if __name__ == "draw":

    reset()
    setDrawOptions(dict(view='front',linewidth=5,fgcolor='red'))
    grid = actors.GridActor(nx=(4,4,0),ox=(-2.0,-2.0,0.0),dx=(1.0,1.0,1.0),planes=False,linewidth=1)
    drawActor(grid)
    linewidth(3)
    FA = None
    setDrawOptions({'bbox':None})
    for n,p in simple.Pattern.items():
        message("%s = %s" % (n,p))
        FB = draw(Formex(pattern(p)),bbox=None,color='red')
        if FA:
            undraw(FA)
        FA = FB

# End

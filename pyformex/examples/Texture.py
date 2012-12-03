# $Id$  *** pyformex ***
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

"""Texture

Shows how to draw with textures and how to set a background image.
"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['Image','Geometry']
_techniques = ['texture']

from gui.draw import *
from plugins.imagearray import image2numpy

def run():
    clear()
    smooth()

    imagefile = os.path.join(pf.cfg['pyformexdir'],'data','butterfly.png')
    image = image2numpy(imagefile,indexed=False)

    import simple
    F = simple.cuboid().centered()
    G = Formex('4:0123').replic2(3,2).toMesh().setProp(range(1,7)).centered()
    draw([F,G],texture=image)
    view('iso')
    zoom(0.5)

    from gui.decors import Rectangle
    R = Rectangle(100,100,400,300,color=yellow,texture=image)
    decorate(R)

    bgcolor(color=white,image=imagefile)

if __name__ == 'draw':
    run()
# End

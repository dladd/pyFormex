# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 09:32:38 2009
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

from numpy import *
from gui.draw import *

p = 0
_name_ = "_dummy_"


def name(s):
    global _name_
    _name_ = str(s)


def position(*args):
    pass

    
def IndexedFaceSet(coords,faces=None):
    global p
    p += 1
    coords = asarray(coords).reshape(-1,3)
    print coords.shape,p
    F = Formex(coords,p)
    print F.p
    draw(F)
    export({"%s-%s" % (_name_,'coords'):F})
    if faces is None:
        return

    
def IndexedLineSet(coords,lines):
    coords = asarray(coords).reshape(-1,3)
    print coords.shape
    F = Formex(coords,p)
    draw(F)
    export({"%s-%s" % (_name_,'coords'):F})
    lines = column_stack([lines[:-1],lines[1:]])
    print lines.shape
    G = Formex(coords[lines],p)
    export({_name_:G})
    draw(G)
   




# End

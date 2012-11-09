# $Id$
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
"""Wrl

"""
from __future__ import print_function

from numpy import *
from gui.draw import *

_prop_ = 0
_name_ = "_dummy_"


def name(s):
    global _name_
    _name_ = str(s)


def position(*args):
    pass

    
def IndexedFaceSet(coords,faces=None):
    global _prop_
    _prop_ += 1
    coords = asarray(coords).reshape(-1,3)
    print(coords.shape,_prop_)
    F = Formex(coords,_prop_)
    print(F.prop)
    draw(F)
    export({"%s-%s" % (_name_,'coords'):F})
    if faces is None:
        return

    
def IndexedLineSet(coords,lines):
    coords = asarray(coords).reshape(-1,3)
    print(coords.shape)
    F = Formex(coords,_prop_)
    draw(F)
    export({"%s-%s" % (_name_,'coords'):F})
    lines = column_stack([lines[:-1],lines[1:]])
    print(lines.shape)
    G = Formex(coords[lines],_prop_)
    export({_name_:G})
    draw(G)
   




# End

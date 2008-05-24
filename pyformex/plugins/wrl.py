# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
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

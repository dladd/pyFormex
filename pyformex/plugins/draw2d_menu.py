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

"""2D drawing menu

This pyFormex plugin menu provides some interactive 2D drawing functions.
While the drawing operations themselves are in 2D, they can be performed
on a plane with any orientation in space. The constructed geometry always
has 3D coordinates in the global cartesian coordinate system.
"""


from draw2d import *
from geometry_menu import *
    
################################## Menu #############################

_menu = 'Draw'

def create_menu(before='help'):
    """Create the menu."""
    MenuData = [
        ("&Select drawable",drawable.ask),
        ("&Set grid",create_grid),
        ("&Remove grid",remove_grid),
        ("---",None),
        ("&Toggle Preview",toggle_preview,{'checkable':True}),
        ("---",None),
        ("&Draw Points",draw_points),
        ("&Draw Polyline",draw_polyline),
        ("&Draw Curve",draw_curve),
        ("&Draw Nurbs",draw_nurbs),
        ("&Draw Circle",draw_circle),
        ("---",None),
        ("&Split Curve",split_curve),
        ("---",None),
        ("&Reload Menu",reload_menu),
        ("&Close Menu",close_menu),
        ("Test menu",test_menu),
        ]
    w = menu.Menu(_menu,items=MenuData,parent=pf.GUI.menu,before=before)
    return w

def show_menu(before='help'):
    """Show the menu."""
    if not pf.GUI.menu.action(_menu):
        create_menu(before=before)

def close_menu():
    """Close the menu."""
    pf.GUI.menu.removeItem(_menu)


def reload_menu():
    """Reload the menu."""
    before = pf.GUI.menu.nextitem(_menu)
    print "Menu %s was before %s" % (_menu,before)
    close_menu()
    import plugins
    plugins.refresh('draw2d')
    show_menu(before=before)
    setDrawOptions({'bbox':'last'})
    print pf.GUI.menu.actionList()

def test_menu():
    print "TEST2"
    
####################################################################

if __name__ == "draw":
    # If executed as a pyformex script
    reload_menu()

# End

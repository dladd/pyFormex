#!/home/bene/prj/pyformex/pyformex/pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
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
"""Super Shape

level = 'advanced'
topic = ['geometry']
techniques = ['dialog','persistence']
"""


from simple import rectangle
from utils import NameSequence
from gui.widgets import *
from gui.draw import *
from gui.imageColor import *

dialog = None
savefile = None
tol = 1.e-4

gname = NameSequence('Grid-0')
sname = NameSequence('Shape-0')


def reset_data(initialize=False):
    """Reset the data to defaults"""
    grid_data = dict(
        grid_size = [24,12],
        x_range = (-180.,180.),
        y_range = (-90.,90.),
        grid_base = 'quad',
        grid_bias = 0.0,
        grid_skewness = 0.0,
        x_clip = (-360.,360.),
        y_clip = (-90.,90.),
        grid_name = gname.peek(),
        grid_color = 'blue',
        )
    shape_data = dict(
        north_south = 1.0,
        east_west = 1.0,
        eggness = 0.0,
        scale = [1.,1.,1.],
        post = '',
        name = sname.peek(),
        color = 'red',
        )
    GD.PF['__SuperShape__grid_data'] = grid_data
    GD.PF['__SuperShape__shape_data'] = shape_data
    globals().update(grid_data)
    globals().update(shape_data)
    if dialog:
        dialog.updateData(grid_data)
        dialog.updateData(shape_data)


def refresh(tgt,src):
    """Refresh tgt dict with values from src dict"""
    tgt.update([ (k,src[k]) for k in tgt if k in src ])


def createGrid():
    """Create the grid from global parameters"""
    global B
    nx,ny = grid_size
    b,h = x_range[1]-x_range[0], y_range[1]-y_range[0]
    if grid_base.startswith('tri'):
        diag = grid_base[-1]
    else:
        diag = ''
    B = rectangle(nx,ny,b,h,diag=diag,bias=grid_bias).translate([x_range[0],y_range[0],1.])
    if grid_skewness != 0.0:
        B = B.shear(0,1,grid_skewness*b*ny/(h*nx))
    if x_clip:
        B = B.clip(B.test('any',dir=0,min=x_clip[0]+tol*b,max=x_clip[1]-tol*b))
    if y_clip:
        B = B.clip(B.test('any',dir=1,min=y_clip[0]+tol*h,max=y_clip[1]-tol*h))
    export({grid_name:B})
    

def createSuperShape():
    """Create a super shape from global parameters"""
    global F
    B = GD.PF[grid_name]
    F = B.superSpherical(n=north_south,e=east_west,k=eggness)
    if scale == [1.0,1.0,1.0]:
        print "No need to scale"
    else:
        print "Scaling"
    F = F.scale(scale)
    if post:
        print "Post transformation"
        F = eval(post)
    export({name:F})


def showGrid():
    """Show the last created grid"""
    clear()
    wireframe()
    view('front')
    draw(B,color=grid_color)
    

def showSuperShape():
    """Show the last created super shape"""
    global color
    clear()
    smoothwire()
    print color
    if type(color) == str and color.startswith('file:'):
        print "trying to convert color"
        im = QtGui.QImage('butterfly.ppm')
        nx,ny = grid_size
        color=image2glcolor(im.scaled(nx,ny))

    draw(F,color=color)


# Button Functions
def show_grid():
    dialog.acceptData()
    refresh(GD.PF['__SuperShape__grid_data'],dialog.results)
    refresh(GD.PF['__SuperShape__shape_data'],dialog.results)
    globals().update(dialog.results)
    createGrid()
    showGrid()

def show_shape():
    dialog.acceptData()
    globals().update(dialog.results)
    createGrid()
    createSuperShape()
    showSuperShape()

def create_and_show(data={}):
    # Currently, this does not update the input dialog
    if (data):
        globals().update(data)
    createGrid()
    createSuperShape()
    showSuperShape()

def close():
    global dialog,savefile
    if dialog:
        dialog.close()
        dialog = None
    if savefile:
        savefile.close()
        savefile = None


def reset():
    reset_data()


def save():
    global savefile
    show_shape()
    if savefile is None:
        filename = askFilename(filter="Text files (*.txt)")
        if filename:
            savefile = file(filename,'a')
    if savefile:
        print "Saving to file"
        savefile.write('%s\n' % str(dialog.results))
        savefile.flush()
        globals().update({'grid_name':gname.next(),'name':sname.next(),})
        if dialog:
           dialog['grid_name'].setValue(grid_name)
           dialog['name'].setValue(name)            

def play():
    global savefile
    if savefile:
        filename = savefile.name
        savefile.close()
    else:
        filename = askFilename(filter="Text files (*.txt)",exist=True)
    if filename:
        savefile = file(filename,'r')
        for line in savefile:
            print line
            globals().update(eval(line))
            create_and_show()
        savefile = file(filename,'a')


################# Dialog


def dialog_timeout():
    #print "DIALOG TIMED OUT!"
    show_shape()
    close()
        
def openSuperShapeDialogs():
    global dialog

    reset()
    smoothwire()
    lights(True)
    transparent(False)
    setView('eggview',(0.,-30.,0.))
    view('eggview')
    
    grid_items = [ [n,globals()[n]] for n in [
        'x_range','y_range','grid_size','grid_base','grid_bias','grid_skewness',
        'x_clip','y_clip','grid_name','grid_color'] ]
    # turn 'diag' into a complex input widget
    grid_items[3].extend(['radio',['quad','tri-u','tri-d','tri-x']])

    print grid_items
    
    items = [ [n,globals()[n]] for n in [
        'north_south','east_west','eggness','scale', 'post',
        'name','color'] ]

    # Action buttons
    actions = [('Close',close),('Reset',reset),('Replay',play),('Save',save),('Show Grid',show_grid),('Show',show_shape)]

    # The dialog
    dialog = InputDialog(grid_items+items,caption='SuperShape parameters',actions=actions,default='Show')

    dialog.timeout = dialog_timeout

    dialog.show()

    
if __name__ == "draw":
    if not ('__SuperShape__grid_data' in GD.PF and
            '__SuperShape__shape_data' in GD.PF):
        reset_data()
    else:
        print "set globals from GD.PF"
        print GD.PF['__SuperShape__grid_data']
        globals().update(GD.PF['__SuperShape__grid_data'])
        globals().update(GD.PF['__SuperShape__shape_data'])
        print globals()

    close()
    openSuperShapeDialogs()
    #smoothwire()

    ## while dialog is not None:
    ##     if dialog.timedOut():
    ##         show_shape()
    ##     GD.app.processEvents()
    ##     sleep(1)


# End

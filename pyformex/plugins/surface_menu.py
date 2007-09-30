#!/usr/bin/env python pyformex.py
# $Id: $
##
## This file is part of pyFormex 0.6 Release Sun Sep 30 14:33:15 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

"""surface_menu.py

STL plugin menu for pyFormex.
"""

import globaldata as GD
from gui import actors
from gui.draw import *
from plugins.surface import *
from plugins.objects import *

import commands, os, timer

##################### select, read and write ##########################


selection = DrawableObjects(clas=Surface)


def read_Surface(fn):
    GD.message("Reading file %s" % fn)
    t = timer.Timer()
    S = Surface.read(fn)
    GD.message("Read surface with %d vertices, %d edges, %d triangles in %s seconds" % (S.ncoords(),S.nedges(),S.nelems(),t.seconds()))
    return S


def readSelection(select=True,draw=True,multi=True):
    """Read a Surface (or list) from asked file name(s).

    If select is True (default), this becomes the current selection.
    If select and draw are True (default), the selection is drawn.
    """
    types = [ 'Surface Files (*.gts *.stl *.off *.neu *.smesh)', 'All Files (*)' ]
    fn = askFilename(GD.cfg['workdir'],types,exist=True,multi=multi)
    if not multi:
        fn = [ fn ]
    if fn:
        chdir(fn[0])
        names = map(utils.projectName,fn)
        GD.gui.setBusy()
        F = map(read_Surface,fn)
        GD.gui.setBusy(False)
        export(dict(zip(names,F)))
        if select:
            GD.message("Set selection to %s" % str(names))
            #for n in names:
            #    print "%s = %s" % (n,named(n))
            selection.set(names)
            if draw:
                selection.draw()
    return fn
    

def printSize(name):
    for s in selection.names:
        S = named(s)
        GD.message("Surface %s has %d vertices, %s edges and %d faces" %
                   (S.ncoords(),S.nedges(),S.nelems()))


def toFormex(suffix=''):
    """Transform the selection to Formices.

    If a suffix is given, the Formices are stored with names equal to the
    surface names plus the suffix, else, the surface names will be used
    (and the surfaces will thus be cleared from memory).
    """
    if not selection.check():
        selection.ask()

    if not selection.names:
        return

    newnames = selection.names
    if suffix:
        newnames = [ n + suffix for n in newnames ]

    newvalues = [ named(n).toFormex() for n in selection.names ]
    export2(newnames,newvalues)

    if not suffix:
        selection.clear()


def toggle_shrink():
    """Toggle the shrink mode"""
    if selection.shrink is None:
        selection.shrink = 0.8
    else:
        selection.shrink = None
    selection.draw()


def toggle_auto_draw():
    global autodraw
    autodraw = not autodraw


def convert_stl_to_off():
    """Converts an stl to off format without reading it into pyFormex."""
    fn = askFilename(GD.cfg['workdir'],"STL files (*.stl)",exist=True)
    if fn:     
        return surface.stl_to_off(fn,sanitize=False)


def sanitize_stl_to_off():
    """Sanitizes an stl to off format without reading it into pyFormex."""
    fn = askFilename(GD.cfg['workdir'],"STL files (*.stl)",exist=True)
    if fn:     
        return surface.stl_to_off(fn,sanitize=True)


def read_surface(fn='',types=['stl/off','stl','off','neu','smesh','gts'],convert=None,show=True):
    """Read STL model from file fn.

    If no file is given, one is asked.
    The file fn should exist and contain a valid surface model.
    
    The STL model is stored in the Formex F.
    The workdir and project name are set from the filename.
    The Formex is stored under the project basename.
    The model is displayed.

    If convert == True, the model is converted to a Formex.
    If convert == False, it will not be converted.
    The default is to ask the user.
    """
    if not (fn and os.path.exists(fn)):
        if type(types) == str:
            types = [ types ]
        types = map(utils.fileDescription,types)
        fn = askFilename(GD.cfg['workdir'],types,exist=True)
    if fn:
        chdir(fn)
        set_project(utils.projectName(fn))
        GD.message("Reading file %s" % fn)
        GD.gui.setBusy()
        try:
            t = timer.Timer()
            nodes,elems =surface.readSurface(fn)
            GD.message("Time to import surface: %s seconds" % t.seconds())
        finally:
            GD.gui.setBusy(False)
        set_surface(nodes,elems)
        if show:
            show_surface(view='front')
        if convert is None:
            convert = ack('Convert to Formex?')
        if convert:
            GD.debug("GOING")
            name = toFormex(PF.get('project',''))
            # This is convenient for the user
            if name:
                formex_menu.selection.set(name)
                if show:
                    formex_menu.drawSelection()
        else:
            pass
        
    return fn


def write_surface(types=['stl/off','stl','off','neu','smesh','gts']):
    if not check_surface():
        return
    if type(types) == str:
        types = [ types ]
    types = map(utils.fileDescription,types)
    fn = askFilename(GD.cfg['workdir'],types,exist=False)
    if fn:
        print "Exporting surface model to %s" % fn
        nodes,elems = PF['surface']
        GD.gui.setBusy()
        surface.writeSurface(fn,nodes,elems)   
        GD.gui.setBusy(False)



def write_stl(types=['stl']):
    if not check_stl():
        types = map(utils.fileDescription,types)
    fn = askFilename(GD.cfg['workdir'],types,exist=False)
    if fn:
        print "Exporting stl model to %s" % fn
        F = PF['stl_model']
        GD.gui.setBusy()
        surface.write_stla(fn,F.f)   
        GD.gui.setBusy(False)

#
# Operations using gts library
#
def coarsen():
    S = selection.check('single')
    if S:
        res = askItems([('min_edges',-1),
                        ('max_cost',-1),
                        ('mid_vertex',False),
                        ('length_cost',False),
                        ('max_fold',1.0),
                        ('volume_weight',0.5),
                        ('boundary_weight',0.5),
                        ('shape_weight',0.0),
                        ('progressive',False),
                        ('log',False),
                        ('verbose',False),
                        ])
        if res:
            selection.remember()
            if res['min_edges'] <= 0:
                res['min_edges'] = None
            if res['max_cost'] <= 0:
                res['max_cost'] = None
            S.coarsen(**res)
            selection.draw()


#############################################################################
# Transformation of the vertex coordinates (based on Coords)

#
# !! These functions could be made identical to those in Formex_menu
# !! (and thus could be unified) if the surface transfromations were not done
# !! inplace but returned a new surface instance instead.
#
            
def scaleSelection():
    """Scale the selection."""
    FL = selection.check()
    if FL:
        res = askItems([['scale',1.0]],
                       caption = 'Scaling Factor')
        if res:
            scale = float(res['scale'])
            selection.remember(True)
            for F in FL:
                F.scale(scale)
            selection.drawChanges()

            
def scale3Selection():
    """Scale the selection with 3 scale values."""
    FL = selection.check()
    if FL:
        res = askItems([['x-scale',1.0],['y-scale',1.0],['z-scale',1.0]],
                       caption = 'Scaling Factors')
        if res:
            scale = map(float,[res['%c-scale'%c] for c in 'xyz'])
            selection.remember(True)
            for F in FL:
                F.scale(scale)
            selection.drawChanges()


def translateSelection():
    """Translate the selection."""
    FL = selection.check()
    if FL:
        res = askItems([['direction',0],['distance','1.0']],
                       caption = 'Translation Parameters')
        if res:
            dir = int(res['direction'])
            dist = float(res['distance'])
            selection.remember(True)
            for F in FL:
                F.translate(dir,dist)
            selection.drawChanges()


def centerSelection():
    """Center the selection."""
    FL = selection.check()
    if FL:
        selection.remember(True)
        for F in FL:
            F.translate(-F.coords.center())
        selection.drawChanges()


def rotateSelection():
    """Rotate the selection."""
    FL = selection.check()
    if FL:
        res = askItems([['axis',2],['angle','90.0']])
        if res:
            axis = int(res['axis'])
            angle = float(res['angle'])
            selection.remember(True)
            for F in FL:
                F.rotate(angle,axis)
            selection.drawChanges()


def rotateAround():
    """Rotate the selection."""
    FL = selection.check()
    if FL:
        res = askItems([['axis',2],['angle','90.0'],['around','[0.0,0.0,0.0]']])
        if res:
            axis = int(res['axis'])
            angle = float(res['angle'])
            around = eval(res['around'])
            GD.debug('around = %s'%around)
            selection.remember(True)
            for F in FL:
                F.rotate(angle,axis,around)
            selection.drawChanges()


def rollAxes():
    """Rotate the selection."""
    FL = selection.check()
    if FL:
        selection.remember(True)
        for F in FL:
            F.coords.rollAxes()
        selection.drawChanges()


        
def clip_surface():
    """Clip the stl model."""
    if not check_surface():
        return
    itemlist = [['axis',0],['begin',0.0],['end',1.0],['nodes','any']]
    res,accept = widgets.InputDialog(itemlist,'Clipping Parameters').getResult()
    if accept:
        updateGUI()
        nodes,elems = PF['old_surface'] = PF['surface']
        F = Formex(nodes[elems])
        bb = F.bbox()
        GD.message("Original bbox: %s" % bb) 
        xmi = bb[0][0]
        xma = bb[1][0]
        dx = xma-xmi
        axis = int(res[0][1])
        xc1 = xmi + float(res[1][1]) * dx
        xc2 = xmi + float(res[2][1]) * dx
        nodid = res[3][1]
        #print nodid
        clear()
        draw(F,color='yellow')
        w = F.test(nodes='any',dir=axis,min=xc1,max=xc2)
        F = F.clip(w)
        draw(F,color='red')
        


def undo_stl():
    """Undo the last transformation."""
    global F,oldF
    clear()
    linewidth(1)
    F = oldF
    draw(F,color='green')

def fill_holes():
    global F,oldF
    fn = project + '.stl'
    fn1 = project + '-closed.stl'
    if os.path.exists(fn):
        sta,out = commands.getstatusoutput('admesh %s -f -a %s' % (fn,fn1))
        GD.message(out)
        if sta == 0:
            clear()
            linewidth(1)
            draw(F,color='yellow',view='front')
            oldF = F
            linewidth(2)
            GD.gui.setBusy()
            surface.readSurface(fn1)
            GD.gui.setBusy(False)


def flytru_stl():
    """Fly through the stl model."""
    global ctr
    Fc = Formex(array(ctr).reshape((-1,1,3)))
    path = connect([Fc,Fc],bias=[0,1])
    flyAlong(path)
    

def export_stl():
    """Export an stl model stored in Formex F in Abaqus .inp format."""
    global project,F
    if ack("Creating nodes and elements.\nFor a large model, this could take quite some time!"):
        GD.app.processEvents()
        GD.message("Creating nodes and elements.")
        nodes,elems = F.feModel()
        nnodes = nodes.shape[0]
        nelems = elems.shape[0]
        GD.message("There are %d unique nodes and %d triangle elements in the model." % (nnodes,nelems))
        stl_abq.abq_export(project+'.inp',nodes,elems,'S3',"Created by stl_examples.py")

def export_surface():
    if PF['surface'] is None:
        return
    types = [ "Abaqus INP files (*.inp)" ]
    fn = askFilename(GD.cfg['workdir'],types,exist=False)
    if fn:
        print "Exporting surface model to %s" % fn
        updateGUI()
        nodes,elems = PF['surface']
        stl_abq.abq_export(fn,nodes,elems,'S3',"Abaqus model generated by pyFormex from input file %s" % os.path.basename(fn))



def export_volume():
    if PF['volume'] is None:
        return
    types = [ "Abaqus INP files (*.inp)" ]
    fn = askFilename(GD.cfg['workdir'],types,exist=False)
    if fn:
        print "Exporting volume model to %s" % fn
        updateGUI()
        nodes,elems = PF['volume']
        stl_abq.abq_export(fn,nodes,elems,'C3D%d' % elems.shape[1],"Abaqus model generated by tetgen from surface in STL file %s.stl" % PF['project'])


def show_nodes():
    n = 0
    data = askItems({'node number':n})
    n = int(data['node number'])
    if n > 0:
        nodes,elems = PF['surface']
        print "Node %s = %s",(n,nodes[n])


def trim_border(elems,nodes,nb,visual=False):
    """Removes the triangles with nb or more border edges.

    Returns an array with the remaining elements.
    """
    b = border(elems)
    b = b.sum(axis=1)
    trim = where(b>=nb)[0]
    keep = where(b<nb)[0]
    nelems = elems.shape[0]
    ntrim = trim.shape[0]
    nkeep = keep.shape[0]
    print "Selected %s of %s elements, leaving %s" % (ntrim,nelems,nkeep) 

    if visual and ntrim > 0:
        prop = zeros(shape=(F.nelems(),),dtype=int32)
        prop[trim] = 2 # red
        prop[keep] = 1 # yellow
        F = Formex(nodes[elems],prop)
        clear()
        draw(F,view='left')
        
    return elems[keep]


def trim_surface():
    check_surface()
    data = GD.cfg.get('stl/border',{'Number of trim rounds':1, 'Minimum number of border edges':1})
    GD.cfg['stl/border'] = askItems(data)
    GD.gui.update()
    n = int(data['Number of trim rounds'])
    nb = int(data['Minimum number of border edges'])
    print "Initial number of elements: %s" % elems.shape[0]
    for i in range(n):
        elems = trim_border(elems,nodes,nb)
        print "Number of elements after border removal: %s" % elems.shape[0]

    

def create_tetgen():
    """Generate a volume tetraeder mesh inside an stl surface."""
    fn = PF['project'] + '.stl'
    if os.path.exists(fn):
        sta,out = commands.getstatusoutput('tetgen -z %s' % fn)
        GD.message(out)


def read_tetgen(surface=True, volume=True):
    """Read a tetgen model from files  fn.node, fn.ele, fn.smesh."""
    ftype = ''
    if surface:
        ftype += ' *.smesh'
    if volume:
        ftype += ' *.ele'
    fn = askFilename(GD.cfg['workdir'],"Tetgen files (%s)" % ftype,exist=True)
    nodes = elems =surf = None
    if fn:
        chdir(fn)
        project = utils.projectName(fn)
        set_project(project)
        nodes,nodenrs = tetgen.readNodes(project+'.node')
#        print "Read %d nodes" % nodes.shape[0]
        if volume:
            elems,elemnrs,elemattr = tetgen.readElems(project+'.ele')
            print "Read %d tetraeders" % elems.shape[0]
            PF['volume'] = (nodes,elems)
        if surface:
            surf = tetgen.readSurface(project+'.smesh')
            print "Read %d triangles" % surf.shape[0]
            PF['surface'] = (nodes,surf)
    if surface:
        show_surface()
    else:
        show_volume()


def read_tetgen_surface():
    read_tetgen(volume=False)

def read_tetgen_volume():
    read_tetgen(surface=False)


def scale_volume():
    if PF['volume'] is None:
        return
    nodes,elems = PF['volume']
    nodes *= 0.01
    PF['volume'] = (nodes,elems) 
    



def show_volume():
    """Display the volume model."""
    if PF['volume'] is None:
        return
    nodes,elems = PF['volume']
    F = Formex(nodes[elems])
    GD.message("BBOX = %s" % F.bbox())
    clear()
    draw(F,color='random',eltype='tet')
    PF['vol_model'] = F


################### menu #################

_menu = None

def create_menu():
    """Create the Surface menu."""
    MenuData = [
        ("&Read Surface Files",readSelection),
        ("&Select Surface(s)",selection.ask),
        ("&Draw Selection",selection.draw),
        ("&Forget Selection",selection.forget),
        ('&List Selection',printSize),
        ("&Convert to Formex",toFormex),
#        ("&Write Surface Model",write_surface),
#        ("&Write STL Model",write_stl),
        ("---",None),
#        ("&Set Property",setProperty),
        ("&Shrink",toggle_shrink),
#        ("&Toggle Names",toggleNames),
#        ("&Toggle Numbers",toggleNumbers),
        ("&Undo Last Changes",selection.undoChanges),
        ("---",None),
        ("&Coarsen surface",coarsen),
        ("---",None),
        ("&Transform",
         [("&Scale Selection",scaleSelection),
          ("&Non-uniformly Scale Selection",scale3Selection),
          ("&Translate Selection",translateSelection),
          ("&Center Selection",centerSelection),
          ("&Rotate Selection",rotateSelection),
          ("&Rotate Selection Around",rotateAround),
#          ("&Roll Axes",rollAxes),
#          ("&Clip Selection",clipSelection),
#          ("&Cut at Plane",cutAtPlane),
          ]),
        ("---",None),
#        ("&Show volume model",show_volume),
        #("&Print Nodal Coordinates",show_nodes),
        # ("&Convert STL file to OFF file",convert_stl_to_off),
        # ("&Sanitize STL file to OFF file",sanitize_stl_to_off),
#        ("&Clip model",clip_surface),
#        ("&Trim border",trim_surface),
#        ("&Undo LAST STL transformation",undo_stl),
#        ("&Fill the holes in STL model",fill_holes),
#        ("&Create tetgen model",create_tetgen),
#        ("&Read Tetgen Volume",read_tetgen_volume),
#        ("&Scale Volume model with factor 0.01",scale_volume),
        ("&Export surface to Abaqus",export_surface),
#        ("&Export volume to Abaqus",export_volume),
        ("&Close Menu",close_menu),
        ]
    return widgets.Menu('Surface',items=MenuData,parent=GD.gui.menu,before='help')


def close_menu():
    """Close the STL menu."""
    global _menu
    if _menu:
        _menu.remove()
    _menu = None

    
def show_menu():
    """Show the surface menu."""
    global _menu
    if not _menu:
        _menu = create_menu()
##     PF['volume'] = None
##     PF['stl_model'] = None
##     PF['stl_color'] = colors.blue    # could be replaced by a general fgcolor


    

if __name__ == "main":
    print __doc__

# End

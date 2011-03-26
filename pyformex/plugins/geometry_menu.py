# $Id$
"""geometry_menu.py

This is a pyFormex plugin menu. It is not intended to be executed as a script,
but to be loaded into pyFormex using the plugin facility.

The geometry menu is intended to become the major interactive geometry menu
in pyFormex. 
"""

import pyformex as pf
import utils
from odict import ODict

from geometry import Geometry
from geomfile import GeometryFile
from formex import Formex
from mesh import Mesh
from trisurface import TriSurface

from gui import actors
from gui import menu
from gui.draw import *

from plugins import objects,trisurface,inertia,partition,sectionize

import commands, os, timer

################### automatic naming of objects ##########################

autoname = {}

def autoName(clas):
    """Return the autoname class instance for objects of type clas.

    clas can be either a class instance, a class object, or a string
    with a class name. The resulting autoname object will generate
    strings starting with the class name in lower case.

    >>> F = Formex()
    >>> print autoName(F).next()
    formex-000
    """
    if isinstance(clas,str):
        name = clas
    else:
        try:
            name = clas.__name__
        except:
            try:
                name = clas.__class__.__name__
            except:
                raise ValueError,"Expected an instance, class or string"
    name = name.lower()
    if not name in autoname:
        autoname[name] = utils.NameSequence(name)
    return autoname[name]

##################### selection of objects ##########################

# We put these in an init function to allow --testmodule to run without GUI

def _init_():
    global drawable,selection,selection_F,selection_M,selection_TS,selection_PL,selection_NC,setSelection,drawSelection,drawAll
    drawable = pf.GUI.drawable
    selection = pf.GUI.selection['geometry']
    ## selection_F = pf.GUI.selection['formex']
    ## selection_M = pf.GUI.selection['mesh']
    ## selection_TS = pf.GUI.selection['surface']
    ## selection_PL = pf.GUI.selection['polyline']
    ## selection_NC = pf.GUI.selection['nurbs']
    ## setSelection = selection.set
    ## drawSelection = selection.draw
    drawAll = drawable.draw
    

def set_selection(section='geometry'):
    print "SETTING %s" % section
    sel = pf.GUI.selection.get(section)
    if sel:
        sel.ask()
        selection.set(sel.names)
        selection.draw()
        

##################### read and write ##########################

def readGeometry(filename,filetype=None):
    """Read geometry from a stored file.

    This is a wrapper function over several other functions specialized
    for some file type. Some file types require the existence of more
    than one file, may need to write intermediate files, or may call
    external programs.
    
    The return value is a dictionary with named geometry objects read from
    the file.

    If no filetype is given, it is derived from the filename extension.
    Currently the following file types can be handled. 

    'pgf': pyFormex Geometry File. This is the native pyFormex geometry
        file format. It can store multiple parts of different type, together
        with their name.
    'surface': a global filetype for any of the following surface formats:
        'stl', 'gts', 'off',
    'stl': 
    """
    res = {}
    if filetype is None:
        filetype = utils.fileTypeFromExt(filename)
        
    print "Reading file of type %s" % filetype

    if filetype == 'pgf':
        res = GeometryFile(filename).read()

    elif filetype in ['surface','stl','off','gts','smesh','neu']:
        surf = TriSurface.read(filename)
        res = {autoName(TriSurface).next():surf}

    else:
        error("Can not import from file %s of type %s" % (filename,filetype))
        
    return res


def importGeometry(select=True,draw=True,ftype=None):
    """Read geometry from file.
    
    If select is True (default), the imported geometry becomes the current
    selection.
    If select and draw are True (default), the selection is drawn.
    """
    if ftype is None:
        ftype = ['pgf','pyf','surface','off','stl','gts','smesh','neu','all']
    else:
        ftype = [ftype]
    types = utils.fileDescription(ftype)
    cur = pf.cfg['workdir']
    fn = askFilename(cur=cur,filter=types)
    if fn:
        message("Reading geometry file %s" % fn)
        res = readGeometry(fn)
        export(res)
        #selection.readFromFile(fn)
        print res.keys()
        if select:
            selection.set(res.keys())
            if draw:
                selection.draw()
                zoomAll()


def importPgf():
    importGeometry(ftype='pgf')

def importSurface():
    importGeometry(ftype='surface')

def importAny():
    importGeometry(ftype=None)
     

def writeGeometry(obj,filename,filetype=None,shortlines=False):
    """Write the geometry items in objdict to the specified file.

    """
    if filetype is None:
        filetype = utils.fileTypeFromExt(filename)
        
    print "Writing file of type %s" % filetype

    if filetype in [ 'pgf', 'pyf' ]:
        # Can write anything
        if filetype == 'pgf':
            res = writeGeomFile(filename,obj,shortlines=shortlines)

    else:
        error("Don't know how to export in '%s' format" % filetype)
        
    return res


def exportGeometry(types=['pgf','all'],shortlines=False):
    """Write geometry to file."""
    drawable.ask()
    if not drawable.check():
        return
    
    filter = utils.fileDescription(types)
    cur = pf.cfg['workdir']
    fn = askNewFilename(cur=cur,filter=filter)
    if fn:
        message("Writing geometry file %s" % fn)
        res = writeGeometry(drawable.odict(),fn,shortlines=shortlines)
        pf.message("Contents: %s" % res)


def exportPgf():
    exportGeometry(['pgf'])
def exportPgfShortlines():
    exportGeometry(['pgf'],shortlines=True)
def exportOff():
    exportGeometry(['off'])
 
    
##################### properties ##########################

def printDataSize():
    for s in selection.names:
        S = named(s)
        #try:
        size = S.dataReport()
        #except:
        #    size = ''
        pf.message("* %s (%s): %s" % (s,S.__class__.__name__,size))

################### menu #################
 
_menu = 'Geometry'

def create_menu():
    """Create the plugin menu."""
    _init_()
    MenuData = [
        ("&Import ",[
            ("pyFormex Geometry File (.pgf)",importPgf),
            ("Surface File (*.gts, *.stl, *.off, *.neu)",importSurface),
            ("All known geometry formats",importAny),
            ]),
        ("&Export ",[
            ("pyFormex Geometry File (.pgf)",exportPgf),
            ("pyFormex Geometry File with short lines",exportPgfShortlines),
            ("Object File Format (.off)",exportOff),
            ]),
        ("&Select ",[
            ('Any',selection.ask),
            ('Formex',set_selection,{'data':'formex'}),
            ('Mesh',set_selection,{'data':'mesh'}),
            ('TriSurface',set_selection,{'data':'surface'}),
            ('PolyLine',set_selection,{'data':'polyline'}),
            ('Curve',set_selection,{'data':'curve'}),
            ('NurbsCurve',set_selection,{'data':'nurbs'}),
            ]),
        ("&Draw Selection",selection.draw),
        ("&Forget Selection",selection.forget),
        ("---",None),
        ("Print &Information ",[
            ('&Bounding Box',selection.printbbox),
            ('&Data Size',printDataSize),
            ]),
        ("Toggle &Annotations ",[
            ("&Names",selection.toggleNames,dict(checkable=True)),
            ("&Elem Numbers",selection.toggleNumbers,dict(checkable=True)),
            ("&Node Numbers",selection.toggleNodeNumbers,dict(checkable=True,checked=selection.hasNodeNumbers())),
            ("&Node Marks",selection.toggleNodes,dict(checkable=True,checked=selection.hasNodeMarks())),
            ('&Toggle Bbox',selection.toggleBbox,dict(checkable=True)),
            ]),
        ## ("---",None),
        ## ("&Set Property",selection.setProperty),
        ## ("&Shrink",shrink),
        ## ("&Bbox",
        ##  [('&Show Bbox Planes',showBbox),
        ##   ('&Remove Bbox Planes',removeBbox),
        ##   ]),
        ## ("&Transform",
        ##  [("&Scale Selection",scaleSelection),
        ##   ("&Scale non-uniformly",scale3Selection),
        ##   ("&Translate",translateSelection),
        ##   ("&Center",centerSelection),
        ##   ("&Rotate",rotateSelection),
        ##   ("&Rotate Around",rotateAround),
        ##   ("&Roll Axes",rollAxes),
        ##   ]),
        ## ("&Clip/Cut",
        ##  [("&Clip",clipSelection),
        ##   ("&Cut With Plane",cutSelection),
        ##   ]),
        ## ("&Undo Last Changes",selection.undoChanges),
        ## ("---",None),
        ## ("Show &Principal Axes",showPrincipal),
        ## ("Rotate to &Principal Axes",rotatePrincipal),
        ## ("Transform to &Principal Axes",transformPrincipal),
        ## ("---",None),
        ## ("&Concatenate Selection",concatenateSelection),
        ## ("&Partition Selection",partitionSelection),
        ## ("&Create Parts",createParts),
        ## ("&Sectionize Selection",sectionizeSelection),
        ## ("---",None),
        ## ("&Fly",fly),
        ("---",None),
        ("&Reload menu",reload_menu),
        ("&Close",close_menu),
        ]
    return menu.Menu(_menu,items=MenuData,parent=pf.GUI.menu,before='help')

    
def show_menu():
    """Show the Tools menu."""
    if not pf.GUI.menu.item(_menu):
        create_menu()


def close_menu():
    """Close the Tools menu."""
    m = pf.GUI.menu.item(_menu)
    if m :
        m.remove()
      

def reload_menu():
    """Reload the Postproc menu."""
    from plugins import refresh
    close_menu()
    refresh('geometry_menu')
    show_menu()


####################################################################
######### What to do when the script is executed ###################

if __name__ == "draw":
    print "GO"
    _init_()
    reload_menu()


# End


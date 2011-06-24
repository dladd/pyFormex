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
from connectivity import Connectivity
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
    global drawable,selection,drawAll
    global selection_F,selection_M,selection_TS,selection_PL,selection_NC,setSelection,drawSelection
    drawable = pf.GUI.drawable
    selection = pf.GUI.selection['geometry']
    drawAll = drawable.draw
    ## selection_F = pf.GUI.selection['formex']
    ## selection_M = pf.GUI.selection['mesh']
    ## selection_TS = pf.GUI.selection['surface']
    ## selection_PL = pf.GUI.selection['polyline']
    ## selection_NC = pf.GUI.selection['nurbs']
    ## setSelection = selection.set
    ## drawSelection = selection.draw
    

def set_selection(clas='geometry'):
    sel = pf.GUI.selection.get(clas)
    if sel:
        sel.ask()
        selection.set(sel.names)
        selection.draw()
    else:
        warning('Nothing to select')
        

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


def getParams(line):
    """Strip the parameters from a comment line"""
    s = line.split()
    d = {'mode': s.pop(0),'filename': s.pop(0)}
    d.update(dict(zip(s[::2],s[1::2])))
    return d
    

def readNodes(fil):
    """Read a set of nodes from an open mesh file"""
    a = fromfile(fil,sep=" ").reshape(-1,3)
    x = Coords(a)
    print(x.shape)
    return x


def readElems(fil,nplex):
    """Read a set of elems of plexitude nplex from an open mesh file"""
    print("Reading elements of plexitude %s" % nplex)
    e = fromfile(fil,sep=" ",dtype=Int).reshape(-1,nplex) 
    e = Connectivity(e)
    print(e.shape)
    return e


def readEsets(fil):
    """Read the eset data of type generate"""
    data = []
    for line in fil:
        s = line.strip('\n').split()
        if len(s) == 4:
            data.append(s[:1]+map(int,s[1:]))
    return data
            

def readMesh(fn):
    """Read a nodes/elems model from file.

    Returns an (x,e) tuple or None
    """
    d = {}
    pf.GUI.setBusy(True)
    fil = file(fn,'r')
    for line in fil:
        if line[0] == '#':
            line = line[1:]
        globals().update(getParams(line))
        dfil = file(filename,'r')
        if mode == 'nodes':
            d['coords'] = readNodes(dfil)
        elif mode == 'elems':
            elems = d.setdefault('elems',[])
            e = readElems(dfil,int(nplex)) - int(offset)
            elems.append(e)
        elif mode == 'esets':
            d['esets'] = readEsets(dfil)
        else:
            print("Skipping unrecognized line: %s" % line)
        dfil.close()

    pf.GUI.setBusy(False)
    fil.close()
    return d                    


def importModel(fn=None):
    """Read one or more element meshes into pyFormex.

    Models are composed of matching nodes.txt and elems.txt files.
    A single nodes fliename or a list of node file names can be specified.
    If none is given, it will be asked from the user.
    """

    if fn is None:
        fn = askFilename(".","*.mesh",multi=True)
        if not fn:
            return
    if type(fn) == str:
        fn = [fn]
        
    for f in fn:
        d = readMesh(f)
        print(type(d))
        x = d['coords']
        e = d['elems']

        modelname = os.path.basename(f).replace('.mesh','')
        export({modelname:d})
        export(dict([("%s-%d"%(modelname,i), Mesh(x,ei)) for i,ei in enumerate(e)])) 


def convert_inp(fn=None):
    """Convert an Abaqus .inp file to pyFormex .mesh.

    """
    if fn is None:
        fn = askFilename(".","*.inp",multi=True)
        if not fn:
            return

        for f in fn:
            convert_inp(f)
        return


    converter = os.path.join(pf.cfg['pyformexdir'],'bin','read_abq_inp.awk')
    dirname = os.path.dirname(fn)
    basename = os.path.basename(fn)
    cmd = 'cd %s;%s %s' % (dirname,converter,basename)
    print(cmd)
    pf.GUI.setBusy()
    print(utils.runCommand(cmd))
    pf.GUI.setBusy(False)
     

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
 

def convertGeometryFile():
    """Convert pyFormex geometry file to latest format."""
    filter = utils.fileDescription(['pgf','all'])
    cur = pf.cfg['workdir']
    fn = askFilename(cur=cur,filter=filter)
    if fn:
        from geomfile import GeometryFile
        message("Converting geometry file %s to version %s" % (fn,GeometryFile._version_))
        GeometryFile(fn).rewrite()
    
##################### properties ##########################

def printDataSize():
    for s in selection.names:
        S = named(s)
        try:
            size = S.info()
        except:
            size = 'no info available'
        pf.message("* %s (%s): %s" % (s,S.__class__.__name__,size))


##################### conversion ##########################

def toFormex(suffix=''):
    """Transform the selected Geometry objects to Formices.

    If a suffix is given, the Formices are stored with names equal to the
    object names plus the suffix, else, the original object names will be
    reused.
    """
    if not selection.check():
        selection.ask()

    if not selection.names:
        return

    names = selection.names
    if suffix:
        names = [ n + suffix for n in names ]

    values = [ named(n).toFormex() for n in names ]
    export2(names,values)

    clear()
    selection.draw()
    

def toMesh(suffix=''):
    """Transform the selected Geometry objects to Meshes.

    If a suffix is given, the TriSurfaces are stored with names equal to the
    Formex names plus the suffix, else, the Formex names will be used
    (and the Formices will thus be cleared from memory).
    """
    if not selection.check():
        selection.ask()

    if not selection.names:
        return

    names = selection.names
    objects = [ named(n) for n in names ]
    if suffix:
        names = [ n + suffix for n in names ]

    print "CONVERTING %s" % names
    meshes =  dict([ (n,o.toMesh()) for n,o in zip(names,objects) if hasattr(o,'toMesh')])
    print("Converted %s" % meshes.keys())
    export(meshes)

    selection.set(meshes.keys())


def splitProp():
    """Split the selected object based on property values"""
    from plugins import partition
    
    F = selection.check(single=True)
    if not F:
        return

    name = selection[0]
    partition.splitProp(F,name)
    
 
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
            ("&Convert Abaqus .inp file",convert_inp),
            ("&Import Converted Abaqus Model",importModel),
            ('&Convert pyFormex Geometry File',convertGeometryFile,dict(tooltip="Convert a pyFormex Geometry File (.pgf) to the latest format, overwriting the file.")),
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
            ('&Type and Size',printDataSize),
            ]),
        ("Toggle &Annotations ",[
            ("&Names",selection.toggleNames,dict(checkable=True)),
            ("&Elem Numbers",selection.toggleNumbers,dict(checkable=True)),
            ("&Node Numbers",selection.toggleNodeNumbers,dict(checkable=True,checked=selection.hasNodeNumbers())),
            ("&Free Edges",selection.toggleFreeEdges,dict(checkable=True,checked=selection.hasFreeEdges())),
            ("&Node Marks",selection.toggleNodes,dict(checkable=True,checked=selection.hasNodeMarks())),
            ('&Toggle Bbox',selection.toggleBbox,dict(checkable=True)),
            ]),
        ("---",None),
        ("&Convert",[
            ("To &Formex",toFormex),
            ("To &Mesh",toMesh),
            ## ("To &TriSurface",toSurface),
            ## ("To &PolyLine",toPolyLine),
            ## ("To &BezierSpline",toBezierSpline),
            ## ("To &NurbsCurve",toNurbsCurve),
            ]),
        ("&Property Numbers",[
            ("&Set",selection.setProp),
            ("&Delete",selection.delProp),
            ("&Split",splitProp),
            ]),
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


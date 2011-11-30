# $Id$
##
##  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
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
from mesh import Mesh,mergeMeshes
from trisurface import TriSurface
from elements import elementType
import simple

from gui import actors
from gui import menu
from gui.draw import *
from gui.widgets import simpleInputItem as _I, groupInputItem as _G

from plugins import objects,trisurface,inertia,partition,sectionize,dxf

import commands, os, timer

################### automatic naming of objects ##########################

_autoname = {}
autoname = _autoname    # alias used by some other plugins that still need to be updated

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
    if not name in _autoname:
        _autoname[name] = utils.NameSequence(name)
    return _autoname[name]

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
    fil = open(fn,'r')
    for line in fil:
        if line[0] == '#':
            line = line[1:]
        globals().update(getParams(line))
        dfil = open(filename,'r')
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


def readInp(fn=None):
    """Read an Abaqus .inp file and convert to pyFormex .mesh.

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

    
def importDxf(convert=False,keep=False):
    """Import a DXF file.

    The user is asked for the name of a .DXF file. Depending on the
    parameters, the following processing is done:

    =======     =====     ================================================
    convert     keep      actions
    =======     =====     ================================================
    False       False     import DXF entities to pyFormex (default)
    False       True      import DXF and save intermediate .dxftext format
    True        any       convert .dxf to .dxftext only
    =======     =====     ================================================

    If convert == False, this function returns the list imported DXF entities.
    """
    fn = askFilename(filter=utils.fileDescription('dxf'))
    if not fn:
        return

    pf.GUI.setBusy()    
    text = dxf.readDXF(fn)
    pf.GUI.setBusy(False)
    if text:
        if convert or keep:
            f = file(utils.changeExt(fn,'.dxftext'),'w')
            f.write(text)
            f.close()
        if not convert:
            return importDxftext(text)


def importSaveDxf():
    """Import a DXF file and save the intermediate .dxftext."""
    importDxf(keep=True)

    
def convertDxf():
    """Read a DXF file and convert to dxftext."""
    importDxf(convert=True)

    
def importDxftext(text=None):
    """Import a dxftext script or file.

    A dxftext script is a script containing only function calls that
    generate dxf entities. See :func:`dxf.convertDXF`.

    - Without parameter, the name of a .dxftext file is asked and the
      script is read from that file.
    - If `text` is a single line string, it is used as the filename of the
      script.
    - If `text` contains at least one newline character, it is interpreted
      as the dxf script text.
    """
    if text is None:
        fn = askFilename(filter=utils.fileDescription('dxftext'))
        if not fn:
            return
        text = open(fn).read()
    elif '\n' not in text:
        text = open(text).read()
        
    pf.GUI.setBusy()    
    dxf_parts = dxf.convertDXF(text)
    pf.GUI.setBusy(False)
    export({'_dxf_import_':dxf_parts})
    draw(dxf_parts,color='black')
    return dxf_parts
           

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


#############################################
###  Property functions
#############################################


def splitProp():
    """Split the selected object based on property values"""
    from plugins import partition
    
    F = selection.check(single=True)
    if not F:
        return

    name = selection[0]
    partition.splitProp(F,name)


#############################################
###  Create Geometry functions
#############################################


def convertFormex(F,totype):
    if totype != 'Formex':
        F = F.toMesh()
        if totype == 'TriSurface':
            F = TriSurface(F)
    return F
    

def createCylinder():
    _data_ = __name__+'_createCylinder_data'
    res = {
        'name':'__auto__',
        'object type':'Formex',
        'base diameter':1.,
        'top diameter':1.,
        'height':2.,
        'angle':360.,
        'div_along_length':6,
        'div_along_circ':12,
        'bias':0.,
        'diagonals':'up',
        }
    if pf.PF.has_key(_data_):
        res.update(pf.PF[_data_])
    res = askItems(store=res, items=[
        _I('name'),
        _I('object type',choices=['Formex','Mesh','TriSurface']),
        _I('base diameter'),
        _I('top diameter'),
        _I('height'),
        _I('angle'),
        _I('div_along_length'),
        _I('div_along_circ'),
        _I('bias'),
        _I('diagonals',choices=['none','up','down']),
        ])

    if res:
        pf.PF[_data_] = res
        name = res['name']
        if name == '__auto__':
            name = autoName(res['object type']).next()

        F = simple.cylinder(L=res['height'],D=res['base diameter'],D1=res['top diameter'],
                            angle=res['angle'],nt=res['div_along_circ'],nl=res['div_along_length'],
                            bias=res['bias'],diag=res['diagonals'][0])

        F = convertFormex(F,res['object type'])
        export({name:F})
        selection.set([name])
        selection.draw()


def createCone():
    _data_ = __name__+'_createCone_data'
    res = {
        'name' : '__auto__',
        'object type':'Formex',
        'radius': 1.,
        'height': 1.,
        'angle': 360.,
        'div_along_radius': 6,
        'div_along_circ':12,
        'diagonals':'up',
        }
    if pf.PF.has_key(_data_):
        res.update(pf.PF[_data_])
        
    res = askItems(store=res, items=[
        _I('name'),
        _I('object type',choices=['Formex','Mesh','TriSurface']),
        _I('radius'),
        _I('height'),
        _I('angle'),
        _I('div_along_radius'),
        _I('div_along_circ'),
        _I('diagonals',choices=['none','up','down']),
        ])
    
    if res:
        pf.PF[_data_] = res
        name = res['name']
        if name == '__auto__':
            name = autoName(res['object type']).next()

        F = simple.sector(r=res['radius'],t=res['angle'],nr=res['div_along_radius'],
                          nt=res['div_along_circ'],h=res['height'],diag=res['diagonals'])
        
        F = convertFormex(F,res['object type'])
        export({name:F})
        selection.set([name])
        selection.draw()

#############################################
###  Mesh functions
#############################################
       
def narrow_selection(clas):
    global selection
    print "BEFORE",selection.names
    selection.set([n for n in selection.names if isinstance(named(n),clas)])
    print "BEFORE",selection.names
    


def fuseMesh():
    """Fuse the nodes of a Mesh"""
    if not selection.check():
        selection.ask()

    narrow_selection(Mesh)

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    res = askItems([
        _I('Relative Tolerance',1.e-5),
        _I('Absolute Tolerance',1.e-5),
        _I('Shift',0.5),
        _I('Nodes per box',1)])

    if not res:
        return

    before = [ m.ncoords() for m in meshes ]
    meshes = [ m.fuse(
        rtol = res['Relative Tolerance'],
        atol = res['Absolute Tolerance'],
        shift = res['Shift'],
        nodesperbox = res['Nodes per box'],
        ) for m in meshes ]
    after = [ m.ncoords() for m in meshes ]
    print "Number of points before fusing: %s" % before
    print "Number of points after fusing: %s" % after

    names = [ "%s_fused" % n for n in selection.names ]
    export2(names,meshes)
    selection.set(names)
    clear()
    selection.draw()



def divideMesh():
    """Create a mesh by subdividing existing elements.

    """
    if not selection.check():
        selection.ask()

    narrow_selection(Mesh)

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    eltypes = set([ m.eltype for m in meshes if m.eltype is not None])
    print "eltypes in selected meshes: %s" % eltypes
    if len(eltypes) > 1:
        warning("I can only divide meshes with the same element type\nPlease narrow your selection before trying conversion.")
        return
    if len(eltypes) == 1:
        fromtype = eltypes.pop()
    showInfo("Sorry, this function is not implemented yet!")


def convertMesh():
    """Transform the element type of the selected meshes.

    """
    if not selection.check():
        selection.ask()

    narrow_selection(Mesh)
    
    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    eltypes = set([ m.eltype.name() for m in meshes if m.eltype is not None])
    print "eltypes in selected meshes: %s" % eltypes
    if len(eltypes) > 1:
        warning("I can only convert meshes with the same element type\nPlease narrow your selection before trying conversion.")
        return
    if len(eltypes) == 1:
        fromtype = elementType(eltypes.pop())
        choices = ["%s -> %s" % (fromtype,to) for to in fromtype.conversions.keys()]
        if len(choices) == 0:
            warning("Sorry, can not convert a %s mesh"%fromtype)
            return
        res = askItems([
            _I('_conversion',itemtype='vradio',text='Conversion Type',choices=choices),
            _I('_compact',True),
            _I('_merge',itemtype='hradio',text="Merge Meshes",choices=['None','Each','All']),
            ])
        if res:
            globals().update(res)
            print "Selected conversion %s" % _conversion
            totype = _conversion.split()[-1]
            names = [ "%s_converted" % n for n in selection.names ]
            meshes = [ m.convert(totype) for m in meshes ]
            if _merge == 'Each':
                meshes = [ m.fuse() for m in meshes ]
            elif  _merge == 'All':
                print _merge
                coords,elems = mergeMeshes(meshes)
                print elems
                ## names = [ "_merged_mesh_%s" % e.nplex() for e in elems ]
                ## meshes = [ Mesh(coords,e,eltype=meshes[0].eltype) for e in elems ]
                ## print meshes[0].elems
                meshes = [ Mesh(coords,e,m.prop,m.eltype) for e,m in zip(elems,meshes) ]
            if _compact:
                print "compacting meshes"
                meshes = [ m.compact() for m in meshes ]
                
            export2(names,meshes)
            selection.set(names)
            clear()
            selection.draw()


def renumberMeshInElemsOrder():
    """Renumber the selected Meshes in elems order.

    """
    if not selection.check():
        selection.ask()

    narrow_selection(Mesh)

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    names = selection.names
    meshes = [ M.renumber() for M in meshes ]
    export2(names,meshes)
    selection.set(names)
    clear()
    selection.draw()


 
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
            ("Abaqus .inp",[
                ("&Convert Abaqus .inp file",readInp),
                ("&Import Converted Abaqus Model",importModel),
                ]),
            ("AutoCAD .dxf",[
                ("&Import AutoCAD .dxf",importDxf),
                ("&Import AutoCAD .dxf and save .dxftext",importSaveDxf),
                ("&Convert AutoCAD .dxf to .dxftext",convertDxf,dict(tooltip="Parse a .DXF file and output dxftext script.")),
                ("&Import .dxftext",importDxftext),
                ]),
            ('&Upgrade pyFormex Geometry File',convertGeometryFile,dict(tooltip="Convert a pyFormex Geometry File (.pgf) to the latest format, overwriting the file.")),
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
        ("&Create Object",[
            ('&Cylinder, Cone, Truncated Cone',createCylinder),
            ('&Circle, Sector, Cone',createCone),
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
        ("Mesh",[
            ("&Convert element type",convertMesh),
            ("&Subdivide",divideMesh),
            ("&Fuse nodes",fuseMesh),
            ("&Renumber nodes in element order",renumberMeshInElemsOrder),
            ]),
        ("---",None),
        ("&Reload menu",reload_menu),
        ("&Close",close_menu),
        ]
    M = menu.Menu(_menu,items=MenuData,parent=pf.GUI.menu,before='help')
    ## if not utils.hasExternal('dxfparser'):
    ##     I = M.item("&Import ").item("AutoCAD .dxf")
    ##     I.setEnabled(False)
    return M

    
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
    _init_()
    reload_menu()


# End


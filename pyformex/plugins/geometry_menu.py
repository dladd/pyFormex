# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
from __future__ import print_function

import pyformex as pf
import utils
from odict import ODict

import fileread
from geometry import Geometry
from geomfile import GeometryFile
from formex import Formex
from connectivity import Connectivity
from mesh import Mesh,mergeMeshes
from trisurface import TriSurface,Sphere
from elements import elementType
import simple

from gui import actors
from gui import menu
from gui.draw import *

from plugins import objects,trisurface,inertia,partition,sectionize,dxf,tetgen,surface_menu

import commands, os, timer

_name_ = 'geometry_menu'

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


def geomList():
    """Return a list with all the currently displayed geometry actors"""
    return selection.check()
            
 
def set_selection(clas='geometry'):
    sel = pf.GUI.selection.get(clas)
    if sel:
        res = sel.ask()
        if res is None:
            warning('Nothing to select')
            return

        if not sel.names:
            message("Nothing selected")
            
        selection.set(sel.names)
        selection.draw()


def shrink():
    """Toggle the shrink mode"""
    if selection.shrink is None:
        selection.shrink = 0.8
    else:
        selection.shrink = None
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
        
    print("Reading file of type %s" % filetype)

    if filetype == 'pgf' or filetype == 'pgf.gz':
        res = GeometryFile(filename).read()

    elif filetype in ['surface','stl','off','gts','neu']:
        surf = TriSurface.read(filename)
        name = autoName(TriSurface).next()
        res = {name:surf}

    elif filetype in tetgen.filetypes:
        res = tetgen.readTetgen(filename)

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
    elif type(ftype) is list:
        pass
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
        print("Items read: %s" % [ "%s(%s)" % (k,res[k].__class__.__name__) for k in res])
        if select:
            selection.set(res.keys())
            print(selection.names)
            surface_menu.selection.set([n for n in selection.names if isinstance(named(n),TriSurface)])
            print(surface_menu.selection.names)

            if draw:
                selection.draw()
                zoomAll()


def importPgf():
    importGeometry(ftype='pgf')

def importSurface():
    importGeometry(ftype=['surface','pgf','all'])

def importTetgen():
    importGeometry(ftype='tetgen')

def importAny():
    importGeometry(ftype=None)


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
        
    pf.GUI.setBusy(True)
    for f in fn:
        d = fileread.readMeshFile(f)
        modelname = os.path.basename(f).replace('.mesh','')
        export({modelname:d})
        M = fileread.extractMeshes(d)
        names = [ "%s-%d"%(modelname,i) for i in range(len(M)) ]
        export(dict(zip(names,M)))
    pf.GUI.setBusy(False)


def readInp(fn=None):
    """Read an Abaqus .inp file and convert to pyFormex .mesh.

    """
    if fn is None:
        fn = askFilename(".","*.inp",multi=True)
        if not fn:
            return

        pf.GUI.setBusy(True)
        for f in fn:
            fileread.convertInp(f)
        pf.GUI.setBusy(False)
        return
        

def writeGeometry(obj,filename,filetype=None,shortlines=False):
    """Write the geometry items in objdict to the specified file.

    """
    if filetype is None:
        filetype = utils.fileTypeFromExt(filename)
        
    print("Writing file of type %s" % filetype)

    if filetype in [ 'pgf', 'pgf.gz', 'pyf' ]:
        # Can write anything
        if filetype in [ 'pgf', 'pgf.gz' ]:
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

    If a suffix is given, the Meshes are stored with names equal to the
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

    print("CONVERTING %s" % names)
    meshes =  dict([ (n,o.toMesh()) for n,o in zip(names,objects) if hasattr(o,'toMesh')])
    print("Converted %s" % meshes.keys())
    export(meshes)

    selection.set(meshes.keys())
    

def toSurface(suffix=''):
    """Transform the selected Geometry objects to TriSurfaces.

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

    ok = [ o.nplex()==3 for o in objects ]
    print(ok)
    if not all(ok):
        warning("Only objects with plexitude 3 can be converted to TriSurface. I can not convert the following objects: %s" % [ n for i,n in zip(ok,names) if not i ])
        return
    
    if suffix:
        names = [ n + suffix for n in names ]
    
    print("CONVERTING %s" % names)
    surfaces =  dict([ (n,TriSurface(o)) for n,o in zip(names,objects)])
    print("Converted %s" % surfaces.keys())
    export(surfaces)

    selection.set(surfaces.keys())


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


def convert_Mesh_TriSurface(F,totype):
    if totype == 'Formex':
        return F.toFormex()
    else:
        return globals()[totype](F)


base_patterns = [
    'l:1',
    'l:2',
    'l:12',
    'l:127',
    ]    

def createGrid():
    _data_ = _name_+'createGrid_data'
    dia = Dialog(
        items = [
            _I('name','__auto__'),
            _I('object type',choices=['Formex','Mesh','TriSurface']),
            _I('base',choices=base_patterns),
            _I('nx',4),
            _I('ny',2),
            _I('stepx',1.),
            _I('stepy',1.),
            _I('taper',0),
            _I('bias',0.),
             ]
        )
    if _data_ in pf.PF:
        dia.updateData(pf.PF[_data_])
    res = dia.getResults()
    if res:
        pf.PF[_data_] = res
        name = res['name']
        if name == '__auto__':
            name = autoName(res['object type']).next()
        F = Formex(res['base']).replic2(
            n1=res['nx'], n2=res['ny'],
            t1=res['stepx'], t2=res['stepy'],
            bias=res['bias'], taper=res['taper'])
        F = convertFormex(F,res['object type'])
        export({name:F})
        selection.set([name])
        if res['object type'] == 'TriSurface':
            surface_menu.selection.set([name])
        selection.draw()


def createRectangle():
    _data_ = _name_+'createRectangle_data'
    dia = Dialog(
        items = [
            _I('name','__auto__'),
            _I('object type',choices=['Formex','Mesh','TriSurface']),
            _I('nx',1),
            _I('ny',1),
            _I('width',1.),
            _I('height',1.),
            _I('bias',0.),
            _I('diag','up',choices=['none','up','down','x-both']),
             ]
        )
    if _data_ in pf.PF:
        dia.updateData(pf.PF[_data_])
    res = dia.getResults()
    if res:
        pf.PF[_data_] = res
        name = res['name']
        if name == '__auto__':
            name = autoName(res['object type']).next()
        F = simple.rectangle(
            nx=res['nx'],ny=res['ny'],
            b=res['width'],h=res['height'],
            bias=res['bias'],diag=res['diag'][0])
        F = convertFormex(F,res['object type'])
        export({name:F})
        selection.set([name])
        if res['object type'] == 'TriSurface':
            surface_menu.selection.set([name])
        selection.draw()
    

def createCylinder():
    _data_ = _name_+'createCylinder_data'
    dia = Dialog(items=[
            _I('name','__auto__'),
            _I('object type',choices=['Formex','Mesh','TriSurface']),
            _I('base diameter',1.),
            _I('top diameter',1.),
            _I('height',2.),
            _I('angle',360.),
            _I('div_along_length',6),
            _I('div_along_circ',12),
            _I('bias',0.),
            _I('diag','up',choices=['none','up','down','x-both']),
            ],
        )
    if _data_ in pf.PF:
        dia.updateData(pf.PF[_data_])
    res = dia.getResults()
    if res:
        pf.PF[_data_] = res
        name = res['name']
        if name == '__auto__':
            name = autoName(res['object type']).next()

        F = simple.cylinder(
            L=res['height'],
            D=res['base diameter'],
            D1=res['top diameter'],
            angle=res['angle'],
            nt=res['div_along_circ'],
            nl=res['div_along_length'],
            bias=res['bias'],
            diag=res['diag'][0]
            )

        F = convertFormex(F,res['object type'])
        export({name:F})
        selection.set([name])
        if res['object type'] == 'TriSurface':
            surface_menu.selection.set([name])
        selection.draw()


def createCone():
    _data_ = _name_+'createCone_data'
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
    if _data_ in pf.PF:
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
        if res['object type'] == 'TriSurface':
            surface_menu.selection.set([name])
        selection.draw()


def createSphere():
    _data_ = _name_+'createSphere_data'
    dia = Dialog(
        items = [
            _I('name','__auto__'),
            _I('object type',itemtype='radio',choices=['TriSurface','Mesh','Formex']),
            _I('method',choices=['icosa','geo']),
            _I('ndiv',8),
            _I('nx',36),
            _I('ny',18),
             ],
        enablers=[
            ('method','icosa','ndiv'),
            ('method','geo','nx','ny'),
            ],
        )
    if _data_ in pf.PF:
        dia.updateData(pf.PF[_data_])
    res = dia.getResults()
    if res:
        pf.PF[_data_] = res
        name = res['name']
        if name == '__auto__':
            name = autoName(res['object type']).next()
        if res['method'] == 'icosa':
            F = simple.sphere(res['ndiv'])
            print("Surface has %s vertices and %s faces" % (F.ncoords(),F.nelems()))
            F = convert_Mesh_TriSurface(F,res['object type'])
        else:
            F = simple.sphere3(res['nx'],res['ny'])
            F = convertFormex(F,res['object type'])
            print("Surface has  %s faces" % F.nelems())
        export({name:F})
        selection.set([name])
        if res['object type'] == 'TriSurface':
            surface_menu.selection.set([name])
        selection.draw()
        

#############################################
###  Transformations
#############################################


def showPrincipal():
    """Show the principal axes."""
    F = selection.check(single=True)
    if not F:
        return
    # compute the axes
    data = F.inertia()
    C,Iaxes,Iprin,I = data
    pf.message("Center of gravity: %s" % C)
    pf.message("Principal Directions:\n %s" % Iaxes)
    pf.message("Principal Values: %s" % Iprin)
    pf.message("Inertia tensor: %s" % I)
    # display the axes
    CS = CoordinateSystem(origin=C,axes=Iaxes.transpose())
    size = 0.6*F.dsize()
    drawAxes(CS,size=size,psize=0.1*size)
    export({'_principal_data_':data})
    return data


def rotatePrincipal():
    """Rotate the selection according to the last shown principal axes."""
    try:
        data = named('_principal_data_')
    except:
        data = showPrincipal() 
    FL = selection.check()
    if FL:
        ctr,rot = data[:2]
        selection.changeValues([ F.trl(-ctr).rot(rot).trl(ctr) for F in FL ])
        selection.drawChanges()


def transformPrincipal():
    """Transform the selection according to the last shown principal axes.

    This is analog to rotatePrincipal, but positions the object at its center.
    """
    try:
        data = named('_principal_data_')
    except:
        data = showPrincipal() 
    FL = selection.check()
    if FL:
        ctr,rot = data[:2]
        selection.changeValues([ F.trl(-ctr).rot(rot) for F in FL ])
        selection.drawChanges()


#############################################
###  Mesh functions
#############################################
       
def narrow_selection(clas):
    global selection
    print("SELECTION ALL TYPES",selection.names)
    selection.set([n for n in selection.names if isinstance(named(n),clas)])
    print("SELECTION MESH TYPE",selection.names)
    

def reverseMesh():
    """Fuse the nodes of a Mesh"""
    if not selection.check():
        selection.ask()

    narrow_selection(Mesh)

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    meshes = [ m.reverse() for m in meshes ]
    export2(selection.names,meshes)
    clear()
    selection.draw()
    

def doOnSelectedMeshes(method):
    """Apply some method to all selected meshes"""
    if not selection.check():
        selection.ask()

    narrow_selection(Mesh)

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    meshes = [ method(m) for m in meshes ]
    export2(selection.names,meshes)
    clear()
    selection.draw()


def removeDegenerate():
    doOnSelectedMeshes(Mesh.removeDegenerate)
    

def compactMesh():
    """Compact the Mesh"""
    if not selection.check():
        selection.ask()

    narrow_selection(Mesh)

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    meshes = [ m.compact() for m in meshes ]
    export2(selection.names,meshes)
    clear()
    selection.draw()
    

def peelOffMesh():
    """Peel the Mesh"""
    if not selection.check():
        selection.ask()

    narrow_selection(Mesh)

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    meshes = [ m.peel() for m in meshes ]
    export2(selection.names,meshes)
    clear()
    selection.draw()
    

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
        _I('Points per box',1)])

    if not res:
        return

    before = [ m.ncoords() for m in meshes ]
    meshes = [ m.fuse(
        rtol = res['Relative Tolerance'],
        atol = res['Absolute Tolerance'],
        shift = res['Shift'],
        ppb = res['Points per box'],
        ) for m in meshes ]
    after = [ m.ncoords() for m in meshes ]
    print("Number of points before fusing: %s" % before)
    print("Number of points after fusing: %s" % after)

    names = [ "%s_fused" % n for n in selection.names ]
    export2(names,meshes)
    selection.set(names)
    clear()
    selection.draw()



def subdivideMesh():
    """Create a mesh by subdividing existing elements.

    """
    if not selection.check():
        selection.ask()

    narrow_selection(Mesh)

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    eltypes = set([ m.eltype.name() for m in meshes if m.eltype is not None])
    print("eltypes in selected meshes: %s" % eltypes)
    if len(eltypes) > 1:
        warning("I can only subdivide meshes with the same element type\nPlease narrow your selection before trying conversion.")
        return

    oktypes = ['tri3','quad4']
    eltype = eltypes.pop()
    if eltype not in ['tri3','quad4']:
        warning("I can only subdivide meshes of types %s" % ', '.join(oktypes))
        return

    if eltype == 'tri3':
        items = [_I('ndiv',4)]
    elif eltype == 'quad4':
        items = [_I('nx',4),_I('ny',4)]
    res = askItems(items)

    if not res:
        return
    if eltype == 'tri3':
        ndiv = [ res['ndiv'] ]
    elif eltype == 'quad4':
        ndiv = [ res['nx'], res['ny'] ]
    meshes = [ m.subdivide(*ndiv) for m in meshes ]
    export2(selection.names,meshes)
    clear()
    selection.draw()


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
    print("eltypes in selected meshes: %s" % eltypes)
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
            print("Selected conversion %s" % _conversion)
            totype = _conversion.split()[-1]
            names = [ "%s_converted" % n for n in selection.names ]
            meshes = [ m.convert(totype) for m in meshes ]
            if _merge == 'Each':
                meshes = [ m.fuse() for m in meshes ]
            elif  _merge == 'All':
                print(_merge)
                coords,elems = mergeMeshes(meshes)
                print(elems)
                ## names = [ "_merged_mesh_%s" % e.nplex() for e in elems ]
                ## meshes = [ Mesh(coords,e,eltype=meshes[0].eltype) for e in elems ]
                ## print meshes[0].elems
                meshes = [ Mesh(coords,e,m.prop,m.eltype) for e,m in zip(elems,meshes) ]
            if _compact:
                print("compacting meshes")
                meshes = [ m.compact() for m in meshes ]
                
            export2(names,meshes)
            selection.set(names)
            clear()
            selection.draw()


def renumberMesh(order='elems'):
    """Renumber the nodes of the selected Meshes.

    """
    if not selection.check():
        selection.ask()

    narrow_selection(Mesh)

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    names = selection.names
    meshes = [ M.renumber(order) for M in meshes ]
    export2(names,meshes)
    selection.set(names)
    clear()
    selection.draw()


def renumberMeshRandom():
    """Renumber the nodes of the selected Meshes in random order. 

    """
    renumberMesh('random')


def renumberMeshFront():
    """Renumber the nodes of the selected Meshes in random order. 

    """
    renumberMesh('front')


def getBorderMesh():
    """Create the border Meshes for the selected Meshes.

    """
    if not selection.check():
        selection.ask()

    narrow_selection(Mesh)

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    names = selection.names
    meshes = [ M.getBorderMesh() for M in meshes ]

    names = [ "%s_border" % n for n in selection.names ]
    export2(names,meshes)
    selection.set(names)
    clear()
    selection.draw()


 
################### menu #################
 
_menu = 'Geometry'

def loadDxfMenu():
    pass

def create_menu():
    """Create the plugin menu."""
    from dxf_menu import importDxf
    _init_()
    MenuData = [
        ("&Import ",[
            (utils.fileDescription('pgf'),importPgf),
            (utils.fileDescription('surface'),importSurface),
            (utils.fileDescription('tetgen'),importTetgen),
            ("All known geometry formats",importAny),
            ("Abaqus .inp",[
                ("&Convert Abaqus .inp file",readInp),
                ("&Import Converted Abaqus Model",importModel),
                ]),
#            ("AutoCAD .dxf",[
#                ("&Import .dxf or .dxftext",importDxf),
#                ("&Load DXF plugin menu",loadDxfMenu),
#                ]),
            ('&Upgrade pyFormex Geometry File',convertGeometryFile,dict(tooltip="Convert a pyFormex Geometry File (.pgf) to the latest format, overwriting the file.")),
            ]),
        ("&Export ",[
            (utils.fileDescription('pgf'),exportPgf),
            ("pyFormex Geometry File with short lines",exportPgfShortlines),
            ("Object File Format (.off)",exportOff),
            ]),
        ("&Select ",[
            ('Any',set_selection),
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
        ("&Annotations ",[
            ("&Names",selection.toggleNames,dict(checkable=True)),
            ("&Elem Numbers",selection.toggleNumbers,dict(checkable=True)),
            ("&Node Numbers",selection.toggleNodeNumbers,dict(checkable=True,checked=selection.hasNodeNumbers())),
            ("&Free Edges",selection.toggleFreeEdges,dict(checkable=True,checked=selection.hasFreeEdges())),
            ("&Node Marks",selection.toggleNodes,dict(checkable=True,checked=selection.hasNodeMarks())),
            ('&Toggle Bbox',selection.toggleBbox,dict(checkable=True)),
            ('&Toggle Shrink Mode',shrink,dict(checkable=True)),
            ]),
        ("---",None),
        ("&Convert",[
            ("To &Formex",toFormex),
            ("To &Mesh",toMesh),
            ("To &TriSurface",toSurface),
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
            ('&Grid',createGrid),
            ('&Rectangle',createRectangle),
            ('&Cylinder, Cone, Truncated Cone',createCylinder),
            ('&Circle, Sector, Cone',createCone),
            ('&Sphere',createSphere),
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
        ("---",None),
        ("Show &Principal Axes",showPrincipal),
        ("Rotate to &Principal Axes",rotatePrincipal),
        ("Transform to &Principal Axes",transformPrincipal),
        ## ("---",None),
        ## ("&Concatenate Selection",concatenateSelection),
        ## ("&Partition Selection",partitionSelection),
        ## ("&Create Parts",createParts),
        ## ("&Sectionize Selection",sectionizeSelection),
        ## ("---",None),
        ## ("&Fly",fly),
        ("Mesh",[
            ("&Reverse mesh elements",reverseMesh),
            ("&Convert element type",convertMesh),
            ("&Subdivide",subdivideMesh),
            ("&Compact",compactMesh),
            ("&Fuse nodes",fuseMesh),
            ("&Remove degenerate",removeDegenerate),
            ("&Renumber nodes",[
                ("In element order",renumberMesh),
                ("In random order",renumberMeshRandom),
                ("In frontal order",renumberMeshFront),
                ]),
            ("&Get border mesh",getBorderMesh),
            ("&Peel off border",peelOffMesh),
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
    """Show the menu."""
    if not pf.GUI.menu.item(_menu):
        create_menu()


def close_menu():
    """Close the menu."""
    m = pf.GUI.menu.item(_menu)
    if m :
        m.remove()
      

def reload_menu():
    """Reload the menu."""
    from plugins import refresh
    close_menu()
    refresh(_menu)
    show_menu()


####################################################################
######### What to do when the script is executed ###################

def run():
    _init_()
    reload_menu()

if __name__ == 'draw':
    run()
    

# End


# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
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
"""FeEx

"""
from __future__ import print_function
_status = 'checked'
_level = 'advanced'
_topics = ['FEA']
_techniques = ['menu', 'dialog', 'persistence', 'color', 'isopar'] 

from gui.draw import *
from pyformex import GUI,PF
from gui import menu

from simple import rectangle
from plugins.fe import *
from plugins.properties import *
from plugins.fe_abq import *
from plugins.fe_post import FeResult
from plugins import postproc_menu
from plugins import geometry_menu
from plugins import isopar
from odict import ODict
import utils

# global data

parts = None
model = None
PDB = None

# check for existing results
feresult_base = 'FeResult'
feresults = [ k for k in pf.PF.keys() if k.startswith(feresult_base) ]
if feresults:
    name = utils.hsorted(feresults)[-1]
else:
    name = feresult_base
    
feresult_name = utils.NameSequence(name)


def resetData():
    global parts,model,PDB
    parts = PF.get('FeEx-parts',[])
    model = PF.get('FeEx-model',None)
    PDB = PF.get('FeEx-propdb',None)
    geometry_menu.selection.set([])


def saveData():
    export({'FeEx-parts':parts,'FeEx-model':model,'FeEx-propdb':PDB})
    
    
def reset():
    clear()
    smoothwire()
    transparent()
    #lights(False)
    geometry_menu.selection.draw()

def deleteAll():
    global parts,model,PDB
    parts = []
    model = None
    PDB = None
    geometry_menu.selection.set([])
    reset()

######################## parts ####################
    
x0,y0 = 0.,0.
x1,y1 = 1.,0.
x2,y2 = 1.,1.
x3,y3 = 0.,1.
nx,ny = 4,4
eltype = 'quad'

def createRectPart(res=None):
    """Create a rectangular domain from user input"""
    global x0,y0,x2,y2,nx,ny,eltype
    if model is not None:
        if ask('You have already merged the parts! I can not add new parts anymore.\nYou should first delete everything and recreate the parts.',['Delete','Cancel']) == 'Delete':
            deleteAll()
        else:
            return
    if res is None:
        res = askItems([
            _I('x0',x0,tooltip='The x-value of one of the corners'),
            _I('y0',y0),
            _I('x2',x2),_I('y2',y2),
            _I('nx',nx),_I('ny',ny),
            _I('eltype',eltype,itemtype='radio',choices=['quad','tri-u','tri-d']),
            ])
    if res:
        globals().update(res)
        if x0 > x2:
            x0,x2 = x2,x0
        if y0 > y2:
            y0,y2 = y2,y0
        diag = {'quad':'', 'tri-u':'u', 'tri-d':'d'}[eltype]
        M = rectangle(nx,ny,x2-x0,y2-y0,diag=diag).toMesh().trl([x0,y0,0])
        addPart(M)


def createQuadPart(res=None):
    """Create a quadrilateral domain from user input"""
    global x0,y0,x1,y1,x2,y2,x3,y3,nx,ny,eltype
    if model is not None:
        if ask('You have already merged the parts! I can not add new parts anymore.\nYou should first delete everything and recreate the parts.',['Delete','Cancel']) == 'Delete':
            deleteAll()
        else:
            return
    if res is None:
        res = askItems([
            _I('Vertex 0',(x0,y0)),
            _I('Vertex 1',(x1,y1)),
            _I('Vertex 2',(x2,y2)),
            _I('Vertex 3',(x3,y3)),
            _I('nx',nx),
            _I('ny',ny),
            _I('eltype',eltype,itemtype='radio',choices=['quad','tri-u','tri-d']),
            ])
    if res:
        x0,y0 = res['Vertex 0']
        x1,y1 = res['Vertex 1']
        x2,y2 = res['Vertex 2']
        x3,y3 = res['Vertex 3']
        nx = res['nx']
        ny = res['ny']
        eltype = res['eltype']
        diag = {'quad':'', 'tri-u':'u', 'tri-d':'d'}[eltype]
        xold = rectangle(1,1).coords
        xnew = Coords([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
        M = rectangle(nx,ny,1.,1.,diag=diag).toMesh().isopar('quad4',xnew,xold)
        addPart(M)


def addPart(M):
    """Add a Mesh to the parts list."""
    global parts
    n = len(parts)
    part = M.setProp(n)
    parts.append(part)
    partname = 'part-%s'%n
    export({partname:part})
    geometry_menu.selection.names.append(partname)
    geometry_menu.selection.draw()


def convertQuadratic(qtype='quad8'):
    """Convert the parts to quadratic"""
    global parts
    parts = [ p.convert(qtype) for p in parts ]
    geometry_menu.selection.changeValues(parts)
    drawParts()
    
def convertQuadratic9():
    """Convert the parts to quadratic9"""
    convertQuadratic(qtype='quad9')
    

def drawParts():
    """Draw all parts"""
    clear()
    ## draw(parts)
    ## [ drawNumbers(p,color=blue) for p in parts ]
    ## [ drawNumbers(p.coords,color=red) for p in parts ]
    ## zoomAll()
    geometry_menu.selection.draw()


######################## the model ####################
    

def createModel():
    """Merge all the parts into a Finite Element model."""
    global model,PDB
    model = Model(*mergeMeshes(parts))
    PDB = PropertyDB()
    export({'FeEx-parts':parts,'FeEx-model':model,'FeEx-propdb':PDB})
    drawModel()


def drawModel(offset=0):
    """Draw the merged parts"""
    if model is None:
        warning("You should first merge the parts!")
        return
    flatwire()
    transparent(True)
    clear()
    meshes =  [ Mesh(model.coords,e,eltype='quad4') for e in model.elems ]
    draw(meshes,color='yellow')
    #drawNumbers(Formex(model.coords),color=red,offset=offset)
    #[ drawNumbers(m,leader='%s-'%i) for i,m in enumerate(meshes) ]
    zoomAll()


def drawCalpy():
    """This should draw node numbers +1"""
    drawModel(offset=1)


def pickNodes():
    """Let user pick nodes and return node numbers.

    This relies on the model being merged and drawn, resulting in a
    single actor having point geometry. Thus we do not bother about the key.
    """
    K = pickPoints()
    for k,v in K.items():
        if len(v) > 0:
            return v
    return None


def getPickedElems(K,p):
    """Get the list of picked elems from part p."""
    if p in K.keys():
        return K[p]
    return []
        

def printModel():
    print("model:",model)


################# Add properties ######################

# Plane stress element types for Abaqus
abq_eltype = {
    'quad4': 'CPS4',
    'quad8': 'CPS8',
    'quad9': 'CPS9',
    }

def warn():
   warning("You should first merge the parts!")

material = ODict([
    ('name','steel'),
    ('young_modulus',207000),
    ('poisson_ratio',0.3),
    ('density',7.85e-9),
    ])
section = ODict([
    ('name','thin steel plate'),
    ('sectiontype','solid'),
    ('thickness',1.0),
    ('material','steel'),
    ])


def setMaterial():
    """Set the material"""
    global section,material
    if model is None:
        warn()
        return
    removeHighlight()
    hicolor('purple')
    res = askItems(autoprefix=True,items=[
        _G('Material',[ _I(k,material[k]) for k in [
            'name','young_modulus','poisson_ratio','density']]),
        _G('Section',[ _I(k,section[k]) for k in [
            'name','sectiontype','thickness']]),
        _I('reduced_integration',False),
        ])

    material = utils.subDict(res,'Material/')
    section = utils.subDict(res,'Section/')
    section['material'] = material['name']
    
    if res:
        K = pickElements()
        if K:
            for k in range(len(parts)):
                e = getPickedElems(K,k) + model.celems[k]
                eltype = abq_eltype[model.elems[k].eltype.name()]
                if res['reduced_integration']:
                    eltype += 'R'
                print(k,e)
                if len(e) > 0:
                    PDB.elemProp(set=e,eltype=eltype,section=ElemSection(section=section,material=material))

def deleteAllMats():
    PDB.delProp(kind='e',attr=['eltype'])


# Boundary conditions

xcon = True
ycon = True

def setBoundary():
    """Pick the points with boundary condition."""
    global PDB,xcon,ycon
    if model is None:
        warn()
        return
    removeHighlight()
    res = askItems([('x-constraint',xcon),('y-constraint',ycon)])
    if res:
        xcon = res['x-constraint']
        ycon = res['y-constraint']
        nodeset = pickNodes()
        if len(nodeset) > 0:
            print(nodeset)
            bcon = [int(xcon),int(ycon),0,0,0,0]
            print("SETTING BCON %s" % bcon)
            PDB.nodeProp(set=nodeset,bound=[xcon,ycon,0,0,0,0])

def deleteAllBcons():
    PDB.delProp(kind='n',attr=['bound'])



# Concentrated loads

xload = 0.0
yload = 0.0
nsteps = 1
step = 1

def setCLoad():
    """Pick the points with load condition."""
    global xload,yload,nsteps,step
    if model is None:
        warn()
        return
    removeHighlight()
    res = askItems([
        ('step',step),
        ('x-load',xload),
        ('y-load',yload)])
    if res:
        step = res['step']
        xload = res['x-load']
        yload = res['y-load']
        nodeset = pickNodes()
        if len(nodeset) > 0:
            print(nodeset)
            print("SETTING CLOAD %s" % [xload,yload,0.,0.,0.,0.])
            PDB.nodeProp(set=nodeset,tag="Step-%s"%step,cload=[xload,yload,0.,0.,0.,0.])
            nsteps = max(nsteps,step)


def deleteAllCLoads():
    PDB.delProp(kind='n',attr=['cload'])


# Edge loads

edge_load = {'x':0., 'y':0.}

def setELoad():
    """Pick the edges with load condition."""
    global edge_load
    if model is None:
        warn()
        return
    removeHighlight()
    edge_load = askItems([
        ('step',step),
        ('x',edge_load['x'],{'text':'x-load'}),
        ('y',edge_load['y'],{'text':'y-load'}),
        ])
    if edge_load:
        K = pickEdges()
        for k in K.keys():
            v = K[k]
            elems,edges = v // 4, v % 4
            print(k,elems,edges)
            for el,edg in zip(elems,edges):
                for label in 'xy':
                    if edge_load[label] != 0.:
                        PDB.elemProp(set=el,group=k,tag="Step-%s"%step,eload=EdgeLoad(edge=edg,label=label,value=edge_load[label]))
            nsteps = max(nsteps,step)

def deleteAllELoads():
    PDB.delProp(kind='e',attr=['eload'])


def printDB():
    print("\n*** Node properties:")
    for p in PDB.nprop:
        print(p)
    print("\n*** Element properties:")
    for p in PDB.eprop:
        print(p)


############################# Abaqus ##############################
def createAbaqusInput():
    """Write the Abaqus input file."""
    
    # ask job name from user
    fn = askNewFilename(filter='*.inp')
    if not fn:
        return

    if not fn.endswith('.inp'):
        fn += '.inp'

    print(nsteps)
    steps = [ Step(time=[1.,1.,0.01,1.],tags=['Step-%s'%i]) for i in range(1,nsteps+1) ]

    print(steps)

    proc = ask('Intended FEA processor:',['Abaqus','Calculix'])
    if proc == 'Abaqus':
        res = [ Result(kind='NODE',keys=['U','COORD']),
                Result(kind='ELEMENT',keys=['S'],pos='AVERAGED AT NODES'),
                Result(kind='ELEMENT',keys=['SINV'],pos='AVERAGED AT NODES'),
                Result(kind='ELEMENT',keys=['SF'],pos='AVERAGED AT NODES'),
                ]
    else:
        res = [ Result(kind='NODE',keys=['U','COORD']),
                Result(kind='ELEMENT',keys=['S']),
                Result(kind='ELEMENT',keys=['SINV']),
                Result(kind='ELEMENT',keys=['SF']),
                ]

    data = AbqData(model,prop=PDB,steps=steps,res=res)
    data.write(jobname=fn,group_by_group=True)

    if ack("Load the Abaqus input file %s in the editor?"% fn):
        editFile(fn)


############################# Calix ##############################
def createCalixInput():
    """Write the Calix input file."""

    checkWorkdir()

    if model is None:
        warn()
        return
    
    # ask job name from user
    res = askItems([
        _I('jobname',feresult_name.next(),text='Job Name'),
        _I('header','A Calix example',text='Header Text'),
        _I('zem','3',text='ZEM control',itemtype='radio',choices=['0','3','6'],),
        ])
    if not res:
        return

    jobname = res['jobname']
    header = res['header']
    nzem = int(res['zem'])
    if not jobname:
        print("No Job Name: writing to sys.stdout")
        jobname = None

    filnam = jobname+'.dta'
    print("Writing calix data file %s in %s" % (filnam,os.getcwd()))
    fil = open(filnam,'w')
    
    nnodes = model.coords.shape[0]
    nelems = model.celems[-1]
    nplex = [ e.shape[1] for e in model.elems ]
    if min(nplex) != max(nplex):
        print([ e.shape for e in model.elems ])
        warning("All parts should have same element type")
        return
    nodel = nplex[0]

    # Get materials
    matprops = PDB.getProp(kind='e',attr=['section']) 
    # E, nu, thickness, rho
    mats = array([[mat.young_modulus,
                   mat.poisson_ratio,
                   mat.thickness,
                   0.0,      # rho was not defined in material
                   ] for mat in matprops]) 
    matnr = zeros(nelems,dtype=int32)
    for i,mat in enumerate(matprops):  # proces in same order as above!
        matnr[mat.set] = i+1
    print(matnr)
    nmats = mats.shape[0]
    nloads = 0
    # Header
    fil.write("""; calix data file generated by %s
; jobname=%s
start: %s
;use cmdlog 'calix.log'
;use for messages cmdlog
file open 'femodel.tmp' write da 1
yes cmdlog
;yes debug
use for cmdlog output
femodel elast stress 2
use for cmdlog cmdlog
;-----------------------------------------
; Aantal knopen:   %s
; Aantal elementen:   %s
; Aantal materialen:     %s
; Aantal belastingsgevallen: %s
"""% (pf.Version,jobname,header,nnodes,nelems,nmats,nloads))
    # Nodal coordinates
    fil.write(""";-----------------------------------------
; Knopen
;--------
nodes coord %s 1
""" % nnodes)
    fil.write('\n'.join(["%5i%10.2f%10.2f"%(i,x[0],x[1]) for i,x in zip(arange(nnodes)+1,model.coords)]))
    fil.write('\n\n')
    # Boundary conditions
    fil.write(""";-----------------------------------------
; Verhinderde verplaatsingen
;-------------------------
bound bcon
plane
""")
    for p in PDB.getProp(kind='n',attr=['bound']):
        bnd = "%5i" + "%5i"*2 % (p.bound[0],p.bound[1])
        if p.set is None:
            nod = arange(model.nnodes)
        else:
            nod = array(p.set)
        fil.write('\n'.join([ bnd % i for i in nod+1 ]))
        fil.write('\n')
    fil.write('\n')
    fil.write("""print bcon 3
                           $$$$$$$$$$$$$$$$$$$$$$$
                           $$      D O F S      $$
                           $$$$$$$$$$$$$$$$$$$$$$$
""")
    
    # Materials
    fil.write(""";-----------------------------------------
; Materialen
;-----------
array mat    %s 4
""" % len(mats))
    fil.write('\n'.join([ "%.4e "*4 % tuple(m) for m in mats]))
    fil.write('\n\n')
    fil.write("""print mat 3
                           $$$$$$$$$$$$$$$$$$$$$$$
                           $$ M A T E R I A L S $$
                           $$$$$$$$$$$$$$$$$$$$$$$
""")
    
    # Elements
    for igrp,grp in enumerate(model.elems):
        nelems,nplex = grp.shape
        fil.write(""";-----------------------------------------
; Elementen
;----------
elements elems-%s matnr-%s  %s %s 1
""" % (igrp,igrp,nplex,nelems))
        fil.write('\n'.join(["%5i"*(nplex+2) % tuple([i,1]+e.tolist()) for i,e in zip(arange(nelems)+1,grp+1)]))
        fil.write('\n\n')
        fil.write("""plane plane-%s coord bcon elems-%s matnr-%s 2 2
""" % (igrp,igrp,igrp))
        
    #########################
    # Nodal Loads
    cloads = [ p for p in PDB.getProp('n',attr=['cload']) ]
    fil.write("""text 3 1
                           $$$$$$$$$$$$$$$$$$$$
                           $$  NODAL  LOADS  $$
                           $$$$$$$$$$$$$$$$$$$$
loads f bcon 1
""")
    if len(cloads) > 0:
        loadcase=1
        for p in cloads:
            if p.set is None:
                nodeset = range(calpyModel.nnodes)
            else:
                nodeset = p.set
            F = [0.0,0.0]
            for i,v in p.cload:
                if i in [0,1]:
                    F[i] = v
            fil.write(''.join(["%5i%5i%10.2f%10.2f\n" % (n+1,loadcase,F[0],F[1]) for n in nodeset]))
    fil.write('\n')
    
    #########################
    # Distributed loads
    eloads = [ p for p in PDB.getProp('e',attr=['eload']) ]
    if len(eloads) > 0:
        fil.write("""text 4 1
                           $$$$$$$$$$$$$$$$$$$$$$$$$$
                           $$  BOUNDARY  ELEMENTS  $$
                           $$$$$$$$$$$$$$$$$$$$$$$$$$
   elem  loadnr  localnodes
""")
        # get the data from database, group by group
        loadcase=1
        for igrp in range(len(model.elems)):
            geloads = [ p for p in eloads if p.group==igrp ]
            neloads = len(geloads)
            loaddata = []
            fil.write("array randen integer   %s 4 0 1\n" % neloads)
            i = 1
            for p in geloads:
                xload = yload = 0.
                if p.label == 'x':
                    xload = p.value
                elif p.label == 'y':
                    yload = p.value
                # Save the load data for later
                loaddata.append((i,loadcase,xload,yload))
                # Because of the way we constructed the database, the set will
                # contain only one element, but let's loop over it anyway in case
                # one day we make the storage more effective
                for e in p.set:
                    fil.write(("%5s"*4+"\n")%(e+1,i,p.edge+1,(p.edge+1)%4+1))
                i += 1
            fil.write("""print randen
tran randen tranden
boundary  rand-%s coord bcon elems-%s matnr-%s tranden 1
""" % ((igrp,)*3))
            fil.write("""text 3 1
                           $$$$$$$$$$$$$$$$$$$$$$$
                           $$  BOUNDARY  LOADS  $$
                           $$$$$$$$$$$$$$$$$$$$$$$
loadvec boundary rand-%s f 1
""" % igrp)
            for eload in loaddata:
                fil.write("%5s%5s%10s%10s\n" % eload)
            fil.write('\n')

    #########################
    # Print total load vector
    fil.write("""
print f 3
                           $$$$$$$$$$$$$$$$$$$$
                           $$  LOAD  VECTOR  $$
                           $$$$$$$$$$$$$$$$$$$$
;
""")
    # Assemble
    for igrp in range(len(model.elems)):
        fil.write("assemble plane-%s mat s 0 0 0 %s\n" % (igrp,nzem))

    # Solve and output
    fil.write(""";------------------------------------------------solve+output
flavia mesh '%s.flavia.msh' %s
flavia nodes coord
""" %  (jobname,nplex))
    for igrp in range(len(model.elems)):
        fil.write("flavia elems elems-%s matnr-%s %s\n" % (igrp,igrp,nplex))
    fil.write("flavia results '%s.flavia.res'\n" % jobname)
    fil.write("""
solbnd s f
delete s
text named 1
"Displacement" "Elastic Analysis"
text typed 1
Vector OnNodes
text names 1
"Stress" "Elastic Analysis"
text types 1
Matrix OnNodes
intvar set 1 1
loop 1
  displ f bcon displ $1 1
  tran displ disp
  flavia result named typed disp $1
""")
    for igrp in range(len(model.elems)):
        fil.write("""
  stress plane-%s mat f stresg $1 1
  gp2nod plane-%s stresg strese 0 1
  nodavg plane-%s elems-%s strese stren nval 1
  tran stren stre
  flavia result names types stre $1
""" % ((igrp,)*4))
    fil.write("""       
  intvar add 1 1
next
stop
""")

    # Done: Close data file
    fil.close()
    showFile(filnam,mono=True)

    if ack("Shall I run the Calix analysis?"):
        # Run the analysis
        outfile = utils.changeExt(filnam,'res')
        cmd = "calix %s %s" % (filnam,outfile)
        utils.runCommand(cmd)
        showFile(outfile,mono=True)
        
        if ack("Shall I read the results for postprocessing?"):
            from plugins import flavia
            meshfile = utils.changeExt(filnam,'flavia.msh')
            resfile = utils.changeExt(filnam,'flavia.res')
            DB = flavia.readFlavia(meshfile,resfile)
            postproc_menu.setDB(DB)
            export({name:DB})
            if showInfo("The results have been exported as %s\nYou can now use the postproc menu to display results" % name,actions=['Cancel','OK']) == 'OK':
                postproc_menu.selection.set(name)
                postproc_menu.selectDB(DB)
                postproc_menu.open_dialog()
    

##############################################################################

def runCalpyAnalysis(jobname=None,verbose=False,flavia=False):
    """Create data for Calpy analysis module and run Calpy on the data.

    While we could write an analysis file in the Calpy format and then
    run the Calpy program on it (like we did with Abaqus), we can (and do)
    take another road here: Calpy has a Python/numpy interface, allowing
    us to directly present the numerical data in arrays to the analysis
    module.
    It is supposed that the Finite Element model has been created and
    exported under the name 'fe_model'.
    """

    ############################
    # Load the needed calpy modules    
    from plugins import calpy_itf
    calpy_itf.check()

    ## # Load development version
    #import sys
    #sys.path.insert(0,'/home/bene/prj/calpy')
    #print sys.path

    import calpy
    reload(calpy)
    print(calpy.__path__)
    
    calpy.options.optimize=True
    from calpy import femodel,fe_util,plane
    from calpy.arrayprint import aprint
    ############################

    checkWorkdir()

    if model is None:
        warn()
        return

    if jobname is None:
        # ask job name from user
        res = askItems([('JobName',feresult_name.peek()),('Verbose Mode',False)])
        if res:
            jobname = res['JobName']
            verbose = res['Verbose Mode']

    if not jobname:
        print("No Job Name: bailing out")
        return
   
    # OK, start calpy
    print("Starting the Calpy analysis module --- this might take some time")
    pf.app.processEvents()
    starttime = time.clock()

    calpyModel = femodel.FeModel(2,"elast","Plane_Stress")
    calpyModel.nnodes = model.coords.shape[0]
    calpyModel.nelems = model.celems[-1]
    print([ e.shape for e in model.elems ])
    nplex = [ e.shape[1] for e in model.elems ]
    if min(nplex) != max(nplex):
        warning("All parts should have same element type")
        return
    
    calpyModel.nnodel = nplex[0]

    # 2D model in calpy needs 2D coordinates
    coords = model.coords[:,:2]
    if verbose:
        fe_util.PrintNodes(coords)

    # Boundary conditions
    bcon = zeros((calpyModel.nnodes,2),dtype=int32)
    bcon[:,2:6] = 1 # leave only ux and uy
    for p in PDB.getProp(kind='n',attr=['bound']):
        bnd = where(p.bound)[0]
        if p.set is None:
            nod = arange(calpyModel.nnodes)
        else:
            nod = array(p.set)
        for i in bnd:
            bcon[p.set,i] = 1
    fe_util.NumberEquations(bcon)
    if verbose:
        fe_util.PrintDofs(bcon,header=['ux','uy'])

    # The number of free DOFs remaining
    calpyModel.ndof = bcon.max()
    print("Number of DOF's: %s" % calpyModel.ndof)

    
    # We extract the materials/sections from the property database
    matprops = PDB.getProp(kind='e',attr=['section'])
    
    # E, nu, thickness, rho
    mats = array([[mat.young_modulus,
                   mat.poisson_ratio,
                   mat.thickness,
                   0.0,      # rho was not defined in material
                   ] for mat in matprops]) 
    calpyModel.nmats = mats.shape[0]
    if verbose:
        fe_util.PrintMats(mats,header=['E','nu','thick','rho'])

   
    ########### Find number of load cases ############
    calpyModel.nloads = nloads
    calpyModel.PrintModelData()
    ngp = 2
    nzem = 3
    calpyModel.banded = True


    # Create element definitions:
    # In calpy, each element is represented by nnod + 1 integer numbers:
    #    matnr,  node1, node2, .... 
    # Also notice that Calpy numbering starts at 1, not at 0 as customary
    # in pyFormex; therefore we add 1 to elems.
    matnr = zeros(calpyModel.nelems,dtype=int32)
    for i,mat in enumerate(matprops):  # proces in same order as above!
        matnr[mat.set] = i+1

    NodesGrp = []
    MatnrGrp = []
    PlaneGrp = []

    for i,e in enumerate(model.elems):
        j,k = model.celems[i:i+2]
        Plane = plane.Quad("part-%s" % i,[ngp,ngp],calpyModel)
        Plane.debug = 0
        PlaneGrp.append(Plane)
        NodesGrp.append(e+1)
        MatnrGrp.append(matnr[j:k])
        if verbose:
            fe_util.PrintElements(NodesGrp[-1],MatnrGrp[-1])

        #Plane.AddElements(e+1,matnr[j:k],mats,coords,bcon)
        
    for Plane,nodenrs,matnrs in zip(PlaneGrp,NodesGrp,MatnrGrp):
        Plane.AddElements(nodenrs,matnrs,mats,coords,bcon)

    # Create load vectors
    # Calpy allows for multiple load cases in a single analysis.
    # However, our script currently puts all loads together in a single
    # load case. So the processing hereafter is rather simple, especially
    # since Calpy provides a function to assemble a single concentrated
    # load into the load vector. We initialize the load vector to zeros
    # and then add all the concentrated loads from the properties database.
    # A single concentrated load consists of 6 components, corresponding
    # to the 6 DOFs of a node.
    #
    # AssembleVector takes 3 arguments: the global vector in which to
    # assemble a nodal vector (length ndof), the nodal vector values
    # (length 6), and a list of indices specifying the positions of the
    # nodal DOFs in the global vector.
    # Beware: The function does not change the global vector, but merely
    # returns the value after assembling.
    # Also notice that the indexing inside the bcon array uses numpy
    # convention (starting at 0), thus no adding 1 is needed!
    print("Assembling Concentrated Loads")
    f = zeros((calpyModel.ndof,calpyModel.nloads),float)
    for p in PDB.getProp('n',attr=['cload']):
        lc = 0
        #lc = int(p.tag)
        if p.set is None:
            nodeset = range(calpyModel.nnodes)
        else:
            nodeset = p.set
        F = [0.0,0.0]
        for i,v in p.cload:
            if i in [0,1]:
                F[i] = v
        for n in nodeset:
            f[:,lc] = fe_util.AssembleVector(f[:,lc],F,bcon[n])

    print("Assembling distributed loads")
    # This is a bit more complex. See Calpy for details
    # We first generate the input data in a string, then read them with the
    # calpy femodel.ReadBoundaryLoads function and finally assemble them with
    # plane.addBoundaryLoads. We have to do this operation per element group.
    # The group number is stored in the property record.
    ngroups = model.ngroups()
    s = [ "" ] * ngroups
    nb = [ 0 ] * ngroups
    lc = 1
    for p in PDB.getProp('e',attr=['eload']):
        xload = yload = 0.
        if p.label == 'x':
            xload = p.value
        elif p.label == 'y':
            yload = p.value
        # Because of the way we constructed the database, the set will
        # contain only one element, but let's loop over it anyway in case
        # one day we make the storage more effective
        # Also, remember calpy numbers are +1 !
        g = p.group
        print("Group %s" % g)
        for e in p.set:
            s[g] += "%s %s %s %s %s\n" % (e+1,p.edge+1,lc,xload,yload)
            nb[g] += 1
    #print s,nb
    for nbi,si,nodes,matnr,Plane in zip(nb,s,NodesGrp,MatnrGrp,PlaneGrp):
        if nbi > 0:
            idloads,dloads = fe_util.ReadBoundaryLoads(nbi,calpyModel.ndim,si)
            #print idloads,dloads
            Plane.AddBoundaryLoads(f,calpyModel,idloads,dloads,nodes,matnr,coords,bcon,mats)
    
    if verbose:
        print("Calpy.Loads")
        print(f)

    ############ Create global stiffness matrix ##########
    s = calpyModel.ZeroStiffnessMatrix(0)
    for elgrp in PlaneGrp:
        s = elgrp.Assemble(s,mats,calpyModel)
    # print "The complete stiffness matrix"
    # print s

    ############ Solve the system of equations ##########
    v = calpyModel.SolveSystem(s,f)
    print("Calpy analysis has finished --- Runtime was %s seconds." % (time.clock()-starttime))
    displ = fe_util.selectDisplacements (v,bcon)
    if verbose:
        print("Displacements",displ)

    if flavia:
        flavia.WriteMeshFile(jobname,"Quadrilateral",model.nnodel,coord,nodes,matnr)
        res=flavia.ResultsFile(jobname)
        
    # compute stresses
    for lc in range(calpyModel.nloads):
        
        print("Results for load case %d" %(lc+1))
        print("Displacements")
        aprint(displ[:,:,lc],header=['x','y'],numbering=True)

        if flavia:
            flavia.WriteResultsHeader(res,'"Displacement" "Elastic Analysis"',lc+1,'Vector OnNodes')
            flavia.WriteResults(res,displ[:,:,lc])
            
        stresn = count = None
        i = 0
        for e,P in zip(model.elems,PlaneGrp):
            i += 1
            #P.debug = 1
            stresg = P.StressGP (v[:,lc],mats)
            if verbose:
                print("elem group %d" % i)
                print("GP Stress\n", stresg)
            
            strese = P.GP2Nodes(stresg)
            if verbose:
                print("Nodal Element Stress\n", strese)

            #print "Nodes",e+1
            stresn,count = P.NodalAcc(e+1,strese,nnod=calpyModel.nnodes,nodata=stresn,nodn=count)
            #print stresn,count
            
        #print stresn.shape
        #print count.shape
        #print "TOTAL",stresn,count
        stresn /= count.reshape(-1,1)
        #print "AVG",stresn
        if verbose:
            print("Averaged Nodal Stress\n")
            aprint(stresn,header=['sxx','syy','sxy'],numbering=True)
                
        if flavia:
            flavia.WriteResultsHeader(res,'"Stress" "Elastic Analysis"',lc+1,'Matrix OnNodes')
            flavia.WriteResults(res,stresn)

    
    DB = FeResult()
    DB.nodes = model.coords
    DB.nnodes = model.coords.shape[0]
    DB.nodid = arange(DB.nnodes)
    DB.elems = dict(enumerate(model.elems))
    DB.nelems = model.celems[-1]
    DB.Finalize()
    DB.datasize['S'] = 3
    #print DB.elems
    for lc in range(calpyModel.nloads):
        DB.Increment(lc,0)
        d = zeros((calpyModel.nnodes,3))
        d[:,:2] = displ[:,:,lc]
        DB.R['U'] = d
        DB.R['S'] = stresn
    postproc_menu.setDB(DB)
    name = feresult_name.next()
    export({name:DB})
    if showInfo("The results have been exported as %s\nYou can now use the postproc menu to display results" % name,actions=['Cancel','OK']) == 'OK':
        postproc_menu.selection.set(name)
        postproc_menu.selectDB(DB)
        postproc_menu.open_dialog()
    

def autoRun(quadratic=False):
    clear()
    if quadratic:
        nx,ny = 2,2
    else:
        nx,ny = 4,4
    createRectPart(dict(x0=0.,x2=1.,y0=0.,y2=1.,nx=nx,ny=ny,eltype='quad'))
    createRectPart(dict(x0=0.,x2=-1.,y0=0.,y2=1.,nx=nx,ny=ny,eltype='quad'))
    if quadratic:
        convertQuadratic()
    createModel()
    nodenrs = arange(model.coords.shape[0])
    PDB.elemProp(eltype='CPS4',section=ElemSection(section=section))
    if quadratic:
        ny *= 2
    PDB.nodeProp(set=nodenrs[:ny+1],bound=[1,1,0,0,0,0])
    PDB.nodeProp(set=nodenrs[-(ny+1):],cload=[10.,0.,0.,0.,0.,0.])
    runCalpyAnalysis('FeEx',verbose=True)

def autoRun2():
    clear()
    nx,ny = 1,1
    createRectPart(dict(x0=0.,x2=1.,y0=0.,y2=1.,nx=nx,ny=ny,eltype='quad'))
    convertQuadratic()
    createModel()
    nodenrs = arange(model.coords.shape[0])
    xmin,xmax = model.coords.bbox()[:,0]
    xtol = (xmax-xmin) / 1000.
    left = model.coords.test(dir=0,min=xmin-xtol,max=xmin+xtol)
    right = model.coords.test(dir=0,min=xmax-xtol,max=xmax+xtol)
    leftnrs = where(left)[0]
    rightnrs = where(right)[0]
    print(leftnrs)
    print(rightnrs)
    
    PDB.elemProp(eltype='CPS4',section=ElemSection(section=section))
    ny *= 2
    PDB.nodeProp(set=leftnrs,bound=[1,1,0,0,0,0])
    PDB.nodeProp(set=rightnrs,cload=[10.,0.,0.,0.,0.,0.])

    print("This example is incomplete.")
    print(PDB)
    #runCalpyAnalysis('FeEx',verbose=True)

def autoConv():
    clear()
    res = askItems([('nx',1),('ny',1)])
    nx = res['nx']
    ny = res['ny']
    createRectPart(dict(x0=0.,x1=10.,y0=0.,y1=1.,nx=nx,ny=ny,eltype='quad'))
    createModel()
    nodenrs = arange(model.coords.shape[0])
    PDB.elemProp(eltype='CPS4',section=ElemSection(section=section))
    PDB.nodeProp(set=nodenrs[:ny+1],bound=[1,1,0,0,0,0])
    PDB.nodeProp(set=nodenrs[-(ny+1):],cload=[0.,1./(ny+1),0.,0.,0.,0.])
    runCalpyAnalysis('FeEx',verbose=True)


def importAll():
    globals().update(pf.PF)

def exportAll():
    pf.PF.update(globals())
                 
#############################################################################
######### Create a menu with interactive tasks #############

def create_menu():
    """Create the FeEx menu."""
    MenuData = [
        ("&Delete All",deleteAll),
        ("&Create Rectangular Part",createRectPart),
        ("&Create QuadrilateralPart",createQuadPart),
        ("&Convert to Quadratic-8",convertQuadratic),
        ("&Convert to Quadratic-9",convertQuadratic9),
        ("&Show All",drawParts),
        ("---",None),
        ("&Merge Parts into Model",createModel),
        ("&Show Merged Model",drawModel),
        ("&Show Calpy Numbers",drawCalpy),
        ("&Print model",printModel),
        ("---",None),
        ("&Add material properties",setMaterial),
        ("&Add boundary conditions",setBoundary),
        ("&Add concentrated loads",setCLoad),
        ("&Add edge loads",setELoad),
        ("&Delete all material properties",deleteAllMats),
        ("&Delete all boundary conditions",deleteAllBcons),
        ("&Delete all concentrated loads",deleteAllCLoads),
        ("&Delete all edge loads",deleteAllELoads),
        ("&Print property database",printDB),
        ("---",None),
        ("&Create Abaqus/Calculix input file",createAbaqusInput),
        ("&Create Calix input file",createCalixInput),
        ("&Run Calpy analysis",runCalpyAnalysis),
        ("---",None),
        ("&Import all",importAll),
        ("&Export all",exportAll),
        ("&Autorun example",autoRun),
        ("&Autorun quadratic example",autoRun2),
        ("&Autoconv example",autoConv),
        ("---",None),
        ("&Close Menu",close_menu),
        ]
    return menu.Menu('FeEx',items=MenuData,parent=pf.GUI.menu,before='help')

 
def show_menu():
    """Show the menu."""
    if not pf.GUI.menu.item('FeEx'):
        create_menu()

def close_menu():
    """Close the menu."""
    m = pf.GUI.menu.item('FeEx')
    if m :
        m.remove()

def reload_menu():
    """Reload the menu."""
    close_menu()
    show_menu()


####################################################################
######### What to do when the script is executed ###################

def run():
    geometry_menu.show_menu()
    resetData()
    reset()
    reload_menu()
    
if __name__ == 'draw':
    run()
# End


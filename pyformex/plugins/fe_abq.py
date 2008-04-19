#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.7 Release Fri Apr  4 18:41:11 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""A number of functions to write an Abaqus input file.

There are low level functions that just generate a part of an Abaqus
input file, conforming to the Keywords manual.

Then there are higher level functions that read data from the property module
and write them to the Abaqus input file.
"""

from plugins.properties import *
from mydict import *
import globaldata as GD
import datetime
import os,sys


##################################################
## Some Abaqus .inp format output routines
##################################################

def abqInputNames(job):
    """Returns corresponding Abq jobname and input filename.

    job can be either a jobname or input file name, with or without
    directory part, with or without extension (.inp)
    
    The Abq jobname is the basename without the extension.
    The abq filename is the abspath of the job with extension '.inp'
    """
    jobname = os.path.basename(job)
    if jobname.endswith('.inp'):
        jobname = jobname[:-4]
    filename = os.path.abspath(job)
    if not filename.endswith('.inp'):
        filename += '.inp'
    return jobname,filename


def nsetName(p):
    """Determine the setname for writing a node property."""
    if p.set is None:
        setname = 'Nall'
    elif type(p.set) is str:
        setname = p.set
    else:
        setname = Nset(p.nr)
    return setname


def esetName(p):
    """Determine the setname for writing a elem property."""
    if p.set is None:
        setname = 'Eall'
    elif type(p.set) is str:
        setname = p.set
    else:
        setname = Eset(p.nr)
    return setname


def writeHeading(fil, text=''):
    """Write the heading of the Abaqus input file."""
    head = """**  Abaqus input file created by pyFormex (c) B.Verhegghe
**  (see http://pyformex.berlios.de)
**
*HEADING
%s
""" % text
    fil.write(head)


def writeNodes(fil,nodes,name='Nall',nofs=1):
    """Write nodal coordinates.

    The nodes are added to the named node set. 
    If a name different from 'Nall' is specified, the nodes will also
    be added to a set named 'Nall'.
    The nofs specifies an offset for the node numbers.
    The default is 1, because Abaqus numbering starts at 1.  
    """
    fil.write('*NODE, NSET=%s\n' % name)
    for i,n in enumerate(nodes):
        fil.write("%d, %14.6e, %14.6e, %14.6e\n" % ((i+nofs,)+tuple(n)))
    if name != 'Nall':
        fil.write('*NSET, NSET=Nall\n%s\n' % name)


def writeElems(fil,elems,type,name='Eall',eid=None,eofs=1,nofs=1):
    """Write element group of given type.

    elems is the list with the element node numbers.
    The elements are added to the named element set. 
    If a name different from 'Eall' is specified, the elements will also
    be added to a set named 'Eall'.
    The eofs and nofs specify offsets for element and node numbers.
    The default is 1, because Abaqus numbering starts at 1.
    If eid is specified, it contains the element numbers increased with eofs.
    """
    fil.write('*ELEMENT, TYPE=%s, ELSET=%s\n' % (type.upper(),name))
    nn = elems.shape[1]
    fmt = '%d' + nn*', %d' + '\n'
    if eid is None:
        eid = arange(elems.shape[0])
    else:
        eid = asarray(eid)
    for i,e in zip(eid+eofs,elems+nofs):
        fil.write(fmt % ((i,)+tuple(e)))
    writeSubset(fil,'ELSET','Eall',name)


def writeSet(fil,type,name,set,ofs=1):
    """Write a named set of nodes or elements (type=NSET|ELSET)"""
    fil.write("*%s,%s=%s\n" % (type,type,name))
    for i in asarray(set)+ofs:
        fil.write("%d,\n" % i)


def writeSubset(fil, type, name, subname):
    """Make a named set a subset of another one (type=NSET|ELSET)"""
    fil.write('*%s, %s=%s\n%s\n' % (type,type,name,subname))


def writeFrameSection(fil,elset,A,I11,I12,I22,J,E,G,
                      rho=None,orient=None):
    """Write a general frame section for the named element set.

    The specified values are:
      A: cross section
      I11: moment of inertia around the 1 axis
      I22: moment of inertia around the 2 axis
      I12: inertia product around the 1-2 axes
      J: Torsional constant
      E: Young's modulus of the material
      G: Shear modulus of the material
    Optional data:
      rho: density of the material
      orient: a vector specifying the direction cosines of the 1 axis
    """
    extra = orientation = ''
    if rho:
        extra = ',DENSITY=%s' % rho
    if orient:
        orientation = '%s %s %s' % (orient[0], orient[1], orient[2])
    fil.write("""*FRAME SECTION,ELSET=%s,SECTION=general%s
%s, %s, %s, %s, %s
%s
%s, %s
""" %(elset,extra,
      A,I11,I12,I22,J,
      orientation,
      E,G))


materialswritten=[]
def writeMaterial(fil,mat):
    """Write a material section.
    
    mat is the property dict of the material.
    If the material has a name and has already been written, this function
    does nothing.
    """
    if mat.name is not None and mat.name not in materialswritten:
        if mat.poisson_ratio is None and mat.shear_modulus is not None:
            mat.poisson_ratio = 0.5 * mat.young_modulus / mat.shear_modulus - 1.0
        fil.write("""*MATERIAL, NAME=%s
*ELASTIC
%s,%s
*DENSITY
%s
"""%(mat.name, float(mat.young_modulus), float(mat.poisson_ratio), float(mat.density)))
        materialswritten.append(mat.name)
        if mat.plastic is not None:
            fil.write('*PLASTIC\n')
            plastic=eval(mat['plastic'])
            for i in range(len(plastic)):
	      fil.write( '%s, %s\n' % (plastic[i][0],plastic[i][1]))
	if mat.damping == 'Yes':
		fil.write("*DAMPING")
		if mat.alpha != 'None':
			fil.write(", ALPHA = %s" %mat.alpha)
		if mat.beta != 'None':
			fil.write(", BETA = %s" %mat.beta)
		fil.write("\n")


def writeTransform(fil,setname,csys):
    """Write transform command for the given set.

    setname is the name of a node set
    csys is a CoordSystem.
    """
    fil.write("*TRANSFORM, NSET=%s, TYPE=%s\n" % (setname,csys.sys))
    fil.write("%s,%s,%s,%s,%s,%s\n" % tuple(csys.data.ravel()))


##################################################
## Some higher level functions, interfacing with the properties module
##################################################

plane_stress_elems = [
    'CPS3','CPS4','CPS4I','CPS4R','CPS6','CPS6M','CPS8','CPS8M']
plane_strain_elems = [
    'CPE3','CPE3H','CPE4','CPE4H','CPE4I','CPE4IH','CPE4R','CPE4RH',
    'CPE6','CPE6H','CPE6M','CPE6MH','CPE8','CPE8H','CPE8R','CPE8RH']
generalized_plane_strain_elems = [
    'CPEG3','CPEG3H','CPEG4','CPEG4H','CPEG4I','CPEG4IH','CPEG4R','CPEG4RH',
    'CPEG6','CPEG6H','CPEG6M','CPEG6MH','CPEG8','CPEG8H','CPEG8R','CPEG8RH']
solid2d_elems = plane_stress_elems + plane_strain_elems + generalized_plane_strain_elems

solid3d_elems = ['C3D4', 'C3D4H','C3D6', 'C3D6H', 'C3D8','C3D8H','C3D8R', 'C3D8RH','C3D10','C3D10H','C3D10M','C3D10MH','C3D15','C3D15H','C3D20','C3D20H','C3D20R','C3D20RH',]


def writeSection(fil,prop):
    """Write an element section.

    prop is a an element property record with a section and eltype attribute
    """
    setname = esetName(prop)
    el = prop.section
    eltype = prop.eltype

    mat = el.material
    if mat is not None:
        writeMaterial(fil,mat)

    ############
    ##FRAME elements
    ##########################
    if eltype.upper() in ['FRAME3D', 'FRAME2D']:
        if el.sectiontype.upper() == 'GENERAL':
            fil.write("""*FRAME SECTION, ELSET=%s, SECTION=GENERAL, DENSITY=%s
%s, %s, %s, %s, %s \n"""%(setname,float(el.density),float(el.cross_section),float(el.moment_inertia_11),float(el.moment_inertia_12),float(el.moment_inertia_22),float(el.torsional_rigidity)))
            if el.orientation != None:
                fil.write("""%s,%s,%s"""%(el.orientation[0],el.orientation[1],el.orientation[2]))
            fil.write("""\n %s, %s \n"""%(float(el.young_modulus),float(el.shear_modulus)))
        if el.sectiontype.upper() == 'CIRC':
            fil.write("""*FRAME SECTION, ELSET=%s, SECTION=CIRC, DENSITY=%s
%s \n"""%(setname,float(el.density),float(el.radius)))
            if el.orientation != None:
                fil.write("""%s,%s,%s"""%(el.orientation[0],el.orientation[1],el.orientation[2]))
            fil.write("""\n %s, %s \n"""%(float(el.young_modulus),float(el.shear_modulus)))

    ##############
    ##connector elements
    ##########################  
    elif eltype.upper() in ['CONN3D2', 'CONN2D2']:
        if el.sectiontype.upper() != 'GENERAL':
            fil.write("""*CONNECTOR SECTION,ELSET=%s
%s
""" %(setname,el.sectiontype.upper()))

    ############
    ##TRUSS elements
    ##########################  
    elif eltype.upper() in ['T2D2', 'T2D2H' , 'T2D3', 'T2D3H', 'T3D2', 'T3D2H', 'T3D3', 'T3D3H']:
        if el.sectiontype.upper() == 'GENERAL':
            fil.write("""*SOLID SECTION, ELSET=%s, MATERIAL=%s
%s
""" %(setname,el.material.name, float(el.cross_section)))
        elif el.sectiontype.upper() == 'CIRC':
            fil.write("""*SOLID SECTION, ELSET=%s, MATERIAL=%s
%s
""" %(setname,el.material.name, float(el.radius)**2*pi))

    ############
    ##BEAM elements
    ##########################
    elif eltype.upper() in ['B21', 'B21H','B22', 'B22H', 'B23','B23H','B31', 'B31H','B32','B32H','B33','B33H']:
        if el.sectiontype.upper() == 'GENERAL':
            fil.write("""*BEAM GENERAL SECTION, ELSET=%s, SECTION=GENERAL, DENSITY=%s
%s, %s, %s, %s, %s \n"""%(setname,float(el.density), float(el.cross_section),float(el.moment_inertia_11),float(el.moment_inertia_12),float(el.moment_inertia_22),float(el.torsional_rigidity)))
            if el.orientation != None:
                fil.write("%s,%s,%s"%(el.orientation[0],el.orientation[1],el.orientation[2]))
            fil.write("\n %s, %s \n"%(float(el.young_modulus),float(el.shear_modulus)))
        if el.sectiontype.upper() == 'CIRC':
            fil.write("""*BEAM GENERAL SECTION, ELSET=%s, SECTION=CIRC, DENSITY=%s
%s \n"""%(setname,float(el.density),float(el.radius)))
            if el.orientation != None:
                fil.write("""%s,%s,%s"""%(el.orientation[0],el.orientation[1],el.orientation[2]))
            fil.write("""\n %s, %s \n"""%(float(el.young_modulus),float(el.shear_modulus)))
	if el.sectiontype.upper() == 'RECT':
            fil.write('*BEAM SECTION, ELSET=%s, material=%s,\n** Section: %s  Profile: %s\ntemperature=GRADIENTS, SECTION=RECT\n %s,%s\n' %(setname,el.material.name,el.name,el.name,float(el.height),float(el.width)))
            if el.orientation != None:
                fil.write("""%s,%s,%s\n"""%(el.orientation[0],el.orientation[1],el.orientation[2]))
		

    ############
    ## SHELL elements
    ##########################
    elif eltype.upper() in ['STRI3', 'S3','S3R', 'S3RS', 'STRI65','S4','S4R', 'S4RS','S4RSW','S4R5','S8R','S8R5', 'S9R5',]:
        if el.sectiontype.upper() == 'SHELL':
            if mat is not None:
                fil.write("""*SHELL SECTION, ELSET=%s, MATERIAL=%s
%s \n""" % (setname,mat.name,float(el.thickness)))

    ############
    ## MEMBRANE elements
    ##########################
    elif eltype.upper() in ['M3D3', 'M3D4','M3D4R', 'M3D6', 'M3D8','M3D8R','M3D9', 'M3D9R',]:
        if el.sectiontype.upper() == 'MEMBRANE':
            if mat is not None:
                fil.write("""*MEMBRANE SECTION, ELSET=%s, MATERIAL=%s
%s \n""" % (setname,mat.name,float(el.thickness)))


    ############
    ## 3DSOLID elements
    ##########################
    elif eltype.upper() in solid3d_elems:
        if el.sectiontype.upper() == '3DSOLID':
            if mat is not None:
                fil.write("""*SOLID SECTION, ELSET=%s, MATERIAL=%s
%s \n""" % (setname,mat.name,1.))

    ############
    ## 2D SOLID elements
    ##########################
    elif eltype.upper() in solid2d_elems:
        if el.sectiontype.upper() == 'SOLID':
            if mat is not None:
                fil.write("""*SOLID SECTION, ELSET=%s, MATERIAL=%s
%s \n""" % (setname,mat.name,float(el.thickness)))
            
    ############
    ## RIGID elements
    ##########################
    elif eltype.upper() in ['R2D2','RB2D2','RB3D2','RAX2','R3D3','R3D4']:
        if el.sectiontype.upper() == 'RIGID':
            fil.write("""*RIGID BODY,REFNODE=%s,density=%s, ELSET=%s\n""" % (el.nodeset,el.density,setname))



    ############
    ## UNSUPPORTED elements
    ##########################
    else:
        warning('Sorry, elementtype %s is not yet supported' % eltype)

    
def writeBoundaries(fil,prop,op='MOD'):
    """Write nodal boundary conditions.

    prop is a list of node property records that should be scanned for
    bound attributes to write.

    By default, the boundary conditions are applied as a modification of the
    existing boundary conditions, i.e. initial conditions and conditions from
    previous steps remain in effect.
    The user can set op='NEW' to remove the previous conditions.
    This will also remove initial conditions!
    """
    for p in prop:
        setname = nsetName(p)
        fil.write("*BOUNDARY, OP=%s\n" % op)
        if isinstance(p.bound,str):
            fil.write("%s, %s\n" % (setname,p.bound))
        else:
            for b in range(6):
                if p.bound[b]==1:
                    fil.write("%s, %s\n" % (setname,b+1))


def writeDisplacements(fil,prop,op='MOD'):
    """Write boundary conditions of type BOUNDARY, TYPE=DISPLACEMENT

    prop is a list of node property records that should be scanned for
    displ attributes to write.
    
    By default, the boundary conditions are applied as a modification of the
    existing boundary conditions, i.e. initial conditions and conditions from
    previous steps remain in effect.
    The user can set op='NEW' to remove the previous conditions.
    This will also remove initial conditions!
    """
    for p in prop:
        setname = nsetName(p)
        fil.write("*BOUNDARY, TYPE=DISPLACEMENT, OP=%s" % op)
        if p.amplitude is not None:
            fil.write(", AMPLITUDE=%s" % p.amplitude)
        fil.write("\n")
        for d in range(len(p.displ)):
            fil.write("%s, %s, %s, %s\n" % (setname,p.displ[d][0],p.displ[d][0],p.displ[d][1]))

            
def writeCloads(fil,prop,op='NEW'):
    """Write cloads.

    prop is a list of node property records that should be scanned for
    displ attributes to write.

    By default, the loads are applied as new values in the current step.
    The user can set op='MOD' to add the loads to already existing ones.
    """
    fil.write("*CLOAD, OP=%s\n" % op)
    for p in prop:
        setname = nsetName(p)
        for i,l in enumerate(p.cload):
            if l != 0.0:
                fil.write("%s, %s, %s\n" % (setname,i+1,l))


def writeDloads(fil,prop,op='NEW'):
    """Write Dloads.
    
    prop is a list property records having an attribute dload

    By default, the loads are applied as new values in the current step.
    The user can set op='MOD' to add the loads to already existing ones.
    """
    for p in prop:
        setname = esetname(p)
        fil.write("*DLOAD, OP=%s" % op)
        if p.dload.amplitude is not None:
            fil.write(", AMPLITUDE=%s" % p.dload.amplitude)
            fil.write("\n")
        if p.dload.label == 'GRAV':
            fil.write("%s, GRAV, 9.81, 0, 0 ,-1\n" % setname)
        else:
            fil.write("%s, %s, %s\n" % (setname,p.dload.label,p.dload.value))
            

#######################################################
# General model data
#

## def writeAmplitude(fil,prop):
##     fil.write("*AMPLITUDE, NAME=%s\n" %name)
##     n = len(the_modelproperties[name].amplitude)
##     for i in range(n-1):
##         fil.write("%s, " % the_modelproperties[name].amplitude[i])
##     fil.write("%s\n" % the_modelproperties[name].amplitude[n-1])

# These are commented out because I do not really understand what
# the data are. 
    
## def writeInteraction(fil , name=None, op='NEW'):
##     if the_modelproperties[name].interactionname.upper()=='ALLWITHSELF':
##         fil.write('** INTERACTIONS, NAME=%s\n*Contact, op=%s\n*contact inclusions,ALL EXTERIOR\n*Contact property assignment\n ,  ,  %s\n'% (the_modelproperties[name].interactionname,op,the_modelproperties[name].interaction.intprop))
##     else:
##         fil.write('** INTERACTIONS, NAME=%s\n*Contact Pair, interaction=%s\n%s,%s\n'%(the_modelproperties[name].interactionname,the_modelproperties[name].interaction.intprop,the_modelproperties[name].interaction.surface1,the_modelproperties[name].interaction.surface2))

    
## def writeIntprop(fil,name=None):
##     fil.write("*Surface interaction, name=%s\n" %name)
##     fil.write("*%s\n%s,\n" %(the_modelproperties[name].intprop.inttype,the_modelproperties[name].intprop.parameter))
    
    
## def writeDamping(fil,name=None):
##     fil.write("*global damping")
##     if the_modelproperties[name].damping.field is not None:
##     	fil.write(", Field = %s"%the_modelproperties[name].damping.field)
##     else:
## 	fil.write(", Field = ALL")
##     if the_modelproperties[name].damping.alpha is not None:
## 	fil.write(", alpha = %s"%the_modelproperties[name].damping.alpha)
##     if the_modelproperties[name].damping.beta is not None:
## 	fil.write(", beta = %s"%the_modelproperties[name].damping.beta)
##     fil.write("\n")

    
## def writeElemSurface(fil,number=None,abqdata=None):
##     if number is not None and abqdata is not None:
##         elemsnumbers=where(abqdata.elemprop==number)[0]
##         fil.write('*ELset,Elset=%s\n'% elemproperties[number].surfaces.setname)
##         for i in elemsnumbers:
##             n=i+1
##             fil.write('%s,\n'%n)
##             fil.write('*Surface, type =Element, name=%s\n%s,%s\n' %(elemproperties[number].surfaces.name,elemproperties[number].surfaces.setname,elemproperties[number].surfaces.arg))

            
## def writeNodeSurface(fil,number=None,abqdata=None):
##     if number is not None and abqdata is not None:
##         nodenumbers=where(abqdata.nodeprop==number)[0]
##         fil.write('*ELset,Elset=%s\n'% nodeproperties[number].surfaces.setname)
##         for i in nodenumbers:
##             n=i+1
##             fil.write('%s,\n'%n)
##             fil.write('*Surface, type =Node, name=%s\n%s,%s\n' %(nodeproperties[number].surfaces.name,nodeproperties[number].surfaces.setname,nodeproperties[number].surfaces.arg))

    
## def writeSurface(fil,name=None,abqdata=None):
##     if name is not None and abqdata is not None:
##         if elemproperties[name].surfaces=='Element':
##             global elemsprops
##             elemssetsurfaces=[]
##             for i in range(len(elemproperties)):
##                 if elemproperties[i].surfaces==the_modelproperties[name].setname:
##                     elemssetsurfaces.append(i)
## 		elemssur=[]
## 		for i in elemssetsurfaces:
##                     elemss=where(abqdata.elemprop[:]==i)[0]
##                     elemssur.extend(elemss)
##                 if len(elemssur)>0:
##                     fil.write('*ELset,Elset=%s, internal\n'% the_modelproperties[name].setname)
##                     for i in range(len(elemssur)):
##                         getal=elemssur[i]+1
##                         fil.write('%s,\n'%getal)
## 			fil.write('*Surface, type =Element, name=%s, internal\n%s,%s\n' %(name,the_modelproperties[name].setname,the_modelproperties[name].arg))

## 	elif the_modelproperties[name].surftype=='Node':
##             nodes=[]
##             nFormex=len(abqdata.elems)
##             length=zeros(nFormex)
##             for j in range(nFormex):
##                 length[j]=len(abqdata.elems[j])
##             partnumbers=[]
##             for j in range(nFormex):
##                 partnumbers=append(partnumbers,ones(length[j])*j)
##             for j in elemssur:
##                 partnumber=int(partnumbers[j])
##                 part=abqdata.elems[partnumber]
##                 elemcount=0
##                 for i in range(partnumber):
##                     elemcount+=length[i]
##                     nodes.extend(part[j-elemcount])
## 		nodes = unique(nodes)
## 		fil.write('*Nset,Nset=%s, internal\n'% the_modelproperties[name].setname)
## 		for i in range(len(nodes)):
##                     getal=nodes[i]+1
##                     fil.write('%s,\n'%getal)
## 		fil.write('*Surface, type =Node, name=%s, internal\n%s,%s\n'%(name,the_modelproperties[name].setname,the_modelproperties[name].arg))


## def writeModelProps(fil,prop):
##     for i in the_modelproperties:
##         if the_modelproperties[i].interaction is not None:
##             writeInteraction(fil, i)
##     for i in the_modelproperties:
##         if the_modelproperties[i].damping is not None:
##             writedamping(fil, i)


### Output requests ###################################
#
# Output: goes to the .odb file (for postprocessing with Abaqus/CAE)
# Result: goes to the .fil file (for postprocessing with other means)
#######################################################

def writeStepOutput(fil,kind,type,variable='PRESELECT'):
    """Write the global step output requests.
    
    type = 'FIELD' or 'HISTORY'
    variable = 'ALL' or 'PRESELECT'
    """
    fil.write("*OUTPUT, %s, VARIABLE=%s\n" %(type.upper(),variable.upper()))


def writeNodeOutput(fil,kind,keys,set='Nall'):
    """ Write a request for nodal result output to the .odb file.

    keys is a list of NODE output identifiers
    set is single item or a list of items, where each item is either:
      - a property number
      - a node set name
      for which the results should be written
    """
    output = 'OUTPUT'
    if type(set) == str or type(set) == int:
        set = [ set ]
    for i in set:
        if type(i) == int:
            setname = Nset(str(i))
        else:
            setname = i
        s = "*NODE %s, NSET=%s" % (output,setname)
        fil.write("%s\n" % s)
        for key in keys:
            fil.write("%s\n" % key)


def writeNodeResult(fil,kind,keys,set='Nall',output='FILE',freq=1,
                    globalaxes=False,
                    summary=False,total=False):
    """ Write a request for nodal result output to the .fil or .dat file.

    keys is a list of NODE output identifiers
    set is single item or a list of items, where each item is either:
      - a property number
      - a node set name
      for which the results should be written
    output is either 'FILE' (.fil) or 'PRINT' (.dat)(Standard only)
    freq is the output frequency in increments (0 = no output)

    Extra arguments:
    globalaxes: If 'YES', the requested output is returned in the global axes.
      Default is to use the local axes wherever defined.

    Extra arguments for output='PRINT':
    summary: if True, a summary with minimum and maximum is written
    total: if True, sums the values for each key

    Remark: the 'kind' argument is not used, but is included so that we can
    easily call it with a Results dict as arguments
    """
    if type(set) == str or type(set) == int:
        set = [ set ]
    for i in set:
        if type(i) == int:
            setname = Nset(str(i))
        else:
            setname = i
        s = "*NODE %s, NSET=%s" % (output,setname)
        if freq != 1:
            s += ", FREQUENCY=%s" % freq
        if globalaxes:
            s += ", GLOBAL=YES"
        if output=='PRINT':
            if summary:
                s += ", SUMMARY=YES"
            if total:
                s += ", TOTAL=YES"
        fil.write("%s\n" % s)
        for key in keys:
            fil.write("%s\n" % key)


def writeElemOutput(fil,kind,keys,set='Eall'):
    """ Write a request for element output to the .odb file.

    keys is a list of ELEMENT output identifiers
    set is single item or a list of items, where each item is either:
      - a property number
      - an element set name
      for which the results should be written
    """
    output = 'OUTPUT'
    if type(set) == str or type(set) == int:
        set = [ set ]
    for i in set:
        if type(i) == int:
            setname = Eset(str(i))
        else:
            setname = i
        s = "*ELEMENT %s, ELSET=%s" % (output,setname)
        fil.write("%s\n" % s)
        for key in keys:
            fil.write("%s\n" % key)


def writeElemResult(fil,kind,keys,set='Eall',output='FILE',freq=1,
                    pos=None,
                    summary=False,total=False):
    """ Write a request for element result output to the .fil or .dat file.

    keys is a list of ELEMENT output identifiers
    set is single item or a list of items, where each item is either:
      - a property number
      - an element set name
      for which the results should be written
    output is either 'FILE' (.fil) or 'PRINT' (.dat)(Standard only)
    freq is the output frequency in increments (0 = no output)

    Extra arguments:
    pos: Position of the points in the elements at which the results are
      written. Should be one of:
      'INTEGRATION POINTS' (default)
      'CENTROIDAL'
      'NODES'
      'AVERAGED AT NODES'
      Non-default values are only available for ABAQUS/Standard.
      
    Extra arguments for output='PRINT':
    summary: if True, a summary with minimum and maximum is written
    total: if True, sums the values for each key

    Remark: the 'kind' argument is not used, but is included so that we can
    easily call it with a Results dict as arguments
    """
    if type(set) == str or type(set) == int:
        set = [ set ]
    for i in set:
        if type(i) == int:
            setname = Eset(str(i))
        else:
            setname = i
        s = "*EL %s, ELSET=%s" % (output,setname)
        if freq != 1:
            s += ", FREQUENCY=%s" % freq
        if pos:
            s += ", POSITION=%s" % pos
        if output=='PRINT':
            if summary:
                s += ", SUMMARY=YES"
            if total:
                s += ", TOTAL=YES"
        fil.write("%s\n" % s)
        for key in keys:
            fil.write("%s\n" % key)


def writeFileOutput(fil,resfreq=1,timemarks=False):
    """Write the FILE OUTPUT command for Abaqus/Explicit"""
    fil.write("*FILE OUTPUT, NUMBER INTERVAL=%s" % resfreq)
    if timemarks:
        fil.write(", TIME MARKS=YES")
    fil.write("\n")


##################################################
## Some classes to store all the required information
################################################## 


class Model(Dict):
    """Contains all FE model data."""
    
    def __init__(self,nodes,elems):
        """Create new model data.

        nodes is an array with nodal coordinates
        elems is either a single element connectivity array, or a list of such.
        In a simple case, nodes and elems can be the arrays obtained by 
            nodes, elems = F.feModel()
        This is however limited to a model where all elements have the same
        number of nodes. Then you can use the list of elems arrays. The 'fe'
        plugin has a helper function to create this list. E.g., if FL is a
        list of Formices (possibly with different plexitude), then
          fe.mergeModels([Fi.feModel() for Fi in FL])
        will return the (nodes,elems) tuple to create the Model.

        """
        if not type(elems) == list:
            elems = [ elems ]
        self.nodes = asarray(nodes)
        self.elems = map(asarray,elems)
        nelems = [elems.shape[0] for elems in self.elems]
        self.celems = cumsum([0]+nelems)
        GD.message("Number of nodes: %s" % self.nodes.shape[0])
        GD.message("Number of elements: %s" % self.celems[-1])
        GD.message("Number of element groups: %s" % len(nelems))
        GD.message("Number of elements per group: %s" % nelems)


    def splitElems(self,set):
        """Splits a set of element numbers over the element groups.

        Returns two lists of element sets, the first in global numbering,
        the second in group numbering.
        Each item contains the element numbers from the given set that
        belong to the corresponding group.
        """
        set = unique1d(set)
        split = []
        n = 0
        for e in self.celems[1:]:
            i = set.searchsorted(e)
            split.append(set[n:i])
            n = i

        return split,[ asarray(s) - ofs for s,ofs in zip(split,self.celems) ]
        
 
    def getElems(self,sets):
        """Return the definitions of the elements in sets.

        sets should be a list of element sets with length equal to the
        number of element groups. Each set contains element numbers local
        to that group.
        
        As the elements can be grouped according to plexitude,
        this function returns a list of element arrays matching
        the element groups in self.elems. Some of these arrays may
        be empty.

        It also provide the global and group element numbers, since they
        had to be calculated anyway.
        """
        return [ e[s] for e,s in zip(self.elems,sets) ]
        
        
class Step(Dict):
    """Contains all data about a step."""
    
    def __init__(self,analysis='STATIC',time=[0.,0.,0.,0.],nlgeom='NO',
                 tags=None,
                 bulkvisc=None):
        """Create new analysis data.
        
        analysis is the analysis type. Should be one of:
          'STATIC', 'DYNAMIC', 'EXPLICIT'
        time is either a single float value specifying the step time,
        or a list of 4 values:
          time inc, step time, min. time inc, max. time inc.
        In most cases, only the step time should be specified.
        If nlgeom='YES', the analysis will be non-linear.

        tags is a list of property tags to include in this step.

        bulkvisc is a list of two floats (default: [0.06,1.2]), only used
        in Explicit steps.
        """
        self.analysis = analysis.upper()
        if type(time) == float:
            time = [ 0., time, 0., 0. ]
        self.time = time
        self.nlgeom = nlgeom
        self.tags = tags
        self.bulkvisc = bulkvisc


    def write(self,fil,propDB,out=[],res=[],resfreq=1,timemarks=False):
        """Write a load step.

        propDB is the properties database to use.
        
        Except for the step data itself, this will also write the passed
        output and result requests.
        out is a list of Output-instances.
        res is a list of Result-instances.
        resfreq and timemarks are global values only used by Explicit
        """
        fil.write("*STEP, NLGEOM=%s\n" % self.nlgeom)
        if self.analysis in ['STATIC','DYNAMIC']:
            fil.write("*%s\n" % self.analysis)
        elif self.analysis == 'EXPLICIT':
            fil.write("*DYNAMIC, EXPLICIT\n")
        else:
            GD.message('Skipping undefined step %s' % self.analysis)
            return

        fil.write("%s, %s, %s, %s\n" % tuple(self.time))

        if self.analysis == 'EXPLICIT':
            if self.bulkvisc is not None:
                fil.write("""*BULK VISCOSITY
%s, %s
""" % self.bulkvisc)

        prop = propDB.getProp('n',tag=self.tags,attr=['bound'])
        if prop:
            GD.message("  Writing step boundary conditions")
            writeBoundaries(fil,prop)
     
        prop = propDB.getProp('n',tag=self.tags,attr=['displ'])
        if prop:
            GD.message("  Writing step displacements")
            writeDisplacements(fil,prop)
        
        prop = propDB.getProp('n',tag=self.tags,attr=['cload'])
        if prop:
            GD.message("  Writing step cloads")
            writeCloads(fil,prop)

        prop = propDB.getProp('e',tag=self.tags,attr=['dload'])
        if prop:
            GD.message("  Writing step dloads")
            writeDloads(fil,prop)
        
##         prop = propDB.getProp('',tag=self.tags)
##         if prop:
##             GD.message("  Writing step model props")
##             writeModelProps(fil,prop)
        
        for i in out:
            if i.kind is None:
                writeStepOutput(fil,**i)
            if i.kind == 'N':
                writeNodeOutput(fil,**i)
            elif i.kind == 'E':
                writeElemOutput(fil,**i)
                
        if res and self.analysis == 'EXPLICIT':
            writeFileOutput(fil,resfreq,timemarks)
        for i in res:
            if i.kind == 'N':
                writeNodeResult(fil,**i)
            elif i.kind == 'E':
                writeElemResult(fil,**i)
        fil.write("*END STEP\n")

    
class Output(Dict):
    """A request for output to .odb and history."""
    
    def __init__(self,kind=None,keys=None,set=None,
                 type='FIELD',variable='PRESELECT'):
        """ Create new output request.
        
        kind = None, 'NODE', or 'ELEMENT' (first character suffices)

        For kind=='':

          type =  'FIELD' or 'HISTORY'
          variable = 'ALL' or 'PRESELECT'

        For kind=='NODE' or 'ELEMENT':

          keys is a list of output identifiers (compatible with kind type)
        
          set is single item or a list of items, where each item is either:
            - a property number
            - a node/elem set name
            for which the results should be written
          If no set is specified, the default is 'Nall' for kind=='NODE'
          and 'Eall' for kind='ELEMENT'
        """
        if kind:
            kind = kind[0].upper()
        if set is None:
            set = "%sall" % kind
        Dict.__init__(self,{'kind':kind})
        if kind is not None:
            self.update({'keys':keys,'set':set})
        else:
            self.update({'type':type,'variable':variable})


class Result(Dict):
    """A request for output of results on nodes or elements."""

    # The following values can be changed to set the output frequency
    # for Abaqus/Explicit
    nintervals = 1
    timemarks = False
    
    def __init__(self,kind,keys,set=None,output='FILE',freq=1,time=False,
                 **kargs):
        """Create new result request.
        
        kind = 'NODE' or 'ELEMENT' (actually, the first character suffices)

        keys is a list of output identifiers (compatible with kind type)
        
        set is single item or a list of items, where each item is either:
          - a property number
          - a node/elem set name
          for which the results should be written
        If no set is specified, the default is 'Nall' for kind=='NODE'
        and 'Eall' for kind='ELEMENT'
        
        output is either 'FILE' (.fil) or 'PRINT' (.dat)(Standard only)
        freq is the output frequency in increments (0 = no output)

        Extra keyword arguments are available: see the writeNodeResults and
        writeElemResults functions for details.
        """
        kind = kind[0].upper()
        if set is None:
            set = "%sall" % kind
        Dict.__init__(self,{'keys':keys,'kind':kind,'set':set,'output':output,
                            'freq':freq})
        self.update(dict(**kargs))


############################################################ AbqData
        
class AbqData(CascadingDict):
    """Contains all data required to write the Abaqus input file."""
    
    def __init__(self,model,prop,steps=[],res=[],out=[],bound=None):
        """Create new AbqData. 
        
        model is a Model instance.
        prop is the property database.
        steps is a list of Step instances.
        res is a list of Result instances.
        out is a list of Output instances.
        bound is tag/list of the initial boundary conditions.
          The default is to apply ALL boundary conditions initially.
          Specify a (possibly non-existing) tag to override the default.
        """
        self.model = model
        self.prop = prop
        self.bound = bound
        self.steps = steps
        self.res = res
        self.out = out


    def write(self,jobname=None,group_by_eset=True,group_by_group=False):
        """Write an Abaqus input file.

        jobname is the name of the inputfile, with or without '.inp' extension.
        If None is specified, output is written to sys.stdout
        """
        global materialswritten
        materialswritten = []
        # Create the Abaqus input file
        if jobname is None:
            jobname,filename = 'Test',None
            fil = sys.stdout
        else:
            jobname,filename = abqInputNames(jobname)
            fil = file(filename,'w')
            GD.message("Writing to file %s" % (filename))
        
        writeHeading(fil, """Model: %s     Date: %s      Created by pyFormex
Script: %s 
""" % (jobname, datetime.date.today(), GD.scriptName))

        nnod = self.nodes.shape[0]
        GD.message("Writing %s nodes" % nnod)
        writeNodes(fil, self.nodes)

        GD.message("Writing node sets")
        for p in self.prop.getProp('n',attr=['set']):
            if type(p.set) is ndarray:
                setname = Nset(p.nr)
                writeSet(fil,'NSET',setname,p.set)
                if p.csys is not None:
                    writeTransform(fil,setname,p.csys)

        GD.message("Writing element sets")
        telems = self.model.celems[-1]
        nelems = 0
        for p in self.prop.getProp('e',attr=['eltype','set']):
            if p.set is None:
                set = arange(telems)
            else:
                set = p.set
            print 'Elements of type %s: %s' % (p.eltype,set)
                
            setname = Eset(p.nr)
            gl,gr = self.model.splitElems(set)
            elems = self.model.getElems(gr)
            for i,elnrs,els in zip(range(len(gl)),gl,elems):
                grpname = Eset('grp',i)
                subsetname = Eset(p.nr,'grp',i,)
                nels = len(els)
                if nels > 0:
                    GD.message("Writing %s elements from group %s" % (nels,i))
                    writeElems(fil,els,p.eltype,name=subsetname,eid=elnrs)
                    nelems += nels
                    if group_by_eset:
                        writeSubset(fil,'ELSET',setname,subsetname)
                    if group_by_group:
                        writeSubset(fil,'ELSET',grpname,subsetname)
                    
        GD.message("Total number of elements: %s" % telems)
        if nelems != telems:
            GD.message("!! Number of elements written: %s !!" % nelems)

        GD.message("Writing element sections")
        for p in self.prop.getProp('e',attr=['section','eltype']):
            writeSection(fil,p)

##         GD.message("Writing surfaces")
##         for i in the_nodeproperties:
##             if the_nodeproperties[i].surfaces is not None:
##                 writeNodeSurface(fil,i,self)
##         for i in the_elemproperties:
##             if the_elemproperties[i].surfaces is not None:
##                 writeElemSurface(fil,i,self)

##         GD.message("Writing model properties")
##         for i in the_modelproperties:
##             if the_modelproperties[i].amplitude is not None:
##                 GD.message("Writing amplitude: %s" % i)
##                 writeAmplitude(fil, i)
##             if the_modelproperties[i].intprop is not None:
##                 GD.message("Writing interaction property: %s" % i)
##                 writeIntprop(fil, i)

        GD.message("Writing initial boundary conditions")
        prop = self.prop.getProp('n',tag=self.bound,attr=['bound'])
        if prop:
            writeBoundaries(fil,prop)
    
        GD.message("Writing steps")
        for step in self.steps:
            step.write(fil,self.prop,self.out,self.res,resfreq=Result.nintervals,timemarks=Result.timemarks)

        if filename is not None:
            fil.close()
        GD.message("Done")



def writeAbqInput(abqdata, jobname=None):
    print "This function is deprecated: use the AbqData.write() method instead"
    abqdata.write(jobname)
    

    
##################################################
## Test
##################################################

if __name__ == "script" or __name__ == "draw":

    print "The data hereafter do not form a complete FE model."
    print "See the FeAbq example for a more comprehensive example."
   
    #creating the formex (just 4 points)
    F=Formex([[[0,0]],[[1,0]],[[1,1]],[[0,1]]],[12,8,2])
    draw(F)
    
    # Create property database
    P = PropertyDB()
    #install example materials and section databases
    # either like this
    pyformexdir = GD.cfg['pyformexdir']
    Mat = MaterialDB(pyformexdir+'/examples/materials.db')
    P.setMaterialDB(Mat)
    # or like this
    P.setSectionDB(SectionDB(pyformexdir+'/examples/sections.db'))
    
    # creating some property data
    S1 = ElemSection('IPEA100', 'steel')
    S2 = ElemSection({'name':'circle','radius':10,'sectiontype':'circ'},'steel','CIRC')
    S3 = ElemSection(sectiontype='join')
    BL1 = ElemLoad(label='PZ',value=0.5)
    BL2 = ElemLoad('Grav')
    S2.cross_section=572
    CYL = CoordSystem('cylindrical',[0,0,0,0,0,1])

    # populate the property database
    np1 = P.nodeProp('d1',nset=[0,1],cload=[2,6,4,0,0,0],displ=[(3,5.4)],csys=CYL)
    np2 = P.nodeProp('b0',nset=[1,2],cload=[9,2,5,3,0,4],bound='pinned')
    np3 = P.nodeProp('d2',nset=Nset(np2.nr),bound=[1,1,1,0,0,1],displ=[(2,6),(4,8.)])

    bottom = P.elemProp(12,section=S2,dload=[BL1],eltype='T2D3')
    top = P.elemProp(2,section=S2,dload=[BL2],eltype='FRAME2D')
    diag = P.elemProp(8,section=S3,eltype='conn3d2')
        
    # create the model
    nodes,elems = F.feModel()
    model = Model(nodes,elems)

    # create the steps
    step1 = Step(tags=['d1'])
    step2 = Step(nlgeom='yes',tags=['d2'])

    #create the output requests
    out = [ Output(type='history'),
            Output(type='field'),
            Output(type='field',kind='element',set=Eset(bottom.nr),keys=['SF']),
            ]
    res = [ Result(kind='NODE',keys=['U']),
            Result(kind='ELEMENT',keys=['SF'],set=Eset(top.nr)),
            ]

    all = AbqData(model,P,[step1,step2],res,out,bound=['b0'])
    all.write('testing')
    
    
# End

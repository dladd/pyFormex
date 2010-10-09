#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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
"""Exporting finite element models in Abaqus\ |trade| input file format.

This module provides functions and classes to export finite element models
from pyFormex in the Abaqus\ |trade| input format (.inp).
The exporter handles the mesh geometry as well as model, node and element
properties gathered in a :class:`PropertyDB` database (see module
:mod:`properties`).

While this module provides only a small part of the Abaqus input file format,
it suffices for most standard jobs. While we continue to expand the interface,
depending on our own necessities or when asked by third parties, we do not
intend to make this into a full implementation of the Abaqus input
specification. If you urgently need some missing function, there is always
the possibility to edit the resulting text file or to import it into the
Abaqus environment for further processing.

The module provides two levels of functionality: on the lowest level, there
are functions that just generate a part of an Abaqus input file, conforming
to the Abaqus\ |trade| Keywords manual.

Then there are higher level functions that read data from the property module
and write them to the Abaqus input file and some data classes to organize all
the data involved with the finite element model.
"""

from plugins.properties import *
from plugins.fe import *
from mydict import Dict,CDict
import pyformex as GD
from datetime import datetime
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
    """Determine the name for writing a node set property."""
    if p.name is None:
        return 'Nall'
    else:
        return p.name


def esetName(p):
    """Determine the name for writing an element set property."""
    if p.name is None:
        return 'Eall'
    else:
        return p.name


###########################################################
##   Output Formatting Following Abaqus Keywords Manual  ##
###########################################################
    
#
#  !! This is only a very partial implementation
#     of the Abaqus keyword specs.
#

## The following output functions return the formatted output
## and should be written to file by the caller.
###############################################


def fmtCmd(cmd='*'):
    """Format a command."""
    return '*'+cmd+'\n'


def fmtData1D(data,npl=8,sep=', ',linesep='\n'):
    """Format numerical data in lines with maximum npl items.

    data is a numeric array. The array is flattened and then the data are
    formatted in lines with maximum npl items, separated by sep.
    Lines are separated by linesep.
    """
    data = asarray(data)
    data = data.flat
    return linesep.join([
        sep.join(map(str,data[i:i+npl])) for i in range(0,len(data),npl)
        ])


def fmtData(data,npl=8,sep=', ',linesep='\n'):
    """Format numerical data in lines with maximum npl items.

    data is a numeric array, which is coerced to be a 2D array, either by
    adding a first axis or by collapsing the first ndim-1 axies.
    Then the data are formatted in lines with maximum npl items, separated
    by sep. Lines are separated by linesep.
    """
    data = asarray(data)
    data = data.reshape(-1,data.shape[-1])
    return linesep.join([fmtData1D(row,npl,sep,linesep) for row in data])


def fmtHeading(text=''):
    """Format the heading of the Abaqus input file."""
    out = """**  Abaqus input file created by %s (%s)
**
*HEADING
%s
""" % (GD.Version,GD.Url,text)
    return out

def fmtPart(name='Part-1'):
    """Start a new Part."""
    out = """**  Abaqus input file created by %s (%s)
**
*PART
""" % (name)
    return out


materialswritten=[]
def fmtMaterial(mat):
    """Write a material section.
    
    `mat` is the property dict of the material.
    If the material has a name and has already been written, this function
    does nothing.

    LINEAR ( Default)
    elasticity=linear  (or None)
    
    REQUIRED
    young_modulus
    mat.shear_modulus
    
    OPTIONAL
    mat.poisson_ratio (calcultated if None)
    
    ========================
    
    HYPERELASTIC
    
    elasticity=str hyperelastic
    model=str ogden , polynomial , reduced polynomial
    constants= list  of int sorted abaqus parameter

    ========================

    ANISOTROPIC HYPERELASTIC
    elasticity=anisotropic hyperelastic
    model= holzapfel
    constants= ist  of int sorted abaqus parameter

    ========================

    USER MATERIAL
    elasticity=user
    constants= list  of int sorted abaqus parameter
    
    ============================
    Additional parametrer
    plastic: list([yield stress, yield strain])
    """
    if mat.name is None or mat.name in materialswritten:
        return ""
    
    out ="*MATERIAL, NAME=%s\n" % mat.name
    materialswritten.append(mat.name)
    
    if mat.elasticity is None or mat.elasticity == 'linear':
        if mat.poisson_ratio is None and mat.shear_modulus is not None:
            mat.poisson_ratio = 0.5 * mat.young_modulus / mat.shear_modulus - 1.0

        out += """*ELASTIC
%s,%s
""" % (float(mat.young_modulus), float(mat.poisson_ratio))
    
    elif mat.elasticity == 'hyperelastic':
        out += "*HYPERELASTIC, %s" % mat.type
        if mat.type == 'ogden' or mat.type == 'polynomial' or mat.type == 'reduced polynomial':
            
            ord = 1
            if mat.has_key('order'):
                ord = mat.order
            out += ", N=%s\n" % ord
    
    elif mat.elasticity == 'anisotropic hyperelastic':
        out += "*ANISOTROPIC HYPERELASTIC, HOLZAPFEL\n"
        #TO DO: add possibility to define local orientations!!!"
            
    elif mat.elasticity == 'user':
        out += "*USER MATERIAL, CONSTANTS=%s\n" % len(mat.constants)
        out += fmtData(mat.constants)
    
    if mat.density is not None:
        out += "*DENSITY\n%s\n" % float(mat.density)

    if mat.plastic is not None:
        mat.plastic = asarray(mat.plastic)
        if mat.plastic.ndim != 2:
            raise ValueError,"Plastic data should be 2-dim array"
        ## if mat.plastic.shape[1] > 8:
        ##     raise ValueError,"Plastic data array should have max. 8 columns"
        
        out += "*PLASTIC\n"
        out += fmtData(mat.plastic.shape)

    if mat.damping == 'Yes':
        out += "*DAMPING"
        if mat.alpha != 'None':
            out +=", ALPHA = %s" %mat.alpha
        if mat.beta != 'None':
            out +=", BETA = %s" %mat.beta
        out += '\n'

    return out


def fmtTransform(setname,csys):
    """Write transform command for the given set.

    - `setname` is the name of a node set
    - `csys` is a CoordSystem.
    """
    out = "*TRANSFORM, NSET=%s, TYPE=%s\n" % (setname,csys.sys)
    out += fmtData(csys.data)
    return out


def fmtFrameSection(el,setname):
    """Write a frame section for the named element set.

    Recognized data fields in the property record:
    
    - sectiontype GENERAL:

      - cross_section 
      - moment_inertia_11
      - moment_inertia_12
      - moment_inertia_22
      - torsional_constant

    - sectiontype CIRC:

      - radius

    - sectiontype RECT:

      - width
      - height

    - all sectiontypes:

      - young_modulus
      - shear_modulus
      
    - optional:

      - density: density of the material
      - yield_stress: yield stress of the material
      - orientation: a vector specifying the direction cosines of the 1 axis
    """
    out = ""
    extra = ''
    if el.density:
        extra += ', DENSITY=%s' % float(el.density)
    if el.yield_stress:
            extra += ', PLASTIC DEFAULTS, YIELD STRESS=%s' % float(el.yield_stress)
    if el.shear_modulus is None and el.poisson_ratio is not None:
        el.shear_modulus = el.young_modulus / 2. / (1.+float(el.poisson_ratio))

    sectiontype = el.sectiontype.upper()
    out += "*FRAME SECTION, ELSET=%s, SECTION=%s%s\n" % (setname,sectiontype,extra)
    if sectiontype == 'GENERAL':
        out += "%s, %s, %s, %s, %s \n" % (setname,float(el.density),float(el.cross_section),float(el.moment_inertia_11),float(el.moment_inertia_12),float(el.moment_inertia_22),float(el.torsional_constant))
    elif sectiontype == 'CIRC':
        out += "%s \n" % float(el.radius)
    elif sectiontype == 'RECT':
        out += "%s, %s\n" % (float(el.width),float(el.height))

    if el.orientation != None:
        out += fmtData(el.orientation)
    else:
        out += '\n'
     
    out += fmtData([float(el.young_modulus),float(el.shear_modulus)])

    return out


def fmtGeneralBeamSection(el,setname):
    """Write a general beam section for the named element set.

    To specify a beam section when numerical integration over the section is not required.

    Recognized data fields in the property record:
    
    - sectiontype GENERAL:

      - cross_section 
      - moment_inertia_11
      - moment_inertia_12
      - moment_inertia_22
      - torsional_constant

    - sectiontype CIRC:

      - radius

    - sectiontype RECT:

      - width, height

    - all sectiontypes:

      - young_modulus
      - shear_modulus or poisson_ration
      
    - optional:

      - density: density of the material (required in Abaqus/Explicit)
    """
    out = ""
    extra = ''
    if el.density:
        extra += ', DENSITY=%s' % float(el.density)

    if el.shear_modulus is None and el.poisson_ratio is not None:
        el.shear_modulus = el.young_modulus / 2. / (1.+float(el.poisson_ratio))

    sectiontype = el.sectiontype.upper()
    out += "*BEAM GENERAL SECTION, ELSET=%s, SECTION=%s%s\n" % (setname,sectiontype,extra)
    if sectiontype == 'GENERAL':
        out += "%s, %s, %s, %s, %s \n" % (setname,float(el.density),float(el.cross_section),float(el.moment_inertia_11),float(el.moment_inertia_12),float(el.moment_inertia_22),float(el.torsional_constant))
    elif sectiontype == 'CIRC':
        out += "%s \n" % float(el.radius)
    elif sectiontype == 'RECT':
        out += "%s, %s\n" % (float(el.width),float(el.height))

    if el.orientation != None:
        out += "%s,%s,%s\n" % tuple(el.orientation)
    else:
        out += '\n'
     
    out += "%s, %s \n" % (float(el.young_modulus),float(el.shear_modulus))

    return out


def fmtBeamSection(el,setname):
    """Write a beam section for the named element set.

    To specify a beam section when numerical integration over the section is required.

    Recognized data fields in the property record:
    
    - all sectiontypes: material
    
    - sectiontype GENERAL:

      - cross_section 
      - moment_inertia_11
      - moment_inertia_12
      - moment_inertia_22
      - torsional_constant

    - sectiontype CIRC:

      - radius
      - intpoints1 (number of integration points in the first direction) optional
      - intpoints2 (number of integration points in the second direction) optional

    - sectiontype RECT:

      - width, height
      - intpoints1 (number of integration points in the first direction) optional
      - intpoints2 (number of integration points in the second direction) optional
      
    """
    out = ""

    sectiontype = el.sectiontype.upper()
    out += "*BEAM SECTION, ELSET=%s, MATERIAL=%s, SECTION=%s\n" % (setname,el.material.name,sectiontype)
    if sectiontype == 'GENERAL':
        out += "%s, %s, %s, %s, %s \n" % (setname,float(el.density),float(el.cross_section),float(el.moment_inertia_11),float(el.moment_inertia_12),float(el.moment_inertia_22),float(el.torsional_constant))
    elif sectiontype == 'CIRC':
        out += "%s \n" % float(el.radius)
    elif sectiontype == 'RECT':
        out += "%s, %s\n" % (float(el.width),float(el.height))

    if el.orientation != None:
        out += "%s,%s,%s\n" % tuple(el.orientation)
    else:
        out += '\n'

    if el.intpoints1 != None:
        out += "%s" % el.intpoints1
        if el.intpoints2 != None:
            out += ", %s" % el.intpoints2
        out += "\n"

    return out


def fmtConnectorSection(el,setname):
    """Write a connector section.

    Optional data:
    
    - `behavior` : connector behavior name
    - `orient`  : connector orientation
    """
    out = ""
    if el.sectiontype.upper() != 'GENERAL':
        out += '*CONNECTOR SECTION, ELSET=%s' % setname
        if el.behavior:
            out += ', BEHAVIOR=%s' % el.behavior
        out += '\n%s\n' % el.sectiontype.upper()
        if el.orient:
            out += '%s\n' % el.orient

    return out


def fmtConnectorBehavior(fil,name):
    return "*CONNECTOR BEHAVIOR, NAME=%s\n" % name


def fmtShellSection(el,setname,matname):
    out = ''
    if el.sectiontype.upper() == 'SHELL':
        if matname is not None:
            out += """*SHELL SECTION, ELSET=%s, MATERIAL=%s
%s \n""" % (setname,matname,float(el.thickness))
    return out

 
def fmtSurface(prop):
    """Format the surface definitions.

    Required:

    - set: the elements/nodes in the surface, either numbers or a set name.
    - name: the surface name
    - surftype: 'ELEMENT' or 'NODE'
    - label: face or edge identifier (only required for surftype = 'NODE')
    """
    out = ''
    for p in prop:
        out += "*Surface, name=%s, type=%s\n" % (p.name,p.surftype)
        for e in p.set:
            if p.label is None:
                out += "%s\n" % e
            else:
                out += "%s, %s\n" % (e,p.label)
    return out

 
def fmtSurfaceInteraction(prop):
    """Format the interactions.

    Optional:

    - cross_section (for node based interaction)
    - friction : friction coeff
    """
    out = ''
    for p in prop:
        out += "*SURFACE INTERACTION, NAME=%s\n" % (p.name)
        if p.cross_section is not None:
            out += "%s\n" % p.cross_section
        if p.friction is not None:
            out += "*FRICTION\n%s\n" % float(p.friction)
    return out


def fmtGeneralContact(prop):
    """Format the general contact.
    
    Only implemented on model level
    
    Required:

    - interaction: interaction properties : name or Dict
    """
    out = ''
    for p in prop:
        if type(p.generalinteraction) is str:
            intername = p.generalinteraction
        else:
            intername = p.generalinteraction.name
            out += fmtSurfaceInteraction([p.generalinteraction])
            
        out += "*Contact\n" 
        out += "*Contact Inclusions, ALL EXTERIOR\n"
        out += "*Contact property assignment\n"
        out += ", , %s\n" % intername
    return out


def fmtContactPair(prop):
    """Format the contact pair.

    Required:

    - master: master surface
    - slave: slave surface
    - interaction: interaction properties : name or Dict
    """
    out = ''
    for p in prop:
        if type(p.interaction) is str:
            intername = p.interaction
        else:
            intername = p.interaction.name
            out += fmtSurfaceInteraction([p.interaction])
            
        out += "*Contact Pair, interaction=%s\n" % intername
        out += "%s, %s\n" % (p.slave,p.master)
    return out


def fmtOrientation(prop):
    """Format the orientation.

    Optional:
    
    - definition 
    - system: coordinate system
    - a: a first point
    - b: a second point
    """
    out = ''
    for p in prop:
        out += "*ORIENTATION, NAME=%s" % (p.name)
        if p.definition is not None:
            out += ", definition=%s" % p.definition
        if p.system is not None:
            out += ", SYSTEM=%s" % p.system
        out += "\n"
        if p.a is not None:
            data = tuple(p.a)
            if p.b is not None:
                data += tuple(p.b)
            out += fmtData(data)
        else:
            raise ValueError,"Orientation needs at least point a"
    return out


## The following output sections with possibly large data
## are written directly to file.
##########################################################

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
    writeSet(fil,'ELSET','Eall',[name])


def writeSet(fil,type,name,set,ofs=1):
    """Write a named set of nodes or elements (type=NSET|ELSET)

    `set` : an ndarray. `set` can be a list of node/element numbers,
    in which case the `ofs` value will be added to them,
    or a list of names the name of another already defined set.
    """
    fil.write("*%s,%s=%s\n" % (type,type,name))
    set = asarray(set)
    if set.dtype.kind == 'S':
        # we have set names
        for i in set:
            fil.write('%s\n' % i)
    else:
        for i in set+ofs:
            fil.write("%d,\n" % i)

    

connector_elems = ['CONN3D2','CONN2D2']
frame_elems = ['FRAME3D','FRAME2D']
truss_elems = [
    'T2D2','T2D2H','T2D3','T2D3H',
    'T3D2','T3D2H','T3D3','T3D3H']
beam_elems = [
    'B21', 'B21H','B22','B22H','B23','B23H',
    'B31', 'B31H','B32','B32H','B33','B33H']
membrane_elems = [
    'M3D3',
    'M3D4','M3D4R',
    'M3D6','M3D8',
    'M3D8R',
    'M3D9','M3D9R']
plane_stress_elems = [
    'CPS3',
    'CPS4','CPS4I','CPS4R',
    'CPS6','CPS6M',
    'CPS8','CPS8M']
plane_strain_elems = [
    'CPE3','CPE3H',
    'CPE4','CPE4H','CPE4I','CPE4IH','CPE4R','CPE4RH',
    'CPE6','CPE6H','CPE6M','CPE6MH',
    'CPE8','CPE8H','CPE8R','CPE8RH']
generalized_plane_strain_elems = [
    'CPEG3','CPEG3H',
    'CPEG4','CPEG4H','CPEG4I','CPEG4IH','CPEG4R','CPEG4RH',
    'CPEG6','CPEG6H','CPEG6M','CPEG6MH',
    'CPEG8','CPEG8H','CPEG8R','CPEG8RH']
solid2d_elems = plane_stress_elems + \
                plane_strain_elems + \
                generalized_plane_strain_elems
shell_elems = [
    'S3','S3R', 'S3RS',
    'S4','S4R', 'S4RS','S4RSW','S4R5',
    'S8R','S8R5',
    'S9R5',
    'STRI3',
    'STRI65']
surface_elems = [
    'SFM3D3',
    'SFM3D4','SFM3D4R',
    'SFM3D6',
    'SFM3D8','SFM3D8R']
solid3d_elems = [
    'C3D4','C3D4H',
    'C3D6','C3D6H',
    'C3D8','C3D8H','C3D8R','C3D8RH','C3D10',
    'C3D10H','C3D10M','C3D10MH',
    'C3D15','C3D15H',
    'C3D20','C3D20H','C3D20R','C3D20RH',]

def writeSection(fil,prop):
    """Write an element section.

    prop is a an element property record with a section and eltype attribute
    """
    out = ""
    setname = esetName(prop)
    el = prop.section
    eltype = prop.eltype.upper()

    mat = el.material
    if mat is not None:
        fil.write(fmtMaterial(mat))
            
    if eltype in connector_elems:
        fil.write(fmtConnectorSection(el,setname))

    elif eltype in frame_elems:
        fil.write(fmtFrameSection(el,setname))
            
    elif eltype in truss_elems:
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
    elif eltype in beam_elems:
        if el.integrate:
            fil.write(fmtBeamSection(el,setname))
        else:
            fil.write(fmtGeneralBeamSection(el,setname))

    ############
    ## SHELL elements
    ##########################
    elif eltype in shell_elems:
        fil.write(fmtShellSection(el,setname,mat.name))
    
    ############
    ## SURFACE elements
    ##########################
    elif eltype in surface_elems:
        if el.sectiontype.upper() == 'SURFACE':
            fil.write("""*SURFACE SECTION, ELSET=%s \n""" % setname)
    
    ############
    ## MEMBRANE elements
    ##########################
    elif eltype in membrane_elems:
        if el.sectiontype.upper() == 'MEMBRANE':
            if mat is not None:
                fil.write("""*MEMBRANE SECTION, ELSET=%s, MATERIAL=%s
%s \n""" % (setname,mat.name,float(el.thickness)))


    ############
    ## 3DSOLID elements
    ##########################
    elif eltype in solid3d_elems:
        if el.sectiontype.upper() == '3DSOLID':
            if mat is not None:
                fil.write("""*SOLID SECTION, ELSET=%s, MATERIAL=%s
%s \n""" % (setname,mat.name,1.))

    ############
    ## 2D SOLID elements
    ##########################
    elif eltype in solid2d_elems:
        if el.sectiontype.upper() == 'SOLID':
            if mat is not None:
                fil.write("""*SOLID SECTION, ELSET=%s, MATERIAL=%s
%s \n""" % (setname,mat.name,float(el.thickness)))
            
    ############
    ## RIGID elements
    ##########################
    elif eltype in ['R2D2','RB2D2','RB3D2','RAX2','R3D3','R3D4']:
        if el.sectiontype.upper() == 'RIGID':
            fil.write("""*RIGID BODY,REFNODE=%s,density=%s, ELSET=%s\n""" % (el.nodeset,el.density,setname))



    ############
    ## UNSUPPORTED elements
    ##########################
    else:
        GD.warning('Sorry, elementtype %s is not yet supported' % eltype)


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
        if p.ampl is not None:
            fil.write(", AMPLITUDE=%s" % p.ampl)
        fil.write("\n")
        for v in p.displ:
            dof = v[0]+1
            fil.write("%s, %s, %s, %s\n" % (setname,dof,dof,v[1]))

            
def writeCloads(fil,prop,op='NEW'):
    """Write cloads.

    prop is a list of node property records that should be scanned for
    displ attributes to write.

    By default, the loads are applied as new values in the current step.
    The user can set op='MOD' to add the loads to already existing ones.
    """
    for p in prop:
        setname = nsetName(p)
        fil.write("*CLOAD, OP=%s" % op)
        if p.ampl is not None:
            fil.write(", AMPLITUDE=%s" % p.ampl)
        fil.write("\n")
        for v in p.cload:
            dof = v[0]+1
            fil.write("%s, %s, %s\n" % (setname,dof,v[1]))


def writeDloads(fil,prop,op='NEW'):
    """Write Dloads.
    
    prop is a list property records having an attribute dload

    By default, the loads are applied as new values in the current step.
    The user can set op='MOD' to add the loads to already existing ones.
    """
    for p in prop:
        setname = esetname(p)
        fil.write("*DLOAD, OP=%s" % op)
        if p.ampl is not None:
            fil.write(", AMPLITUDE=%s" % p.ampl)
            fil.write("\n")
        if p.dload.label == 'GRAV':
            fil.write("%s, GRAV, 9.81, 0, 0 ,-1\n" % setname)
        else:
            fil.write("%s, %s, %s\n" % (setname,p.dload.label,p.dload.value))


def writeDsloads(fil,prop,op='NEW'):
    """Write Dsloads.
    
    prop is a list property records having an attribute dsload

    By default, the loads are applied as new values in the current step.
    The user can set op='MOD' to add the loads to already existing ones.
    """
    for p in prop:
        fil.write("*DSLOAD, OP=%s" % op)
        if p.ampl is not None:
            fil.write(", AMPLITUDE=%s" % p.ampl)
        fil.write("\n")
        fil.write("%s, %s, %s\n" % (p.dsload.surface,p.dsload.label,p.dsload.value))

#######################################################
# General model data
#

def writeAmplitude(fil,prop):
    for p in prop:
        fil.write("*AMPLITUDE, NAME=%s, DEFINITION=%s\n" % (p.name,p.amplitude.type))
        for i,v in enumerate(p.amplitude.data):
            fil.write("%s, %s," % tuple(v))
            if i % 4 == 3:
                fil.write("\n")
        if i % 4 != 3:
            fil.write("\n")
        

### Output requests ###################################
#
# Output: goes to the .odb file (for postprocessing with Abaqus/CAE)
# Result: goes to the .fil file (for postprocessing with other means)
#######################################################


def writeNodeOutput(fil,kind,keys,set='Nall'):
    """ Write a request for nodal result output to the .odb file.

    - `keys`: a list of NODE output identifiers
    - `set`: a single item or a list of items, where each item is either
      a property number or a node set name for which the results should
      be written
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
                    globalaxes=False,lastmode=None,
                    summary=False,total=False):
    """ Write a request for nodal result output to the .fil or .dat file.

    - `keys`: a list of NODE output identifiers
    - `set`: a single item or a list of items, where each item is either
      a property number or a node set name for which the results should
      be written
    - `output` is either ``FILE`` (for .fil output) or ``PRINT`` (for .dat
      output)(Abaqus/Standard only)
    - `freq` is the output frequency in increments (0 = no output)

    Extra arguments:

    - `globalaxes`: If 'YES', the requested output is returned in the global
      axes. Default is to use the local axes wherever defined.

    Extra arguments for output=``PRINT``:

    - `summary`: if True, a summary with minimum and maximum is written
    - `total`: if True, sums the values for each key

    Remark: the `kind` argument is not used, but is included so that we can
    easily call it with a `Results` dict as arguments
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
        if lastmode is not None:
            s += ", LAST MODE=%s" % lastmode
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

    - `keys`: a list of ELEMENT output identifiers
    - `set`: a single item or a list of items, where each item is either
      a property number or an element set name for which the results should
      be written
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

    - `keys`: a list of ELEMENT output identifiers
    - `set`: a single item or a list of items, where each item is either
      a property number or an element set name for which the results should
      be written
    - `output` is either ``FILE`` (for .fil output) or ``PRINT`` (for .dat
      output)(Abaqus/Standard only)
    - `freq` is the output frequency in increments (0 = no output)

    Extra arguments:
    
    - `pos`: Position of the points in the elements at which the results are
      written. Should be one of:

      - 'INTEGRATION POINTS' (default)
      - 'CENTROIDAL'
      - 'NODES'
      - 'AVERAGED AT NODES'
      
      Non-default values are only available for ABAQUS/Standard.
      
    Extra arguments for output='PRINT':

    - `summary`: if True, a summary with minimum and maximum is written
    - `total`: if True, sums the values for each key

    Remark: the ``kind`` argument is not used, but is included so that we can
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

    
def writeModelProps(fil,prop):
    """Write model props for this step"""
    for p in prop:
        if p.extra:
            fil.write(p.extra)


##################################################
## Some classes to store all the required information
################################################## 
        
        
class Step(Dict):
    """The basic logical unit in the simulation history.

    In Abaqus, a step is the smallest logical entity in the simulation
    history. It is typically a time step in a dynamic simulation, but it
    can also describe different loading states in a (quasi-)static simulation.

    Our Step class holds all the data describing the global step parameters.
    It combines the Abaqus '*STEP', '*STATIC', '*DYNAMIC' and '*BUCKLE'
    commands (and even some more global parameter setting commands).

    Parameters
    ----------
    
    analysis : str
        The analysis type, one of: 'STATIC', 'DYNAMIC',
        'EXPLICIT', 'PERTURBATION', 'BUCKLE', 'RIKS'

    time : 
      either
    
      - a single float value specifying the step time,
      - a list of 4 values: time inc, step time, min. time inc, max. time inc
      - for LANCZOS: a list of 5 values
      - for RIKS: a list of 8 values
       
      In most cases, only the step time should be specified.

    nlgeom : 'YES' ot 'NO' (default)
        If 'YES', the analysis will be geometrically non-linear. Analysis type
        'RIKS' always sets `nlgeom` to 'YES', 'BUCKLE' sets it to 'NO',
        'PERTURBATION' ignores `nlgeom`.
        
    tags : a list of property tags to include in this step.
        If specified, only the property records having one of the listed values
        as their `tag` attribute will be included in this step.

    
        - `inc`:  the maximum number of increments in a step (the default is 100)
        - `sdi`: determines how severe discontinuities are accounted for
        - timeinc: 
        - `buckle`: specifies the BUCKLE type: 'SUBSPACE' or 'LANCZOS'
        - `incr`: the increment in 'RIKS' type
        - `bulkvisc`:  a list of two floats (default: [0.06,1.2]), only used
          in Explicit steps.
        - `out` and `res`: specific output/result records for this step. They
          come in addition to the global ones.
   x : type
   Description of parameter `x`.

    """

    analysis_types = [ 'STATIC', 'DYNAMIC', 'EXPLICIT', \
                       'PERTURBATION', 'BUCKLE', 'RIKS' ]
    
    def __init__(self,analysis='STATIC',time=[0.,0.,0.,0.],nlgeom='NO',
                 tags=None,inc=None,sdi=None,timeinc=None,
                 buckle='SUBSPACE',incr=0.1,
                 name=None,bulkvisc=None,out=None,res=None):
        """Create new analysis step."""
        
        self.analysis = analysis.upper()
        self.name = name
        if not self.analysis in Step.analysis_types:
            raise ValueError,'analysis should be one of %s' % analysis_types
        if type(time) == float:
            time = [ 0., time, 0., 0. ]
        self.time = time
        if self.analysis == 'RIKS':
            #self.nlgeom = 'YES'
            self.incr = incr
        elif self.analysis == 'BUCKLE':
            self.nlgeom = 'NO'
            self.buckle = buckle
        else:
            self.nlgeom = nlgeom
        self.tags = tags
        self.inc = inc
        self.sdi = sdi
        self.bulkvisc = bulkvisc
        self.out = out
        self.res = res


    def write(self,fil,propDB,out=[],res=[],resfreq=1,timemarks=False):
        """Write a load step.

        propDB is the properties database to use.
        
        Except for the step data itself, this will also write the passed
        output and result requests.
        out is a list of Output-instances.
        res is a list of Result-instances.
        resfreq and timemarks are global values only used by Explicit
        """
        cmd = '*STEP'
        if self.name:
            cmd += ', %s' % self.name
        if self.analysis == 'PERTURBATION':
            cmd += ', PERTURBATION'
        else:
            cmd += ', NLGEOM=%s' % self.nlgeom
            if self.analysis == 'RIKS':
                cmd += ', INCR=%s' % self.incr
        if self.inc:
            cmd += ',INC=%s' % self.inc
        if self.sdi:
            cmd += ',CONVERT SDI=%s' % self.sdi        
        fil.write("%s\n" % cmd)
        if self.analysis in ['STATIC','DYNAMIC']:
            fil.write("*%s\n" % self.analysis)
        elif self.analysis == 'EXPLICIT':
            fil.write("*DYNAMIC, EXPLICIT\n")
        elif self.analysis == 'BUCKLE':
            fil.write("*BUCKLE, EIGENSOLVER=%s\n" % self.buckle)
        elif self.analysis == 'PERTURBATION':
            fil.write("*STATIC")
        elif self.analysis == 'RIKS':
            fil.write("*STATIC, RIKS")
            
                      
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

        prop = propDB.getProp('',tag=self.tags,attr=['dsload'])
        if prop:
            GD.message("  Writing step dsloads")
            writeDsloads(fil,prop)

        prop = propDB.getProp('',tag=self.tags)
        if prop:
            GD.message("  Writing step model props")
            writeModelProps(fil,prop)

        if self.out:
            out += self.out
        
        for i in out:
            if i.kind is None:
                fil.write(i.fmt())
            if i.kind == 'N':
                writeNodeOutput(fil,**i)
            elif i.kind == 'E':
                writeElemOutput(fil,**i)
                
        if self.res:
            res += self.res
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
    
    def __init__(self,kind=None,keys=None,set=None,type='FIELD',variable='PRESELECT',extra='',**options):
        """ Create new output request.

        - `type`: 'FIELD' or 'HISTORY'
        - `kind`: None, 'NODE', or 'ELEMENT' (first character suffices)
        - `extra`: an extra string to be added to the command line. This
          allows to add Abaqus options not handled by this constructor.
          The string will be appended to the command line preceded by a comma.

        For kind=='':

          - `variable`: 'ALL', 'PRESELECT' or ''
          
        For kind=='NODE' or 'ELEMENT':

          - `keys`: a list of output identifiers (compatible with kind type)
          - `set`: a single item or a list of items, where each item is either
            a property number or a node/element set name for which the results
            should be written. If no set is specified, the default is 'Nall'
            for kind=='NODE' and 'Eall' for kind='ELEMENT'
        """
        if 'history' in options:
            GD.warning("The `history` argument in an output request is deprecated.\nPlease use `type='history'` instead.")
        if 'numberinterval' in options:
            GD.warning("The `numberinterval` argument in an output request is deprecated.\nPlease use the `extra` argument instead.")

        if kind:
            kind = kind[0].upper()
        if set is None:
            set = "%sall" % kind
        Dict.__init__(self,{'kind':kind})
        if kind is None:
            self.update({'type':type,'variable':variable,'extra':extra})
        else:
            self.update({'keys':keys,'set':set})


    def fmt(self):
        """Format an output request.

        Return a string with the formatted output command.
        """
        out = ['*OUTPUT',self.type.upper()]
        if self.variable:
            out.append('VARIABLE=%s' % self.variable.upper())
        if self.extra:
            out.append(self.extra)
        return ', '.join(out)+'\n'


class Result(Dict):
    """A request for output of results on nodes or elements."""

    # The following values can be changed to set the output frequency
    # for Abaqus/Explicit
    nintervals = 1
    timemarks = False
    
    def __init__(self,kind,keys,set=None,output='FILE',freq=1,time=False,
                 **kargs):
        """Create new result request.
        
        - `kind`: 'NODE' or 'ELEMENT' (first character suffices)
        - `keys`: a list of output identifiers (compatible with kind type)
        - `set`: a single item or a list of items, where each item is either
           a property number or a node/element set name for which the results
           should be written. If no set is specified, the default is 'Nall'
           for kind=='NODE' and 'Eall' for kind='ELEMENT'
        - `output` is either ``FILE`` (for .fil output) or ``PRINT`` (for .dat
          output)(Abaqus/Standard only)
        - `freq` is the output frequency in increments (0 = no output)

        Extra keyword arguments are available: see the `writeNodeResults` and
        `writeElemResults` methods for details.
        """
        kind = kind[0].upper()
        if set is None:
            set = "%sall" % kind
        Dict.__init__(self,{'keys':keys,'kind':kind,'set':set,'output':output,
                            'freq':freq})
        self.update(dict(**kargs))


############################################################ AbqData
        
class AbqData(object):
    """Contains all data required to write the Abaqus input file."""
    
    def __init__(self,model,prop,nprop=None,eprop=None,steps=[],res=[],out=[],bound=None):
        """Create new AbqData. 
        
        - `model` : a :class:`Model` instance.
        - `prop` : the `Property` database.
        - `steps` : a list of `Step` instances.
        - `res` : a list of `Result` instances.
        - `out` : a list of `Output` instances.
        - `bound` : a tag or alist of the initial boundary conditions.
          The default is to apply ALL boundary conditions initially.
          Specify a (possibly non-existing) tag to override the default.
        """
        if not isinstance(model,Model) or not isinstance(prop,PropertyDB):
            raise ValueError,"Invalid arguments: expected Model and PropertyDB, got %s and %s" % (type(model),type(prop))
        
        self.model = model
        self.prop = prop
        self.nprop = nprop
        self.eprop = eprop
        self.bound = bound
        self.steps = steps
        self.res = res
        self.out = out


    def write(self,jobname=None,group_by_eset=True,group_by_group=False,header='',create_part=False):
        """Write an Abaqus input file.

        - `jobname` : the name of the inputfile, with or without '.inp'
          extension. If None is specified, the output is written to sys.stdout
          An extra header text may be specified.
        - `create_part` : if True, the model will be created as an Abaqus Part,
          followed by and assembly of that part.
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
        
        fil.write(fmtHeading("""Model: %s     Date: %s      Created by pyFormex
Script: %s 
%s
""" % (jobname, datetime.now(), GD.scriptName, header)))

        if create_part:
            fil.write("*PART, name=Part-0\n")
        
        nnod = self.model.nnodes()
        GD.message("Writing %s nodes" % nnod)
        writeNodes(fil,self.model.coords)

        GD.message("Writing node sets")
        for p in self.prop.getProp('n',attr=['set']):
            if p.set is not None:
                # set is directly specified
                set = p.set
            elif p.prop is not None:
                # set is specified by nprop nrs
                if self.nprop is None:
                    print(p)
                    raise ValueError,"nodeProp has a 'prop' field but no 'nprop'was specified"
                set = where(self.nprop == p.prop)[0]
            else:
                # default is all nodes
                set = range(self.model.nnodes())
                
            setname = nsetName(p)
            writeSet(fil,'NSET',setname,set)

        GD.message("Writing coordinate transforms")
        for p in self.prop.getProp('n',attr=['csys']):
            fil.write(fmtTransform(p.name,p.csys))

        GD.message("Writing element sets")
        telems = self.model.celems[-1]
        nelems = 0
        for p in self.prop.getProp('e'):
            if p.set is not None:
                # element set is directly specified
                set = p.set
            elif p.prop is not None:
                # element set is specified by eprop nrs
                if self.eprop is None:
                    print(p)
                    raise ValueError,"elemProp has a 'prop' field but no 'eprop'was specified"
                set = where(self.eprop == p.prop)[0]
            else:
                # default is all elements
                set = range(telems)

            if p.has_key('eltype'):
                print('Elements of type %s: %s' % (p.eltype,set))

                setname = esetName(p)
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
                            writeSet(fil,'ELSET',setname,[subsetname])
                        if group_by_group:
                            writeSet(fil,'ELSET',grpname,[subsetname])
            else:
                writeSet(fil,'ELSET',p.name,p.set)
                    
        GD.message("Total number of elements: %s" % telems)
        if nelems != telems:
            GD.message("!! Number of elements written: %s !!" % nelems)

        ## # Now process the sets without eltype
        ## for p in self.prop.getProp('e',noattr=['eltype']):
        ##     setname = esetName(p)
        ##     writeSet(fil,'ELSET',setname,p.set)

        GD.message("Writing element sections")
        for p in self.prop.getProp('e',attr=['section','eltype']):
            writeSection(fil,p)

        if create_part:
            fil.write("*END PART\n")
            fil.write("*ASSEMBLY, name=Assembly\n")
            fil.write("*INSTANCE, name=Part-0-0, part=Part-0\n")
            fil.write("*END INSTANCE\n")
            fil.write("*END ASSEMBLY\n")

        GD.message("Writing global model properties")
            
        prop = self.prop.getProp('',attr=['amplitude'])
        if prop:
            GD.message("Writing amplitudes")
            writeAmplitude(fil,prop)

        prop = self.prop.getProp('',attr=['orientation'])
        if prop:
            GD.message("Writing orientations")
            fil.write(fmtOrientation(prop))

        prop = self.prop.getProp('',attr=['surftype'])
        if prop:
            GD.message("Writing surfaces")
            fil.write(fmtSurface(prop))
            prop = self.prop.getProp('',attr=['interaction'])
        if prop:       
            GD.message("Writing contact pairs")
            fil.write(fmtContactPair(prop))

        prop = self.prop.getProp('',attr=['generalinteraction'])
        if prop:  
                GD.message("Writing general contact")
                fil.write(fmtGeneralContact(prop))

        prop = self.prop.getProp('n',tag=self.bound,attr=['bound'])
        if prop:
            GD.message("Writing initial boundary conditions")
            writeBoundaries(fil,prop)
    
        GD.message("Writing steps")
        for step in self.steps:
            step.write(fil,self.prop,self.out,self.res,resfreq=Result.nintervals,timemarks=Result.timemarks)

        if filename is not None:
            fil.close()
        GD.message("Wrote Abaqus input file %s" % filename)


    
##################################################
## Some convenience functions
##################################################

def exportMesh(filename,mesh,eltype=None,header=''):
    """Export a finite element mesh in Abaqus .inp format.

    This is a convenience function to quickly export a mesh to Abaqus
    without having to go through the whole setup of a complete
    finite element model.
    This just writes the nodes and elements specified in the mesh to
    the file with the specified name. The resulting file  can then be
    imported in Abaqus/CAE or manual be edited to create a full model.
    If an eltype is specified, it will oerride the value stored in the mesh.
    This should be used to set a correct Abaqus element type matchin the mesh.
    """
    fil = file(filename,'w')
    fil.write(fmtHeading(header))
    if eltype is None:
        eltype = mesh.eltype
    writeNodes(fil,mesh.coords)
    writeElems(fil,mesh.elems,eltype,nofs=1)
    fil.close()
    GD.message("Abaqus file %s written." % filename)

    
##################################################
## Test
##################################################

if __name__ == "script" or __name__ == "draw":

    def TestwriteFormatLines():
        a = arange(27)
        print fmtData1D(a)
        print fmtData1D(a,5)
        print fmtData1D(a,12)

        a = a.reshape(3,9)
        print fmtData(a)
        print fmtData(a,5)
        print fmtData(a,12)

    TestwriteFormatLines()
    exit()

    print("The data hereafter are incorrect and inconsistent.")
    print("See the FeAbq example for a comprehensive example.")
   
    # Create the geometry (4 quads)
    F = Formex(mpattern('123')).replic2(2,2)

    # Create Finite Element model
    nodes,elems = F.feModel()

    if GD.GUI:
        draw(F)
        drawNumbers(F)
        drawNumbers(Formex(nodes),color=red)

    # Create property database
    P = PropertyDB()
    #install example materials and section databases
    # either like this
    Mat = MaterialDB(getcfg('datadir')+'/materials.db')
    P.setMaterialDB(Mat)
    # or like this
    P.setSectionDB(SectionDB(getcfg('datadir')+'/sections.db'))
    
    exit()
    # creating some property data
    S1 = ElemSection('IPEA100', 'steel')
    S2 = ElemSection({'name':'circle','radius':10,'sectiontype':'circ'},'steel','CIRC')
    S3 = ElemSection(sectiontype='join')
    BL1 = ElemLoad(label='PZ',value=0.5)
    BL2 = ElemLoad('Grav')
    S2.cross_section=572
    CYL = CoordSystem('cylindrical',[0,0,0,0,0,1])

    # populate the property database
    P.nodeProp(tag='d1',set=[0,1],cload=[2,6,4,0,0,0],displ=[(3,5.4)],csys=CYL)
    p = P.nodeProp(tag='b0',set=[1,2],cload=[9,2,5,3,0,4],bound='pinned')
    P.nodeProp(tag='d2',setname=p.name,bound=[1,1,1,0,0,1],displ=[(2,6),(4,8.)])

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

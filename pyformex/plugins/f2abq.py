#!/usr/bin/env python pyformex
# $Id$
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
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

import os
from properties import *
from mydict import *
import globaldata as GD
import datetime
import math
from numpy import *


##################################################
## Some Abaqus .inp format output routines
##################################################

# Create automatic names for node and element sets

def autoName(base,*args):
    return (base + '_%s' * len(args)) % args 

def Nset(*args):
    return autoName('Nset',*args)

def Eset(*args):
    return autoName('Eset',*args)


def writeHeading(fil, text=''):
    """Write the heading of the Abaqus input file."""
    head = """**  Abaqus input file created by pyFormex (c) B.Verhegghe
**  (see http://pyformex.berlios.de)
**
*HEADING
%s
""" % text
    fil.write(head)


def writeNodes(fil, nodes, name='Nall', nofs=1):
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


def writeElems(fil, elems, type, name='Eall', eofs=1, nofs=1):
    """Write element group of given type.

    The elements are added to the named element set. 
    If a name different from 'Eall' is specified, the elements will also
    be added to a set named 'Eall'.
    The eofs and nofs specify offsets for element and node numbers.
    The default is 1, because Abaqus numbering starts at 1.  
    """
    fil.write('*ELEMENT, TYPE=%s, ELSET=%s\n' % (type.upper(),name))
    nn = elems.shape[1]
    fmt = '%d' + nn*', %d' + '\n'
    for i,e in enumerate(elems+nofs):
        fil.write(fmt % ((i+eofs,)+tuple(e)))
    writeSubset(fil, 'ELSET', 'Eall', name)


def writeSet(fil, type, name, set, ofs=1):
    """Write a named set of nodes or elements (type=NSET|ELSET)"""
    fil.write("*%s,%s=%s\n" % (type,type,name))
    for i in set+ofs:
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
def writeMaterial(fil, mat):
    """Write a material section.
    
    mat is the property dict of the material.
    If the matrial has a name and has already been written, this function
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
        

##################################################
## Some higher level functions, interfacing with the properties module
##################################################

def writeSection(fil, nr):
    """Write an element section for the named element set.
    
    nr is the property number of the element set.
    """
    el = elemproperties[nr]

    mat = el.material
    if mat is not None:
        writeMaterial(fil,mat)

    ############
    ##FRAME elements
    ##########################
    if el.elemtype.upper() in ['FRAME3D', 'FRAME2D']:
        if el.sectiontype.upper() == 'GENERAL':
            fil.write("""*FRAME SECTION, ELSET=%s, SECTION=GENERAL, DENSITY=%s
%s, %s, %s, %s, %s \n"""%(Eset(nr),float(el.density),float(el.cross_section),float(el.moment_inertia_11),float(el.moment_inertia_12),float(el.moment_inertia_22),float(el.torsional_rigidity)))
            if el.orientation != None:
                fil.write("""%s,%s,%s"""%(el.orientation[0],el.orientation[1],el.orientation[2]))
            fil.write("""\n %s, %s \n"""%(float(el.young_modulus),float(el.shear_modulus)))
        if el.sectiontype.upper() == 'CIRC':
            fil.write("""*FRAME SECTION, ELSET=%s, SECTION=CIRC, DENSITY=%s
%s \n"""%(Eset(nr),float(el.density),float(el.radius)))
            if el.orientation != None:
                fil.write("""%s,%s,%s"""%(el.orientation[0],el.orientation[1],el.orientation[2]))
            fil.write("""\n %s, %s \n"""%(float(el.young_modulus),float(el.shear_modulus)))

    ##############
    ##connector elements
    ##########################  
    elif el.elemtype.upper() in ['CONN3D2', 'CONN2D2']:
        if el.sectiontype.upper() != 'GENERAL':
            fil.write("""*CONNECTOR SECTION,ELSET=%s
%s
""" %(Eset(nr),el.sectiontype.upper()))

    ############
    ##TRUSS elements
    ##########################  
    elif el.elemtype.upper() in ['T2D2', 'T2D2H' , 'T2D3', 'T2D3H', 'T3D2', 'T3D2H', 'T3D3', 'T3D3H']:
        if el.sectiontype.upper() == 'GENERAL':
            fil.write("""*SOLID SECTION, ELSET=%s, MATERIAL=%s
%s
""" %(Eset(nr),el.material.name, float(el.cross_section)))
        elif el.sectiontype.upper() == 'CIRC':
            fil.write("""*SOLID SECTION, ELSET=%s, MATERIAL=%s
%s
""" %(Eset(nr),el.material.name, float(el.radius)**2*math.pi))

    ############
    ##BEAM elements
    ##########################
    elif el.elemtype.upper() in ['B21', 'B21H','B22', 'B22H', 'B23','B23H','B31', 'B31H','B32','B32H','B33','B33H']:
        if el.sectiontype.upper() == 'GENERAL':
            fil.write("""*BEAM GENERAL SECTION, ELSET=%s, SECTION=GENERAL, DENSITY=%s
%s, %s, %s, %s, %s \n"""%(Eset(nr),float(el.density), float(el.cross_section),float(el.moment_inertia_11),float(el.moment_inertia_12),float(el.moment_inertia_22),float(el.torsional_rigidity)))
            if el.orientation != None:
                fil.write("%s,%s,%s"%(el.orientation[0],el.orientation[1],el.orientation[2]))
            fil.write("\n %s, %s \n"%(float(el.young_modulus),float(el.shear_modulus)))
        if el.sectiontype.upper() == 'CIRC':
            fil.write("""*BEAM GENERAL SECTION, ELSET=%s, SECTION=CIRC, DENSITY=%s
%s \n"""%(Eset(nr),float(el.density),float(el.radius)))
            if el.orientation != None:
                fil.write("""%s,%s,%s"""%(el.orientation[0],el.orientation[1],el.orientation[2]))
            fil.write("""\n %s, %s \n"""%(float(el.young_modulus),float(el.shear_modulus)))

    ############
    ## SHELL elements
    ##########################
    elif el.elemtype.upper() in ['STRI3', 'S3','S3R', 'S3RS', 'STRI65','S4','S4R', 'S4RS','S4RSW','S4R5','S8R','S8R5', 'S9R5',]:
        if el.sectiontype.upper() == 'SHELL':
            if mat is not None:
                fil.write("""*SHELL SECTION, ELSET=%s, MATERIAL=%s
%s \n""" % (Eset(nr),mat.name,float(el.thickness)))

    ############
    ## 2D SOLID elements
    ##########################
    elif el.elemtype.upper() in ['CPE3','CPE3H','CPE4','CPE4H','CPE4I','CPE4IH','CPE4R','CPE4RH','CPE6','CPE6H','CPE6M','CPE6MH','CPE8','CPE8H','CPE8R','CPE8RH']:
        if el.sectiontype.upper() == 'SOLID':
            if mat is not None:
                fil.write("""*SOLID SECTION, ELSET=%s, MATERIAL=%s
%s \n""" % (Eset(nr),mat.name,float(el.thickness)))
            
    ############
    ## UNSUPPORTED elements
    ##########################
    else:
        warning('Sorry, elementtype %s is not yet supported' % el.elemtype)
    

def transform(fil, propnr):
    """Transform the nodal coordinates of the nodes with a given property number."""
    n = nodeproperties[propnr]
    if n.coords.lower()=='cartesian':
        if n.coordset!=[]:
            fil.write("""*TRANSFORM, NSET=%s, TYPE=R
%s,%s,%s,%s,%s,%s
"""%(Nset(propnr),n.coordset[0],n.coordset[1],n.coordset[2],n.coordset[3],n.coordset[4],n.coordset[5]))
    elif n.coords.lower()=='spherical':
        fil.write("""*TRANSFORM, NSET=%s, TYPE=S
%s,%s,%s,%s,%s,%s
"""%(Nset(propnr),n.coordset[0],n.coordset[1],n.coordset[2],n.coordset[3],n.coordset[4],n.coordset[5]))
    elif n.coords.lower()=='cylindrical':
        fil.write("""*TRANSFORM, NSET=%s, TYPE=C
%s,%s,%s,%s,%s,%s
"""%(Nset(propnr),n.coordset[0],n.coordset[1],n.coordset[2],n.coordset[3],n.coordset[4],n.coordset[5]))
    else:
        warning('%s is not a valid coordinate system'%nodeproperties[propnr].coords)

    
def writeBoundaries(fil, boundset='ALL', opb=None):
    """Write nodal boundary conditions.
    
    boundset is a list of property numbers of which the boundaries should be written.
    The user can set opb='NEW' to remove the previous boundary conditions, or set opb='MOD' to modify them.
    """
    if boundset!=None:
        fil.write("*BOUNDARY")
        if opb!=None:
            fil.write(", OP=%s" % opb)
        fil.write('\n')
        if isinstance(boundset, list):
            for i in boundset:
                if nodeproperties[i].bound!=None:
                    if isinstance(nodeproperties[i].bound,list):
                        for b in range(6):
                            if nodeproperties[i].bound[b]==1:
                                fil.write("%s, %s\n" % (Nset(i),b+1))
                    elif isinstance(nodeproperties[i].bound,str):
                        fil.write("%s, %s\n" % (Nset(i),nodeproperties[i].bound))
        elif boundset.upper() =='ALL':
            for i in nodeproperties.iterkeys():
                if nodeproperties[i].bound!=None:
                    if isinstance(nodeproperties[i].bound,list):
                        for b in range(6):
                            if nodeproperties[i].bound[b]==1:
                                fil.write("%s, %s\n" % (Nset(i),b+1))
                    elif isinstance(nodeproperties[i].bound,str):
                        fil.write("%s, %s\n" % (Nset(i),nodeproperties[i].bound))
        else:
            warning("The boundaries have to defined in a list 'boundset'")


def writeDisplacements(fil, dispset='ALL', op='MOD'):
    """Write BOUNDARY, TYPE=DISPLACEMENT boundary conditions.

    dispset is a list of the property number of which the displacement should be written.
    By default, the boundary conditions are applied as a modification of the
    existing boundary conditions, i.e. initial conditions and conditions from
    previous steps remain in effect.
    The user can set op='NEW' to remove the previous conditions.
    !!!! This means that initial condtions are also removed!
    """
    fil.write("*BOUNDARY, TYPE=DISPLACEMENT, OP=%s\n" % op)
    if isinstance(dispset, list):
        for i in dispset:
            if nodeproperties[i].displacement!=None:
                for d in range(len(nodeproperties[i].displacement)):
                    fil.write("%s, %s, %s, %s\n" % (Nset(i),nodeproperties[i].displacement[d][0],nodeproperties[i].displacement[d][0],nodeproperties[i].displacement[d][1]))
    elif dispset.upper()=='ALL':
        for i in nodeproperties.iterkeys():
            if nodeproperties[i].displacement!=None:
                for d in range(len(nodeproperties[i].displacement)):
                    fil.write("%s, %s, %s, %s\n" % (Nset(i),nodeproperties[i].displacement[d][0],nodeproperties[i].displacement[d][0],nodeproperties[i].displacement[d][1]))
            
            
def writeCloads(fil, cloadset='ALL', opcl='NEW'):
    """Write cloads.
    
    cloadset is a list of property numbers of which the cloads should be written.
    The user can set opcl='NEW' to remove the previous cloads, or set opcl='MOD' to modify them.
    """
    fil.write("*CLOAD, OP=%s\n" % opcl)
    if isinstance(cloadset, list):
        for i in cloadset:
            if nodeproperties[i].cload!=None:
                for cl in range(6):
                    if nodeproperties[i].cload[cl]!=0:
                        fil.write("%s, %s, %s\n" % (Nset(i),cl+1,nodeproperties[i].cload[cl]))
    elif cloadset.upper()=='ALL':
        for i in nodeproperties.iterkeys():
            if nodeproperties[i].cload!=None:
                for cl in range(6):
                    if nodeproperties[i].cload[cl]!=0:
                        fil.write("%s, %s, %s\n" % (Nset(i),cl+1,nodeproperties[i].cload[cl]))
    else:
        warning("The loads have to be defined in a list 'cloadset'")


def writeDloads(fil, dloadset='ALL', opdl='NEW'):
    """Write Dloads.
    
    dloadset is a list of property numbers of which the dloads should be written.
    The user can set opdl='NEW' to remove the previous cloads, or set opdl='MOD' to modify them.
    """
    fil.write("*DLOAD, OP=%s\n" % opdl)
    if isinstance(dloadset, list):
        for i in dloadset:
            if isinstance(elemproperties[i].elemload, list):
                for load in range(len(elemproperties[i].elemload)):
                    if elemproperties[i].elemload[load].loadlabel.upper() == 'GRAV':
                        fil.write("%s, GRAV, 9.81, 0, 0 ,-1\n" % (Eset(i)))
                    else:
                        fil.write("%s, %s, %s\n" % (Eset(i),elemproperties[i].elemload[load].loadlabel,elemproperties[i].elemload[load].magnitude))
    elif dloadset.upper()=='ALL':
        for i in elemproperties.iterkeys():
            if isinstance(elemproperties[i].elemload, list):
                for load in range(len(elemproperties[i].elemload)):
                    if elemproperties[i].elemload[load].loadlabel.upper() == 'GRAV':
                        fil.write("%s, GRAV, 9.81, 0, 0 ,-1\n" % (Eset(i)))
                    else:
                        fil.write("%s, %s, %s\n" % (Eset(i),elemproperties[i].elemload[load].loadlabel,elemproperties[i].elemload[load].magnitude))
    else:
        warning("The loads have to be defined in a list 'dloadset'")


def writeStepOutput(fil, type='FIELD', variable='PRESELECT', kind='' , set='ALL', ID=None):
    """Write the step output requests.
    
    type =  'FIELD' or 'HISTORY'
    variable = 'ALL' or 'PRESELECT'
    kind = '', 'NODE', or 'ELEMENT'
    set is a list of property numbers of which the data should be written to the ODB-file.
    ID is a list of output variable identifiers. 
    """
    fil.write("*OUTPUT, %s, VARIABLE=%s\n" %(type.upper(),variable.upper()))
    if kind.upper()=='ELEMENT':
        if isinstance(set,list):
            for j in range(len(set)):
                fil.write("*ELEMENT OUTPUT, ELSET=%s\n" % Eset(str(set[j])))
                if ID!=None:
                    for j in range(len(ID)):
                        fil.write("%s \n"%ID[j])
        elif set.upper()=='ALL':
            fil.write("*ELEMENT OUTPUT, ELSET=Eall\n")
            if ID!=None:
                for j in range(len(ID)):
                    fil.write("%s \n"%ID[j])
        else:
            warning("The set should be a list")
    if kind.upper()=='NODE':
        if isinstance(set,list):
            for j in range(len(set)):
                fil.write("*NODE OUTPUT, NSET=%s\n"% Nset(str(set[j])))
                if ID!=None:
                    for j in range(len(ID)):
                        fil.write("%s \n"%ID[j])
        elif set.upper()=='ALL':
            fil.write("*NODE OUTPUT, NSET=Nall\n")
            if ID!=None:
                for j in range(len(ID)):
                    fil.write("%s \n"%ID[j])
        else:
            warning("The set should be a list of property numbers.")


def writeStepData(fil, kind , set='ALL', ID=None, globalaxes='No'):
    """ Write the requested output to the .dat-file.
    
    kind = 'NODE' or 'ELEMENT'
    set is a set of property numbers of which the data should be written to the .dat-file
    ID is a list of output variable identifiers
    If globalaxes = 'yes', all requested output is returned in the global axis system. Otherwise, if the nodeproperties were definied in a local axis system, the output is returned in this axis system.
    """
    if ID!=None:
        if kind.upper()=='NODE':
            if isinstance(set, list):
                for i in set:
                    fil.write("*NODE PRINT, NSET=%s, GLOBAL=%s\n"%(Nset(str(i)),globalaxes))
                    for j in range(len(ID)):
                        fil.write("%s \n"%ID[j])
            if isinstance(set, str):
                fil.write("*NODE PRINT, GLOBAL=%s \n"%globalaxes)
                for j in range(len(ID)):
                    fil.write("%s \n"%ID[j])
        if kind.upper()=='ELEMENT':
            if isinstance(set, list):
                for i in set:
                    fil.write("*EL PRINT, ELSET=%s \n" % Eset(str(i)))
                    for j in range(len(ID)):
                        fil.write("%s \n"%ID[j])
            if isinstance(set, str):
                fil.write("*EL PRINT \n")
                for j in range(len(ID)):
                    fil.write("%s \n"%ID[j])


def writeStep(fil, analysis='STATIC', time=[0,0,0,0], nlgeom='NO', cloadset='ALL', opcl='NEW', dloadset='ALL', opdl='NEW', boundset=None, opb=None, dispset='ALL', op='MOD', outp=[], dat=[]):
    """Write a load step.
        
    analysistype is the analysis type. Currently, only STATIC is supported.
    time is a list which defines the time step.
    If nlgeom='YES', the analysis will be non-linear.
    Cloadset is a list of property numbers of which the cloads will be used in this analysis.
    Dloadset is a list of property numbers of which the dloads will be used in this analysis.
    Boundset is a list of propery numbers of which the bounds will be used in this analysis.
    By default, the load is applied as a new load, i.e. loads
    from previous steps are removed. The user can set op='MOD'
    to keep/modify the previous loads.
    outp is a list of Odb-instances.
    dat is a list of Dat-instances.
    """ 
    if analysis.upper()=='STATIC':
        fil.write("""*STEP, NLGEOM=%s
*STATIC
%s, %s, %s, %s
""" % (nlgeom, time[0], time[1], time[2], time[3]))
        writeBoundaries(fil, boundset, opb)
        writeDisplacements(fil, dispset,op)
        writeCloads(fil, cloadset, opcl)
        writeDloads(fil, dloadset, opdl)
        for i in range(len(outp)):
            writeStepOutput(fil, outp[i].type,outp[i].variable,outp[i].kind,outp[i].set,outp[i].ID)
        for i in range(len(dat)):
            writeStepData(fil, dat[i].kind, dat[i].set, dat[i].ID, dat[i].globalaxes)
        fil.write("*END STEP\n")


##################################################
## Some classes to store all the required information
################################################## 


class Model(Dict):
    """Contains all model data."""
    
    def __init__(self, nodes, elems, nodeprop, elemprop, initialboundaries='ALL'):
        """Create new model data.
        
        Nodes and elems are arrays, such as those obtained by 
            nodes, elems = F.feModel()
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
        !! This limits the model to elements with the same number of nodes
        !! A solution would be to use a list of elems arrays 
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        nodeprop is a list of all the node property numbers.
        elemprop is a list of all the element property numbers. This list can be obtained by 
            elemprop = F.p
        initialboundaries is a list of all the initial boundaries. It can also be the string 'ALL'. This is the default.
        """
##         if not type(elems) == list:
##             elems = [ elems ]
        Dict.__init__(self, {'nodes':nodes, 'elems':elems, 'nodeprop':nodeprop, 'elemprop':elemprop, 'initialboundaries':initialboundaries}) 


class Analysis(Dict):
    """Contains all data about the analysis."""
    
    def __init__(self, analysistype='STATIC', time=[0,0,0,0], nlgeom='NO', cloadset='ALL', opcl='NEW', dloadset='ALL', opdl='NEW', boundset=None, opb=None, dispset='ALL', op='MOD'):
        """Create new analysis data.
        
        analysistype is the analysis type. Currently, only STATIC is supported.
        time is a list which defines the time step.
        If nlgeom='YES', the analysis will be non-linear.
        Cloadset is a list of property numbers of which the cloads will be used in this analysis.
        Dloadset is a list of property numbers of which the dloads will be used in this analysis.
        Boundset is a list of property numbers of which the bounds will be used in this analysis. Initial boundaries are defined in a Model instance.
        By default, the load is applied as a new load, i.e. loads
        from previous steps are removed. The user can set op='MOD'
        to keep/modify the previous loads.
        """
        Dict.__init__(self,{'analysistype':analysistype, 'time':time, 'nlgeom':nlgeom, 'cloadset':cloadset, 'opcl':opcl, 'dloadset':dloadset, 'opdl':opdl, 'boundset':boundset, 'opb': opb, 'dispset' : dispset , 'op': op})

    
class Odb(Dict):
    """Contains all data about the output requests to the .ODB-file."""
    
    def __init__(self, type='FIELD', variable='PRESELECT', kind = '' , set='all', ID=None):
        """ Create new ODB data.
        
        type =  'FIELD' or 'HISTORY'
        variable = 'ALL' or 'PRESELECT'
        kind = 'NODE', or 'ELEMENT'
        set is a list of property numbers of which the data should be written to the ODB-file.
        ID is a list of output variable identifiers. 
        """
        Dict.__init__(self, {'type':type, 'variable':variable, 'kind':kind, 'set':set, 'ID': ID})


class Dat(Dict):
    """Contains all data about output requests to the .dat-file."""
    
    def __init__(self, kind='NODE', set='ALL', ID=['COORD'], globalaxes = 'No'):
        """Create new Dat data.
        
        kind = 'NODE' or 'ELEMENT'
        set is a set of property numbers of which the data should be written to the .dat-file.
        ID is a list of output variable identifiers.
        If globalaxes = 'yes', all requested output is returned in the global axis system. Otherwise, if the nodeproperties were definied in a local axis system, the output is returned in this axis system.
        """
        Dict.__init__(self, {'kind':kind, 'set':set, 'ID':ID, 'globalaxes':globalaxes})


class AbqData(CascadingDict):
    """Contains all data required to write the abaqus input file."""
    
    def __init__(self, model, analysis=[], dat=[], odb=[]):
        """Create new AbqData. 
        
        model is a Model instance.
        analysis is a list of Analysis instances.
        dat is a list of Dat instances.
        odb is a list of Odb instances.
        """
        CascadingDict.__init__(self, {'model':model, 'analysis':analysis, 'dat':dat, 'odb':odb})

    
##################################################
## Combine all previous functions to write the Abaqus input file
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


def writeAbqInput(abqdata, jobname=None):
    """Write an Abaqus input file.
    
    abqdata is an AbqData-instance.
    job is the name of the inputfile.
    """
    global materialswritten
    materialswritten = []
    # Create the Abaqus input file
    if jobname is None:
        jobname = str(GD.scriptName)[:-3]
    jobname,filename = abqInputNames(jobname)
    fil = file(filename,'w')
    GD.message("Writing to file %s" % (filename))
    
    #write the heading
    writeHeading(fil, """Model: %s     Date: %s      Created by pyFormex
Script: %s 
""" % (jobname, datetime.date.today(), GD.scriptName))
    
    #write all nodes
    nnod = abqdata.nodes.shape[0]
    GD.message("Writing %s nodes" % nnod)
    writeNodes(fil, abqdata.nodes)
    
    #write nodesets and their transformations
    GD.message("Writing node sets")
    nlist = arange(nnod)
    for i in nodeproperties:
        nodeset = nlist[array(abqdata.nodeprop) == i]
        writeSet(fil, 'NSET', Nset(str(i)), nodeset)
        transform(fil,i)

    #write elemsets
    GD.message("Writing element sets")
    n=0
    # we process the elementgroups one by one
    GD.message("Number of element groups: %s" % len(abqdata.elems))
    for j,elems in enumerate(abqdata.elems):
        ne = len(elems)
        eprop = abqdata.elemprop[n:n+ne]
        GD.message("Number of elements in group %s: %s" % (j,ne))
        for i in elemproperties:
            elemset = arange(ne)[eprop == i] # The elems with property i
            if len(elemset) > 0:
                print "Elements in group %s with property %s: %s" % (j,i,elemset)
                subsetname = '%s' % Eset(j,i)
                writeElems(fil, elems[elemset],elemproperties[i].elemtype, subsetname,eofs=n+1)
                n += len(elemset)
                writeSubset(fil, 'ELSET', Eset(i), subsetname)
    GD.message("Total number of elements: %s" % n)
    # Write element sections
    GD.message("Writing element sections")
    for i in elemproperties:
        writeSection(fil, i)

    #write steps
    GD.message("Writing steps")
    writeBoundaries(fil, abqdata.initialboundaries)
    for i in range(len(abqdata.analysis)):
        a=abqdata.analysis[i]
        writeStep(fil, a.analysistype,a.time, a.nlgeom, a.cloadset, a.opcl, a.dloadset, a. opdl, a.boundset, a.opb, a.dispset, a.op, abqdata.odb, abqdata.dat)

    GD.message("Done")


##################################################
## Test
##################################################

if __name__ == "__main__":
    from formex import *
    
    #creating the formex
    F=Formex([[[0,0]],[[1,0]],[[1,1]],[[0,1]]],[12,8,2])
    
    #install example databases
    # either like this
    Mat = MaterialDB('examples/materials.db')
    setMaterialDB(Mat)
    # or like this
    setSectionDB(SectionDB('examples/sections.db'))
    
    # creating properties
    S1=ElemSection('IPEA100', 'steel')
    S2=ElemSection({'name':'circle','radius':10},'steel','CIRC')
    S3=ElemSection(sectiontype='join')
    BL1=ElemLoad(0.5,loadlabel='PZ')
    BL2=ElemLoad(loadlabel='Grav')
    #S2.density=7850
    S2.cross_section=572
    np1=NodeProperty(9,[2,6,4,0,0,0], displacement=[[3,5]],coords='cylindrical',coordset=[0,0,0,0,0,1])
    np2=NodeProperty(8,cload=[9,2,5,3,0,4], bound='pinned')
    np3=NodeProperty(7,None,[1,1,1,0,0,1], displacement=[[2,6],[4,8]])
    bottom = ElemProperty(12,S2,[BL1],'T2D3')
    top = ElemProperty(2,S2,[BL2],elemtype='FRAME2D')
    diag = ElemProperty(8,S3,elemtype='conn3d2')
    
    #creating the input file
    nodes,elems = F.feModel()
    step1=Analysis(nlgeom='yes', cloadset=[], boundset=[8])
    step2=Analysis(cloadset=[9], dloadset=[], dispset=[9])
    outhist = Odb(type='history')
    outfield = Odb(type='field', kind='node', set= [9,8], ID='SF')
    elemdat = Dat(kind='element',ID=['U','coord'])
    nodedat = Dat(kind='node',set=[7,9], ID=['U','coord'])
    model = Model(nodes, elems, [9,8,0,7], F.p, initialboundaries=[7])
    total = AbqData(model, analysis=[step1, step2], dat=[elemdat, nodedat], odb=[outhist, outfield])
    print model
    writeAbqInput(total, jobname='testing')
    
    

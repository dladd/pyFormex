#!/usr/bin/env python
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
"""General framework for attributing properties to Formex elements.

Properties can really be just about any Python object.
Properties are identified and connected to a Formex element by the
prop values that are stored in the Formex.
"""

import globaldata as GD

from flatkeydb import *
from mydict import *


class Database(Dict):
    """A class for storing properties in a database."""
    
    def __init__(self,data={}):
        """Initialize a database.

        The database can be initialized with a dict.
        """
        Dict.__init__(self,data)

        
    def readDatabase(self,filename,*args,**kargs):
        """Import all records from a database file.

        For now, it can only read databases using flatkeydb.
        args and kargs can be used to specify arguments for the
        FlatDB constructor.
        """
        mat = FlatDB(*args,**kargs)
        mat.readFile(filename)
        for k,v in mat.iteritems():
            self[k] = Dict(v)

            
class MaterialDB(Database):
    """A class for storing material properties."""
    
    def __init__(self,data={}):
        """Initialize a materials database.

        If data is a dict, it contains the database.
        If data is a string, it specifies a filename where the
        database can be read.
        """
        Database.__init__(self,data)
        if type(data) == str:
            self.readDatabase(data,['name'],beginrec='material',endrec='endmaterial')
        elif type(data) == dict:
            self.update(data)
        else:
            raise ValueError,"Expected a filename or a dict."


class SectionDB(Database):
    """A class for storing section properties."""
    
    def __init__(self,data={}):
        """Initialize a section database.

        If data is a dict, it contains the database.
        If data is a string, it specifies a filename where the
        database can be read.
        """
        Database.__init__(self,{})
        if type(data) == str:
            self.readDatabase(data,['name'],beginrec='section',endrec='endsection')
        elif type(data) == dict:
            self.update(data)
        else:
            raise ValueError,"Expected a filename or a dict."

    
materials = MaterialDB()
sections = SectionDB()
properties = CascadingDict()
nodeproperties = CascadingDict()
elemproperties = CascadingDict()

## def init_properties():
##     global materials, sections, properties, nodeproperties, elemproperties
##     print "INITIALIZING THE properties MODULE"
##     if materials is None:
##         materials = MaterialDB({})
##     if sections is None:
##         sections = SectionDB()
##     if properties is None:
##         properties = CascadingDict()
##     if nodeproperties is None:
##         properties = CascadingDict()
##     if elemproperties is None:
##         properties = CascadingDict()



def setMaterialDB(aDict):
    global materials
    if isinstance(aDict,MaterialDB):
        materials = aDict

def setSectionDB(aDict):
    global sections
    if isinstance(aDict,SectionDB):
        sections = aDict

def setNodePropDB(aDict):
    global nodeproperties
    if isinstance(aDict,CascadingDict):
        nodeproperties = aDict


class Property(CascadingDict):
    """A general properties class.

    This class should only provide general methods, such as
    add, change and delete properties, lookup, print, and
    of course, connect properties to Formex elements.
    """
    global properties
    def __init__(self, nr, data = {}):
        """Create a new property. Empty by default.
        
        A property is created and the data is stored in a Dict called 'properties'. 
        The key to access the property is the number.
        This number should be the same as the property number of the Formex element.
        """
        CascadingDict.__init__(self, data)
        properties[nr] = self 


class NodeProperty(Property):
    """Properties related to a single node."""

    def __init__(self,nr,nset=None,cload=None,bound=None,displacement=None,coords='cartesian',coordset=[]):
        """Create a new node property, empty by default.
        
        Each new node property is stored in the global Dict 'nodeproperties'. 
        The key to access the node property is the unique number 'nr'.

        The nodes for which this property holds are identified either by
        an explicit 'nset' node list or by the nodes having a matching global
        property number set to 'nr'.

        A node property can hold the following fields:
        - nset : a list of node numbers 
        - cload : a concentrated load
        - bound : a boundary condition
        - displacement: prescribe a displacement
        - coords: the coordinate system which is used for the definition of cload and bound. There are three options:
        cartesian, spherical and cylindrical
        - coordset: a list of 6 coordinates; the 2 points that specify the transformation 
        """
        global nodeproperties
        if ((cload is None or (isinstance(cload,list) and len(cload)==6)) and
            (bound is None or (isinstance(bound,list) and len(bound)==6) or
             isinstance(bound, str))): 
            CascadingDict.__init__(self, {'nset':nset,'cload':cload,'bound':bound,'displacement':displacement,'coords':coords,'coordset':coordset})
            nodeproperties[nr] = self
        else: 
            warning('A pointload and a boundary condition have to be a list containing 6 items')


class ElemProperty(Property):
    """Properties related to a single element."""
    
    def __init__(self,nr,elemsection=None,elemload=None,elemtype=None,eset=None): 
        """Create a new element property, empty by default.
        
        Each new element property is stored in a global Dict 'elemproperties'. 
        The key to access the element property is the unique number 'nr'.

        The elements for which this property holds are identified either by
        an explicit 'eset' node list or by the nodes having a matching global
        property number set to 'nr'.
        
        An element property can hold the following fields:
        - eset : a list of element numbers 
        - elemsection : the section properties of the element. This is an ElemSection instance.
        - elemload : the loading of the element. This is a list of ElemLoad instances.
        - elemtype: the type of element that is to be used in the analysis. 
        """    
        CascadingDict.__init__(self, {'eset':eset,'elemsection':elemsection,'elemload':elemload,'elemtype':elemtype})
        elemproperties[nr] = self


class ElemSection(Property):
    """Properties related to the section of a beam."""

    def __init__(self, section = None, material = None, orientation = None, behavior = None, range = 0.0):

        ### sectiontype is now an attribute of section ###
        
        """Create a new element section property. Empty by default.
        
        An element section property can hold the following sub-properties:
        - section: the section properties of the element. This can be a dict
          or a string. The required data in this dict depend on the
          sectiontype. Currently known keys to f2abq.py are:
            cross_section, moment_inertia_11, moment_inertia_12,
            moment_inertia_22, torsional_rigidity, radius
        - material: the element material. This can be a dict or a string.
          Currently known keys to f2abq.py are:
            young_modulus, shear_modulus, density, poisson_ratio
        - sectiontype: the sectiontype of the element. 
        - 'orientation' is a list [First direction cosine, second direction cosine, third direction cosine] of the first beam section axis. This allows to change the orientation of the cross-section.
        - behavior: the behavior of the connector
        """    
        CascadingDict.__init__(self,{})
        #self.sectiontype = sectiontype
        self.orientation = orientation
        self.behavior = behavior
        self.range = range
        self.addMaterial(material)
        self.addSection(section)
        if self.section.sectiontype is None:
            self.sectiontype = 'general'
    
    def addSection(self, section):
        """Create or replace the section properties of the element.

        If 'section' is a dict, it will be added to 'sections'.
        If 'section' is a string, this string will be used as a key to
        search in 'sections'.
        """
        if isinstance(section, str):
            if sections.has_key(section):
                self.section = sections[section]
            else:
                warning("This section is not available in the database")
        elif isinstance(section,dict):
            sections[section['name']] = CascadingDict(section)
            self.section = sections[section['name']]
        elif section==None:
            self.section = section
        else: 
            warning("addSection requires a string or dict")
    
    def addMaterial(self, material):
        """Create or replace the material properties of the element.

        If the argument is a dict, it will be added to 'materials'.
        If the argument is a string, this string will be used as a key to
        search in 'materials'.
        """
        if isinstance(material, str) :
            if materials.has_key(material):
                self.material = materials[material] 
            else:
                warning("This material is not available in the database")
        elif isinstance(material, dict):
            materials[material['name']] = CascadingDict(material)
            self.material = materials[material['name']]
        elif material==None:
            self.material=material
        else:
            warning("addMaterial requires a string or dict")


class ElemLoad(Property):
    """Properties related to the load of a beam."""

    def __init__(self, magnitude = None, loadlabel = None):
        """Create a new element load. Empty by default.
        
        An element load can hold the following sub-properties:
        - magnitude: the magnitude of the distibuted load.
        - loadlabel: the distributed load type label.
        """          
        Dict.__init__(self, {'magnitude' : magnitude, 'loadlabel' : loadlabel})



## # INITIALIZE

## init_properties()



# Test

if __name__ == "__main__":

    import sys,os
    #print __file__
    #print os.path.dirname(__file__)
    #sys.exit()
    Mat = MaterialDB('examples/materials.db')
    setMaterialDB(Mat)
    Sec = SectionDB('examples/sections.db')
    setSectionDB(Sec)

    Stick=Property(1, {'colour':'green', 'name':'Stick', 'weight': 25, 'comment':'This could be anything: a gum, a frog, a usb-stick,...'})
    author=Property(5,{'Name':'Tim Neels', 'Address':CascadingDict({'street':'Krijgslaan', 'city':'Gent','country':'Belgium'})})
    
#    print Stick
#    print properties[1] 
    
    Stick.weight=30
    Stick.length=10
    print properties[1]
    
#    print properties[5]
    properties[5].street='Voskenslaan'
    print author
    print properties[5]
    print author.street
    print author.Address.street
    
    P1 = [ 1.0,1.0,1.0, 0.0,0.0,0.0 ]
    P2 = [ 0.0 ] * 3 + [ 1.0 ] * 3 
    B1 = [ 0.0 ] * 6

    print Mat
    print Sec
    vert = ElemSection('IPEA100', 'steel')
    hor = ElemSection({'name':'IPEM800','A':951247,'I':CascadingDict({'Ix':1542,'Iy':6251,'Ixy':352})}, {'name':'S400','E':210,'fy':400})
    q = ElemLoad(magnitude=2.5, loadlabel='PZ')
    top = ElemProperty(1,hor,[q],'B22')
    column = ElemProperty(2,vert, elemtype='B22')
    diagonal = ElemProperty(4,hor,elemtype='B22')
    print 'elemproperties'
    for key, item in elemproperties.iteritems():
       print key, item	

    bottom=ElemProperty(3,hor,[q])


    topnode = NodeProperty(1,cload=[5,0,-75,0,0,0])
    foot = NodeProperty(2,bound='pinned')

    np = {}
    np['1'] = NodeProperty(1, P1)
    np['2'] = NodeProperty(2, cload=P2)
    np['3'] = np['2']
    np['3'].bound = B1
    np['1'].cload[1] = 33.0
    np['7'] = NodeProperty(7, bound=B1)

    for key, item in materials.iteritems():
        print key, item

    print 'properties'
    for key, item in properties.iteritems():
        print key, item

    print 'nodeproperties'    
    for key, item in nodeproperties.iteritems():
        print key, item
    
    print 'elemproperties'
    for key, item in elemproperties.iteritems():
        print key, item
##        
    print elemproperties[3].A
    bottom.A=555
    print elemproperties[3]
    print elemproperties[3].A
    elemproperties[3].A=444
    print bottom.A
    print elemproperties[3].A
    
    print "beamsection attributes"
    for key,item in elemproperties.iteritems():
        print key,item.elemload
    
    for key,item in elemproperties.iteritems():
        print key,item.E
    
    print "cload attributes"
    for key,item in nodeproperties.iteritems():
        print key,item.cload

    print "cload attributes"
    for key,item in np.iteritems():
        print key,item.cload

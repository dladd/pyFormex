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
"""General framework for attributing properties to Formex elements.

Properties can really be just about any Python object.
Properties are identified and connected to a Formex element by the
prop values that are stored in the Formex.
"""

from flatkeydb import *
from mydict import *
from numpy import *


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
        Database.__init__(self,{})
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

    
the_materials = MaterialDB()
the_sections = SectionDB()
the_properties = CascadingDict()
the_nodeproperties = CascadingDict()
the_elemproperties = CascadingDict()
the_modelproperties = CascadingDict()

## def init_properties():
##     global the_materials, the_sections, the_properties,\
##            the_nodeproperties, the_elemproperties
##     print "INITIALIZING THE properties MODULE"
##     if the_materials is None:
##         materials = MaterialDB({})
##     if the_sections is None:
##         sections = SectionDB()
##     if the_properties is None:
##         properties = CascadingDict()
##     if the_nodeproperties is None:
##         properties = CascadingDict()
##     if the_elemproperties is None:
##         properties = CascadingDict()



def setMaterialDB(aDict):
    global the_materials
    if isinstance(aDict,MaterialDB):
        the_materials = aDict

def setSectionDB(aDict):
    global the_sections
    if isinstance(aDict,SectionDB):
        the_sections = aDict

def setNodePropDB(aDict):
    global the_nodeproperties
    if isinstance(aDict,CascadingDict):
        the_nodeproperties = aDict


class Property(CascadingDict):
    """A general properties class.

    This class should only provide general methods, such as
    add, change and delete properties, lookup, print, and
    of course, connect properties to Formex elements.
    """
    global the_properties
    def __init__(self, nr, data = {}):
        """Create a new property. Empty by default.
        
        A property is created and the data is stored in a Dict called 'the_properties'. 
        The key to access the property is the number.
        This number should be the same as the property number of the Formex element.
        """
        CascadingDict.__init__(self, data)
        the_properties[nr] = self 

class ModelProperty(Property):
    """Properties related to a model."""

    def __init__(self, name, amplitude=None,intprop=None,surface=None,interaction=None,damping=None):
        """Create a new node property. Empty by default.
        
        A model property is created and the data is stored in a Dict called 'modelproperties'. 
        The key to access the model property is the name.
        """
        global the_modelproperties
        CascadingDict.__init__(self, {'amplitude':amplitude,'intprop':intprop,'surface':surface,'interaction':interaction,'damping':damping})
        the_modelproperties[name] = self


class NodeProperty(Property):
    """Properties related to a single node."""

    def __init__(self,nr,nset=None,cload=None,bound=None,displacement=None,coords='cartesian',coordset=[]):
        """Create a new node property, empty by default.
        
        Each new node property is stored in the global Dict 'the_nodeproperties'. 
        The key to access the node property is the unique number 'nr'.

        The nodes for which this property holds are identified either by
        an explicit 'nset' node list or by the nodes having a matching global
        property number set to 'nr'.

        A node property can hold the following fields:
        - nset : a list of node numbers 
        - cload : a concentrated load
        - bound : a boundary condition
        - displacement: prescribe a displacement
        - coords: the coordinate system which is used for the definition of
          cload and bound. There are three options:
            cartesian, spherical and cylindrical
        - coordset: a list of 6 coordinates; the two points that specify
          the transformation 
        """
        global the_nodeproperties
        if ((cload is None or (isinstance(cload,list) and len(cload)==6)) and
            (bound is None or (isinstance(bound,list) and len(bound)==6) or
             isinstance(bound, str))): 
            CascadingDict.__init__(self, {'nset':nset,'cload':cload,'bound':bound,'displacement':displacement,'coords':coords,'coordset':coordset})
            the_nodeproperties[nr] = self
        else: 
            raise ValueError,"cload/bound property requires a list of 6 items"


class ElemProperty(Property):
    """Properties related to a single element."""
    
    def __init__(self,nr,elemsection=None,elemload=None,elemtype=None,eset=None): 
        """Create a new element property, empty by default.
        
        Each new element property is stored in a global Dict 'the_elemproperties'. 
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
        the_elemproperties[nr] = self


class ElemSection(Property):
    """Properties related to the section of an element."""

    def __init__(self,section=None,material=None,orientation=None,behavior=None,range=0.0,sectiontype=None):

        ### sectiontype is now preferably an attribute of section ###
        
        """Create a new element section property. Empty by default.
        
        An element section property can hold the following sub-properties:
       - section: the section properties of the element. This can be a dict
          or a string. The required data in this dict depend on the
          sectiontype. Currently the following keys are used by fe_abq.py:
            - sectiontype: the type of section: one of following:
              'solid': a solid 2D or 3D section,
              'circ' : a plain circular section,
              'rect' : a plain rectangular section,
              'pipe' : a hollow circular section,
              'box'  : a hollow rectangular section,
              'I'    : an I-beam,
              'general' : anything else (automatically set if not specified).
              !! Currently only 'solid' and 'general' are allowed.
            - the cross section characteristics :
              cross_section, moment_inertia_11, moment_inertia_12,
              moment_inertia_22, torsional_rigidity
            - for sectiontype 'circ': radius
         - material: the element material. This can be a dict or a string.
          Currently known keys to fe_abq.py are:
            young_modulus, shear_modulus, density, poisson_ratio
        - 'orientation' is a list of 3 direction cosines of the first beam
          section axis.
        - behavior: the behavior of the connector
        """
        CascadingDict.__init__(self,{})
        if sectiontype is not None:
            self.sectiontype = sectiontype
        self.orientation = orientation
        self.behavior = behavior
        self.range = range
        self.addMaterial(material)
        self.addSection(section)

    
    def addSection(self, section):
        """Create or replace the section properties of the element.

        If 'section' is a dict, it will be added to 'the_sections'.
        If 'section' is a string, this string will be used as a key to
        search in 'the_sections'.
        """
        if isinstance(section, str):
            if the_sections.has_key(section):
                self.section = the_sections[section]
            else:
                warning("Section '%s' is not in the database" % section)
        elif isinstance(section,dict):
            # WE COULD ADD AUTOMATIC CALCULATION OF SECTION PROPERTIES
            #self.computeSection(section)
            #print section
            the_sections[section['name']] = CascadingDict(section)
            self.section = the_sections[section['name']]
        elif section==None:
            self.section = section
        else: 
            raise ValueError,"Expected a string or a dict"


    def computeSection(self,section):
        """Compute the section characteristics of specific sections."""
        if not section.has_key('sectiontype'):
            return
        if section['sectiontype'] == 'circ':
            r = section['radius']
            A = pi * r**2
            I = pi * r**4 / 4
            section.update({'cross_section':A,
                            'moment_inertia_11':I,
                            'moment_inertia_22':I,
                            'moment_inertia_12':0.0,
                            'torsional_rigidity':2*I,
                            })
        else:
            raise ValueError,"Invalid sectiontype"
        
    
    def addMaterial(self, material):
        """Create or replace the material properties of the element.

        If the argument is a dict, it will be added to 'the_materials'.
        If the argument is a string, this string will be used as a key to
        search in 'the_materials'.
        """
        if isinstance(material, str) :
            if the_materials.has_key(material):
                self.material = the_materials[material] 
            else:
                warning("Material '%s'  is not in the database" % material)
        elif isinstance(material, dict):
            the_materials[material['name']] = CascadingDict(material)
            self.material = the_materials[material['name']]
        elif material==None:
            self.material=material
        else:
            raise ValueError,"Expected a string or a dict"


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

if __name__ == "script" or  __name__ == "draw":

    workHere()
        
    Mat = MaterialDB('../examples/materials.db')
    setMaterialDB(Mat)
    Sec = SectionDB('../examples/sections.db')
    setSectionDB(Sec)

    Stick=Property(1, {'colour':'green', 'name':'Stick', 'weight': 25, 'comment':'This could be anything: a gum, a frog, a usb-stick,...'})
    author=Property(5,{'Name':'Tim Neels', 'Address':CascadingDict({'street':'Krijgslaan', 'city':'Gent','country':'Belgium'})})
    
#    print Stick
#    print properties[1] 
    
    Stick.weight=30
    Stick.length=10
    print the_properties[1]
    
#    print the_properties[5]
    the_properties[5].street='Voskenslaan'
    print author
    print the_properties[5]
    print author.street
    print author.Address.street
    
    P1 = [ 1.0,1.0,1.0, 0.0,0.0,0.0 ]
    P2 = [ 0.0 ] * 3 + [ 1.0 ] * 3 
    B1 = [ 0.0 ] * 6

    print Mat
    print Sec
    vert = ElemSection('IPEA100', 'steel')
    hor = ElemSection({'name':'IPEM800','A':951247,'I':CascadingDict({'Ix':1542,'Iy':6251,'Ixy':352})}, {'name':'S400','E':210,'fy':400})
    circ = ElemSection({'name':'circle','radius':10,'sectiontype':'circ'},'steel')

    print the_sections
    exit()

    q = ElemLoad(magnitude=2.5, loadlabel='PZ')
    top = ElemProperty(1,hor,[q],'B22')
    column = ElemProperty(2,vert, elemtype='B22')
    diagonal = ElemProperty(4,hor,elemtype='B22')
    print 'the_elemproperties'
    for key, item in the_elemproperties.iteritems():
       print key, item	

    bottom=ElemProperty(3,hor,[q])


    topnode = NodeProperty(1,cload=[5,0,-75,0,0,0])
    foot = NodeProperty(2,bound='pinned')

    np = {}
    np['1'] = NodeProperty(1, cload=P1)
    np['2'] = NodeProperty(2, cload=P2)
    np['3'] = np['2']
    np['3'].bound = B1
    np['1'].cload[1] = 33.0
    np['7'] = NodeProperty(7, bound=B1)

    for key, item in the_materials.iteritems():
        print key, item

    print 'the_properties'
    for key, item in the_properties.iteritems():
        print key, item

    print 'the_nodeproperties'    
    for key, item in the_nodeproperties.iteritems():
        print key, item
    
    print 'the_elemproperties'
    for key, item in the_elemproperties.iteritems():
        print key, item
##        
    print the_elemproperties[3].A
    bottom.A=555
    print the_elemproperties[3]
    print the_elemproperties[3].A
    the_elemproperties[3].A=444
    print bottom.A
    print the_elemproperties[3].A
    
    print "beamsection attributes"
    for key,item in the_elemproperties.iteritems():
        print key,item.elemload
    
    for key,item in the_elemproperties.iteritems():
        print key,item.E
    
    print "cload attributes"
    for key,item in the_nodeproperties.iteritems():
        print key,item.cload

    print "cload attributes"
    for key,item in np.iteritems():
        print key,item.cload

# End

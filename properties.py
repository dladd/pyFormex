#!/usr/bin/env python
# $Id$

"""General framework for attributing properties to Formex elements.

Properties can really be just about any Python object.
Properties are identified and connected to a Formex element by the
prop values that are stored in the Formex.
"""

from flatkeydb import *
from mydict import *

materials = Dict({})
sections = Dict({})

properties = Dict({})
nodeproperties = Dict({})
elemproperties = Dict({})

def readMaterials(database):
    """Import all materials from a database
    
    For now, it can only read databases using flatkeydb.
    """
    mat = FlatDB(['name'], beginrec = 'material', endrec = 'endmaterial')
    mat.readFile(database)
    for key, item in mat.iteritems():#not materials=Dict(mat), because this would erase any material that was already added
        materials[key] = item

def readSections(database):
    """Import all sections from a database
    
    For now, it can only read databases using flatkeydb.
    """
    sect = FlatDB(['name'], beginrec = 'section', endrec = 'endsection')
    sect.readFile('sections.db')
    for key, item in sect.iteritems():
        sections[key] = item


class Property(CascadingDict):
    """A general properties class.

    This class should only provide general methods, such as
    add, change and delete properties, lookup, print, and
    of course, connect properties to Formex elements.
    """

    def __init__(self, nr, data={}):
        """Create a new property. Empty by default.
        
        A property is created and the data is stored in a Dict called 'properties'. 
        The key to access the property is the number.
        This number should be the same as the property number of the Formex element.
        """
        CascadingDict.__init__(self, data)
        properties[nr] = self 


class NodeProperty(Property):
    """Properties related to a single node."""

    def __init__(self, nr, cload = None, bound = None, coords = 'cartesian', coordset=[]):
        """Create a new node property. Empty by default
        
        A node property is created and the data is stored in a Dict called 'nodeproperties'. 
        The key to access the node property is the number.
        This number should be the same as the node property number of the Formex element.
        A node property can hold the following sub-properties:
        - cload : a concentrated load
        - bound : a boundary condition
        - coords: the coordinate system which is used for the definition of cload and bound. There are three options:
        cartesian, spherical and cylindrical
		-coordset: a list of 6 coordinates; the 2 points that specify the transformation 
        """
        if (isinstance(cload,list) and len(cload)==6 or cload==None) and (isinstance(bound,list) and len(bound)==6 or bound==None): 
            CascadingDict.__init__(self, {'cload' : cload, 'bound' : bound, 'coords' : coords, 'coordset' : coordset})
            nodeproperties[nr] = self
        else: 
            print 'A pointload or a boundary condition has to be a list containing 6 items'


class ElemProperty(Property):
    """Properties related to a single element."""
    
    def __init__(self, nr, elemsection = None, elemload = None, elemtype = None): 
        """Create a new element property. Empty by default
        
        An element property is created and the data is stored in a Dict called 'elemproperties'. 
        The key to access the element property is the number.
        This number should be the same as the element property number of the Formex element.
        An element property can hold the following sub-properties:
        - elemsection : the section properties of the element. This is an ElemSection instance.
        - elemload : the loading of the element. This is an ElemLoad instance.
        - elemtype: the type of element that is to be used in the analysis. 
        """    
        CascadingDict.__init__(self, {'elemsection' : elemsection, 'elemload' : elemload, 'elemtype' : elemtype})
        elemproperties[nr] = self

class ElemSection(Property):
    """Properties related to the section of a beam."""

    def __init__(self, section = None, material = None, sectiontype = 'general', orientation = None):  
        """Create a new element section property. Empty by default
        
        An element section property can hold the following sub-properties:
        - section : the section properties of the element. The required data in this dict depends on the sectiontype.
        - material : the element material. 
        - sectiontype: the sectiontype of the element. 
        """    
        CascadingDict.__init__(self,{})
        self.sectiontype = sectiontype
        self.orientation = orientation
        self.addMaterial(material)
        self.addSection(section)
    
    def addSection(self, section):
        """Create or replace the section properties of the element.

		If 'section' is a dict, it will be added to 'sections'.
        If 'section' is a string, this string will be used as a key to search in 'sections'
		'orientation' is a list [First direction cosine, second direction cosine, third direction cosine] of the first beam section 			axis. This allows to change the orientation of the cross-section.
        """
        if isinstance(section, str):
            if sections.has_key(section):
                self.section = sections[section]
            else:
                print "This section is not available in the database"
        elif isinstance(section,dict):
            sections[section['name']] = CascadingDict(section)
            self.section = sections[section['name']]
        elif section==None:
            self.section = section
        else: 
            print "argument needs to be string or dict"
    
    def addMaterial(self, material):
        """Create or replace the material properties of the element. If the argument is a dict, it will be added to 'materials'.
        If the argument is a string, this string will be used as a key to search in 'materials'
        """
        if isinstance(material, str) :
            if materials.has_key(material):
                self.material = materials[material] 
            else:
                print "This material is not available in the database"
        elif isinstance(material, dict):
            materials[material['name']] = CascadingDict(material)
            self.material = materials[material['name']]
        elif material==None:
            self.material=material
        else:
            print "argument needs to be a string or dict"

class ElemLoad(Property):
    """Properties related to the load of a beam."""

    def __init__(self, magnitude = None, loadlabel = None):
        CascadingDict.__init__(self, {'magnitude' : magnitude, 'loadlabel' : loadlabel})





# Test

if __name__ == "__main__":

    readMaterials('materials.db')
    readSections('sections.db')
    Pr1=Property(35, {'colour':'green', 'section':CascadingDict({'I':{'Ix':124,'Iy':65},'A':658}),'comment':'This could be a green pen'})
    P1 = [ 1.0,1.0,1.0, 0.0,0.0,0.0 ]
    P2 = [ 0.0 ] * 3 + [ 1.0 ] * 3 
    B1 = [ 0.0 ] * 6
    S1=ElemSection('IPEA100', 'steel')
    S2=ElemSection({'name':'IPEM800','A':951247,'I':CascadingDict({'Ix':1542,'Iy':6251,'Ixy':352})},{'name':'Steel','E':240})
    BL1=ElemLoad(0.5)
    BL2=ElemLoad(cload=[0.2,1])
    top=ElemProperty(2,S1,BL1)
    bottom=ElemProperty(3,S2,BL2)
    diagonal=ElemProperty(4,S1,elemload=BL2)

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

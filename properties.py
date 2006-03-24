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
    mat = FlatDB(['name'], beginrec = 'material', endrec = 'endmaterial')
    mat.readFile(database)
    for key, item in mat.iteritems():#not materials=Dict(mat), because this would erase any material that was already added
	materials[key] = item

def readSections(database):
    sect = FlatDB(['name'], beginrec = 'section', endrec = 'endsection')
    sect.readFile('sections.db')
    for key, item in sect.iteritems():
	sections[key] = item
    print materials

class Property(CascadingDict):
    """A general properties class.

    This class should only provide general methods, such as
    add, change and delete properties, lookup, print, and
    of course, connect properties to Formex elements.
    """

    def __init__(self, nr, data={}):
        """Create a new property. Empty by default."""
	CascadingDict.__init__(self, data)
        properties[nr] = self 
    
    def __repr__(self):
        """Format a property into a string."""
        s = ""			#"PropertyClass{ default=%s" % self.default
        for i in self.items():
            s += "\n  %s = %s" % i
        return s + "\n"
	#it would be great if every level of Properties would indent...!

class NodeProperty(Property):
    """Properties related to a single node."""

    def __init__(self, nr, cload = None, bound = None, coords = 'cartesian'):
        """Create a new node property. Empty by default

        A node property can hold the following sub-properties:
        - cload : a concentrated load
        - bound : a boundary condition
        - coords: the coordinate system which is used for the definition of cload and bound. There are three options:
        cartesian, spherical and cylindrical
        """
        #controleren of cload, bound is lijst van 6 elementen, vervolgens gecreeerd
        if (isinstance(cload,list) and len(cload)==6 or cload==None) and (isinstance(bound,list) and len(bound)==6 or bound==None): 
            CascadingDict.__init__(self, {'cload' : cload, 'bound' : bound, 'coords' : coords})
            nodeproperties[nr] = self
        else: 
            print 'A pointload or a boundary condition has to be a list containing 6 items'


class ElemProperty(Property):
    """Properties related to a single beam"""

    def __init__(self, nr, elemsection = None, elemload = None, elemtype = None): 
        CascadingDict.__init__(self, {'elemsection' : elemsection, 'elemload' : elemload, 'elemtype' : elemtype})
        elemproperties[nr] = self

class ElemSection(Property):
    """Properties related to the section of a beam."""

    def __init__(self, section = None, material = None, sectiontype = 'general'): #shoudn't be empty! 
   #dict-> make a Dict in 'materials{}'
  #string-> zoeken in 'materials{}'
            # daarna zoeken in database-file 
        CascadingDict.__init__(self,{})
        self.sectiontype = sectiontype
        self.addMaterial(material)
        self.addSection(section)
    
    def addSection(self, section):
        if isinstance(section, str):
            if sections.has_key(section):
                self.section = sections[section]
            else:
                print "This section is not available in the database"
        elif isinstance(section,dict):
            sections[section['name']] = CascadingDict(section)
            self.section = sections[section['name']]
        else: 
            print "argument needs to be string or dict"
    
    def addMaterial(self, material):
        if isinstance(material, str) :
            if materials.has_key(material):
                self.material = materials[material] #like this if you want to call it like 'beamsection.section.A'
            else:
                print "This material is not available in the database"
        elif isinstance(material, dict):
            materials[material['name']] = CascadingDict(material)
            self.material = materials[material['name']]
        else:
            print "argument needs to be a string or dict"

class ElemLoad(Property):
    """Properties related to the load of a beam."""
    # cload kan enkel in knoop -> hier weg??

    def __init__(self, dload = None, cload = None, coords = 'cartesian'):
        """there are three options: cartesian (global), cylindrical and local""" 
        CascadingDict.__init__(self, {'dload' : dload, 'cload' : cload})
#lload=list, 3 elems, cload=? (load+coord-> list [x,y,z,loadx,loady, loadz]? )
# multiple cloads!! -> ?




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

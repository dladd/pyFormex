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

#######################################################
# This should probably be moved to a separate module

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
the_elemproperties = CascadingDict()
the_modelproperties = CascadingDict()


class CoordSystem(object):
    """A class for storing coordinate systems."""

    valid_csys = 'RSC'
    
    def __init__(self,csys,cdata):
        """Create a new coordinate system.

        csys is one of 'Rectangular', 'Spherical', 'Cylindrical'. Case is
          ignored and the first letter suffices.
        cdata is a list of 6 coordinates specifying the two points that
          determine the coordinate transformation 
        """
        try:
            csys = csys[0].upper()
            if not csys in valid_csys:
                raise
            cdata = asarray(cdata).reshape(2,3)
        except:
            raise ValueError,"Invalid initialization data for CoordSystem"
        self.sys = csys
        self.data = cdata


def checkArray1D(a,size=None,kind=None,allow=None):
    """Check that an array a has the correct size and type.

    Either size and or kind can be specified.
    If kind does not match, but is included in allow, conversion to the
    requested type is attempted.
    Returns the array if valid.
    Else, an error is raised.
    """
    try:
        a = asarray(a).ravel()
        if (size is not None and a.size != size):
            raise
        if kind is not None:
            if allow is None and a.dtype.kind != kind:
                raise
            if kind == 'f':
                a = a.astype(float32)
        return a
    except:
        print "Expected size %s, kind %s, got: %s" % (size,kind,a)
    raise ValueError


def checkString(a,valid):
    """Check that a string a has one of the valid values.

    This is case insensitive, and returns the upper case string if valid.
    Else, an error is raised.
    """
    try:
        a = a.upper()
        if a in valid:
            return a
    except:
        print "Expected one of %s, got: %s" % (valid,a)
    raise ValueError


##################### Properties Database ###################    
class PropertiesDB(Dict):
    """A database class for all properties.

    This class collects all properties that can be set on a
    geometrical model.

    This should allow for storing:
       - materials
       - sections
       - node properties
       - elem properties
       - model properties

    Currently, only the former NodeProperties have been converted.
    """

    bound_strings = [ 'XSYMM', 'YSYMM', 'ZSYMM', 'ENCASTRE', 'PINNED' ]

    def __init__(self):
        """Create a new properties database."""
        self.mats = MaterialDB()
        self.sect = SectionDB()
        self.nprop = []
        self.eprop = []
        self.mprop = []
        

    def setMaterialDB(self,aDict):
        """Set the materials database to an external source"""
        if isinstance(aDict,MaterialDB):
            self.mats = aDict


    def setSectionDB(self,aDict):
        """Set the sections database to an external source"""
        if isinstance(aDict,SectionDB):
            self.sect = aDict


    def nodeProp(self,tag=None,nset=None,cload=None,bound=None,displ=None,csys=None):
        """Create a new node property, empty by default.
. 
        A node property can contain any combination of the following fields:
        - tag : an identification tag used to group properties (this is e.g.
                used to flag Step, increment, load case, ...)
        - nset : a single number or a list of numbers identifying the node(s)
                 for which this property will be set
        - cload : a concentrated load
        - bound : a boundary condition
        - displ: a prescribed displacement
        - csys: a coordinate system
        """
        try:
            d = CascadingDict()
            if tag is not None:
                d.tag = str(tag)
            if nset is not None:
                if type(nset) is int:
                    nset = [ int ]
                d.nset = unique1d(nset)
            if cload is not None:
                d.cload = checkArray1D(cload,6,'f','i')
            if bound is not None:
                if type(bound) == str:
                    print "A STRING"
                    d.bound = checkString(bound,self.bound_strings)
                else:
                    print "bound"
                    d.bound = checkArray1D(bound,6,'i')
            if displ is not None:
                d.displ = checkArray1D(displ,6,'f')
            if csys is not None and not isinstance(csys,CoordSystem):
                raise
            d.nr = len(self.nprop)
            self.nprop.append(d)
            #print "Created Node Property %s" % d
            return d
        except:
            print "tag=%s,nset=%s,cload=%s,bound=%s,displ=%s,csys=%s" % (tag,nset,cload,bound,displ,csys)
            raise ValueError,"Invalid Node Property skipped"


    def getProp(self,kind,recs=None,tags=None,attr=[]):
        """Return all properties of type kind matching tag and having attr.

        kind is either 'n', 'e' or 'm'
        If recs is given, it is a list of record numbers or a single number.
        If a tag or a list of tags is given, only the properties having a
        matching tag attribute are returned.
        If a list of attibutes is given, only the properties having those
        attributes are returned.
        """
        prop = getattr(self,kind+'prop')
        if recs is not None:
            if type(recs) != list:
                recs = [ recs ]
            recs = [ i for i in recs if i < len(prop) ]
            prop = [ prop[i] for i in recs ]
        if tags is not None:
            if type(tags) != list:
                tags = [ tags ]
            tags = map(str,tags)   # tags are always converted to strings!
            prop = [ p for p in prop if p.has_key('tag') and p['tag'] in tags ]
        for a in attr:
            prop = [ p for p in prop if p.has_key(a) ]
        return prop


# Used as a transitional global DB
the_P = PropertiesDB()

def NodeProperty(tag,nset=None,cload=None,bound=None,displacement=None,coords=None,coordset=None):
    
    if coords is not None:
        csys = CoordSystem(coords,coordset)
    else:
        csys = None
    return the_P.nodeProp(tag,nset,cload,bound,displacement,csys)


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





# Test

if __name__ == "script" or  __name__ == "draw":

    workHere()
    print os.getcwd()
    P = PropertiesDB()
    Mat = MaterialDB('../examples/materials.db')
    P.setMaterialDB(Mat)
    Sec = SectionDB('../examples/sections.db')
    P.setSectionDB(Sec)

    P.nodeProp(1,cload=[5,0,-75,0,0,0])
    P.nodeProp(2,bound='pinned')
    
    P1 = [ 1.0,1.0,1.0, 0.0,0.0,0.0 ]
    P2 = [ 0.0 ] * 3 + [ 1.0 ] * 3 
    B1 = [ 1 ] + [ 0 ] * 5

    P.nodeProp(1,cload=P1)
    P.nodeProp(2,cload=P2)
    P.nodeProp(7,bound=B1)

    print 'nodeproperties'
    print P.nprop

    print 'all nodeproperties'
    print P.getProp('n')
    
    print "properties 0 and 2"
    for p in P.getProp('n',recs=[0,2]):
        print p
        
    print "tag 1 and 7"
    for p in P.getProp('n',tags=[1,7]):
        print p

    print "cload attributes"
    for p in P.getProp('n',attr=['cload']):
        print p
    exit()

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

    print Mat
    print Sec
    vert = ElemSection('IPEA100', 'steel')
    hor = ElemSection({'name':'IPEM800','A':951247,'I':CascadingDict({'Ix':1542,'Iy':6251,'Ixy':352})}, {'name':'S400','E':210,'fy':400})
    circ = ElemSection({'name':'circle','radius':10,'sectiontype':'circ'},'steel')

    print the_sections

    q = ElemLoad(magnitude=2.5, loadlabel='PZ')
    top = ElemProperty(1,hor,[q],'B22')
    column = ElemProperty(2,vert, elemtype='B22')
    diagonal = ElemProperty(4,hor,elemtype='B22')
    print 'the_elemproperties'
    for key, item in the_elemproperties.iteritems():
       print key, item	

    bottom=ElemProperty(3,hor,[q])


    for key, item in the_materials.iteritems():
        print key, item

    print 'the_properties'
    for key, item in the_properties.iteritems():
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

# End

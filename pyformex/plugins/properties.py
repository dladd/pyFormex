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


## the_materials = MaterialDB()
## the_sections = SectionDB()


class ElemSection(CascadingDict):
    """Properties related to the section of an element."""

    matDB = MaterialDB()
    secDB = SectionDB()


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

        If 'section' is a dict, it will be added to 'self.secDB'.
        If 'section' is a string, this string will be used as a key to
        search in 'self.secDB'.
        """
        if isinstance(section, str):
            if self.secDB.has_key(section):
                self.section = self.secDB[section]
            else:
                warning("Section '%s' is not in the database" % section)
        elif isinstance(section,dict):
            # WE COULD ADD AUTOMATIC CALCULATION OF SECTION PROPERTIES
            #self.computeSection(section)
            #print section
            self.secDB[section['name']] = CascadingDict(section)
            self.section = self.secDB[section['name']]
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

        If the argument is a dict, it will be added to 'self.matDB'.
        If the argument is a string, this string will be used as a key to
        search in 'self.matDB'.
        """
        if isinstance(material, str) :
            if self.matDB.has_key(material):
                self.material = self.matDB[material] 
            else:
                warning("Material '%s'  is not in the database" % material)
        elif isinstance(material, dict):
            self.matDB[material['name']] = CascadingDict(material)
            self.material = self.matDB[material['name']]
        elif material==None:
            self.material=material
        else:
            raise ValueError,"Expected a string or a dict"


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


############ Utility routines #####################

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

class PropertyDB(Dict):
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
            ElemSection.matDB = aDict


    def setSectionDB(self,aDict):
        """Set the sections database to an external source"""
        if isinstance(aDict,SectionDB):
            self.sect = aDict
            ElemSection.secDB = aDict


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


    def nodeProp(self,tag=None,nset=None,cload=None,bound=None,displ=None,csys=None):
        """Create a new node property, empty by default.
. 
        A node property can contain any combination of the following fields:
        - tag: an identification tag used to group properties (this is e.g.
               used to flag Step, increment, load case, ...)
        - nset: a single number or a list of numbers identifying the node(s)
                for which this property will be set
        - cload: a concentrated load
        - bound: a boundary condition
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
            return d.nr
        except:
            print "tag=%s,nset=%s,cload=%s,bound=%s,displ=%s,csys=%s" % (tag,nset,cload,bound,displ,csys)
            raise ValueError,"Invalid Node Property skipped"


    def elemProp(self,tag=None,eset=None,section=None,eltype=None,dload=None,dloadlbl=None): 
        """Create a new element property, empty by default.
        
        An elem property can contain any combination of the following fields:
        - tag: an identification tag used to group properties (this is e.g.
               used to flag Step, increment, load case, ...)
        - eset: a single number or a list of numbers identifying the element(s)
                for which this property will be set
        - eltype: the element type (currently in Abaqus terms). 
        - section: an ElemSection specifying the element section properties.
        - dload: a distributed load on the element: this is a tuple of dload
                 label and magnitude
        """    
        try:
            d = CascadingDict()
            if tag is not None:
                d.tag = str(tag)
            if eset is not None:
                if type(eset) is int:
                    eset = [ int ]
                d.eset = unique1d(eset)
            if eltype is not None:
                d.eltype = eltype.upper()
            if section is not None:
                d.section = section
            if dload is not None:
                d.dload = dload
            d.nr = len(self.eprop)
            self.eprop.append(d)
            return d.nr
        except:
            print "tag=%s,eset=%s,eltype=%s,section=%s,dload=%s" % (tag,eset,eltype,section,dload)
            raise ValueError,"Invalid Node Property skipped"


# Used as a transitional global DB, will disappear in future
the_P = PropertyDB()

def NodeProperty(tag,nset=None,cload=None,bound=None,displacement=None,coords=None,coordset=None):
    
    if coords is not None:
        csys = CoordSystem(coords,coordset)
    else:
        csys = None
    return the_P.nodeProp(tag,nset,cload,bound,displacement,csys)



def ElemProperty(tag,eset=None,elemsection=None,elemtype=None,elemload=None): 
    return the_P.elemProp(tag,eset,elemsection,elemtype,(elemload.loadlabel,elemload.magnitude))


class ElemLoad(CascadingDict):
    """Properties related to the load of a beam."""

    def __init__(self, magnitude = None, loadlabel = None):
        """Create a new element load. Empty by default.
        
        An element load can hold the following sub-properties:
        - magnitude: the magnitude of the distibuted load.
        - loadlabel: the distributed load type label.
        """          
        Dict.__init__(self, {'magnitude' : magnitude, 'loadlabel' : loadlabel})


class ModelProperty(CascadingDict):
    """Properties related to a model."""

    def __init__(self, name, amplitude=None,intprop=None,surface=None,interaction=None,damping=None):
        """Create a new node property. Empty by default.
        
        A model property is created and the data is stored in a Dict called 'modelproperties'. 
        The key to access the model property is the name.
        """
        global the_modelproperties
        CascadingDict.__init__(self, {'amplitude':amplitude,'intprop':intprop,'surface':surface,'interaction':interaction,'damping':damping})
        the_modelproperties[name] = self





# Test

if __name__ == "script" or  __name__ == "draw":

    workHere()
    print os.getcwd()
    
    Mat = MaterialDB('../examples/materials.db')
    Sec = SectionDB('../examples/sections.db')
    P = PropertyDB()
    
    P.setMaterialDB(Mat)
    P.setSectionDB(Sec)

    # node properties
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

    # materials and sections
    ElemSection.matDB = Mat
    ElemSection.secDB = Sec
    
    vert = ElemSection('IPEA100', 'steel')
    hor = ElemSection({'name':'IPEM800','A':951247,'I':CascadingDict({'Ix':1542,'Iy':6251,'Ixy':352})}, {'name':'S400','E':210,'fy':400})
    circ = ElemSection({'name':'circle','radius':10,'sectiontype':'circ'},'steel')

    print "Materials"
    for m in Mat:
        print Mat[m]
        
    print "Sections"
    for s in Sec:
        print Sec[s]

    q = ('PZ',2.5)

    top = P.elemProp(eltype='B22',section=hor,dload=q)
    column = P.elemProp(eltype='B22',section=vert)
    diagonal = P.elemProp(eltype='B22',section=hor)
    bottom = P.elemProp(section=hor,dload=q)


    print 'elemproperties'
    for p in P.eprop:
        print p
    
    print "section properties"
    for p in P.getProp('e',attr=['section']):
        print p.nr

# End

#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.7.1 Release Fri May  9 08:39:30 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""General framework for attributing properties to geometrical elements.

Properties can really be just about any Python object.
Properties can be attributed to a set of geometrical elements.
"""

from pyformex.flatkeydb import FlatDB
from pyformex.mydict import Dict,CDict
from numpy import *

#################################################################
# This first part still needs to be changed.
# It should probably be moved to a separate module

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


class ElemSection(CDict):
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
        CDict.__init__(self,{})
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
            self.secDB[section['name']] = CDict(section)
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
            self.matDB[material['name']] = CDict(material)
            self.material = self.matDB[material['name']]
        elif material==None:
            self.material=material
        else:
            raise ValueError,"Expected a string or a dict"


class ElemLoad(CDict):
    """Distributed loading on an element."""

    def __init__(self,label=None,value=None,amplitude=None):
        """Create a new element load. Empty by default.
        
        An element load can hold the following sub-properties:
        - label: the distributed load type label.
        - value: the magnitude of the distibuted load.
        - amplitude: an amplitude instance.
        """          
        Dict.__init__(self,{'label':label,'value':value,'amplitude':amplitude})


############## Basic property data classes ########################

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
            if not csys in CoordSystem.valid_csys:
                raise
            cdata = asarray(cdata).reshape(2,3)
        except:
            raise ValueError,"Invalid initialization data for CoordSystem"
        self.sys = csys
        self.data = cdata

        
class Amplitude(object):
    """A class for storing an amplitude."""
    
    def __init__(self,data,definition='TABULAR'):
        """Create a new amplitude."""
        if definition == 'TABULAR':
            self.data = checkArray(data,(-1,2),'f','i')
            self.type = definition 
            
###################################################
############ Utility routines #####################

def checkArray(a,shape=None,kind=None,allow=None):
    """Check that an array a has the correct shape and type.

    Either shape and or kind can be specified.
    The dimensions where shape contains a -1 value are not checked. The
    number of dimensions should match, though.
    If kind does not match, but is included in allow, conversion to the
    requested type is attempted.
    Returns the array if valid.
    Else, an error is raised.
    """
    try:
        a = asarray(a)
        shape = asarray(shape)
        w = where(shape >= 0)[0]
        if asarray(a.shape)[w] != shape[w]:
            raise
        if kind is not None:
            if allow is None and a.dtype.kind != kind:
                raise
            if kind == 'f':
                a = a.astype(float32)
        return a
    except:
        print "Expected shape %s, kind %s, got: %s" % (shape,kind,a)
    raise ValueError

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


# Create automatic names for node and element sets

def autoName(base,*args):
    return (base + '_%s' * len(args)) % args 

def Nset(*args):
    return autoName('Nset',*args)

def Eset(*args):
    return autoName('Eset',*args)

#############################################################
##################### Properties Database ###################

     

class PropertyDB(Dict):
    """A database class for all properties.

    This class collects all properties that can be set on a
    geometrical model.

    This should allow for storing:
       - materials
       - sections
       - any properties
       - node properties
       - elem properties
       - model properties (current unused: use unnamed properties)
    """

    bound_strings = [ 'XSYMM', 'YSYMM', 'ZSYMM', 'ENCASTRE', 'PINNED' ]

    def __init__(self):
        """Create a new properties database."""
        Dict.__init__(self)
        self.mats = MaterialDB()
        self.sect = SectionDB()
        self.prop = []
        self.nprop = []
        self.eprop = []
        #self.mprop = []
        

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


    def getProp(self,kind='',rec=None,tag=None,attr=[]):
        """Return all properties of type kind matching tag and having attr.

        kind is either '', 'n', 'e' or 'm'
        If rec is given, it is a list of record numbers or a single number.
        If a tag or a list of tags is given, only the properties having a
        matching tag attribute are returned.
        If a list of attibutes is given, only the properties having those
        attributes are returned.
        """
        prop = getattr(self,kind+'prop')
        if rec is not None:
            if type(rec) != list:
                rec = [ rec ]
            rec = [ i for i in rec if i < len(prop) ]
            prop = [ prop[i] for i in rec ]
        if tag is not None:
            if type(tag) != list:
                tag = [ tag ]
            tag = map(str,tag)   # tags are always converted to strings!
            prop = [ p for p in prop if p.has_key('tag') and p['tag'] in tag ]
        for a in attr:
            prop = [ p for p in prop if p.has_key(a) and p[a] is not None ]
        return prop


    def Prop(self,kind='',tag=None,set=None,setname=None,**kargs):
        """Create a new property, empty by default.

        A property can hold almost anything. It has only two predefined fields:
        - tag: an identification tag used to group properties
        - set: a single number or a list of numbers identifying the geometrical
               elements for wich the property is set, or the name of a
               previously defined set.
        - setname: the name to be used for this set. Default is to use an
               automatically generated name. If setname is specified without
               a set, this is interpreted as a set= field.
        As all properties, a nr field will be added automatically.
        Besides these, any other fields may be defined and will be added
        without check.
        """
        d = CDict()
        # update with kargs first, to make sure tag,set and nr are sane
        d.update(dict(**kargs))
        if tag is not None:
            d.tag = str(tag)
##         if d.has_key('setname'):
##             del d['setname']
        if setname is not None and type(setname) is not str:
            raise ValueError,"setname should be a string"
        if set is None and setname:
            set = setname
        if set is not None:
            if type(set) is str:
                d.set = set
            else:
                if type(set) is int:
                    set = [ set ]
                d.set = unique1d(set)
                if setname:
                    d.setname = setname
        prop = getattr(self,kind+'prop')
        d.nr = len(prop)
        prop.append(d)
        return d


    def nodeProp(self,set=None,tag=None,cload=None,bound=None,displ=None,csys=None):
        """Create a new node property, empty by default.

        A node property can contain any combination of the following fields:
        - tag: an identification tag used to group properties (this is e.g.
               used to flag Step, increment, load case, ...)
        - set: a single number or a list of numbers identifying the node(s)
                for which this property will be set, or a set name
                If None, the property will hold for all nodes.
        - cload: a concentrated load: a list of 6 values
        - bound: a boundary condition: a list of 6 codes (0/1)
        - displ: a prescribed displacement: a list of tuples (dofid,value)
        - csys: a coordinate system
        """
        try:
            d = {}
            if cload is not None:
                d['cload'] = checkArray1D(cload,6,'f','i')
            if bound is not None:
                if type(bound) == str:
                    d['bound'] = checkString(bound,self.bound_strings)
                else:
                    d['bound'] = checkArray1D(bound,6,'i')
            if displ is not None:
                d['displ'] = displ
            if csys is not None:
                if isinstance(csys,CoordSystem):
                    d['csys'] = csys
                else:
                    raise
            return self.Prop(kind='n',tag=tag,set=set,**d)
        except:
            print "tag=%s,set=%s,tag=%s,cload=%s,bound=%s,displ=%s,csys=%s" % (tag,set,cload,bound,displ,csys)
            raise ValueError,"Invalid Node Property"


    def elemProp(self,grp=None,set=None,tag=None,section=None,eltype=None,dload=None): 
        """Create a new element property, empty by default.
        
        An elem property can contain any combination of the following fields:
        - tag: an identification tag used to group properties (this is e.g.
               used to flag Step, increment, load case, ...)
        - set: a single number or a list of numbers identifying the element(s)
                for which this property will be set, or a set name
                If None, the property will hold for all elements.
        - grp: an elements group number (default None). If specified, the
               element numbers given in set are local to the specified group.
               If not, element are global and should match the global numbering
               according to the order in which element groups will be specified
               in the Model.
        - eltype: the element type (currently in Abaqus terms). 
        - section: an ElemSection specifying the element section properties.
        - dload: an ElemLoad specifying a distributed load on the element.
        """    
        try:
            d = {}
            if eltype is not None:
                d['eltype'] = eltype.upper()
            if section is not None:
                d['section'] = section
            if dload is not None:
                d['dload'] = dload
            p = self.Prop(kind='e',tag=tag,set=set,**d)
            if p.eltype is not None and type(p.set) is str:
                raise
            return p
        except:
            raise ValueError,"Invalid Elem Property\n  tag=%s,set=%s,eltype=%s,section=%s,dload=%s" % (tag,set,eltype,section,dload)


##################################### Test ###########################

if __name__ == "script" or  __name__ == "draw":

    if GD.gui:
        workHere()
    print os.getcwd()
    
    P = PropertyDB()

    Stick = P.Prop(color='green',name='Stick',weight=25,comment='This could be anything: a gum, a frog, a usb-stick,...')
    print Stick
    
    author = P.Prop(tag='author',alias='Alfred E Neuman',address=CDict({'street':'Krijgslaan', 'city':'Gent','country':'Belgium'}))

    print P.getProp(tag='author')[0]
    
    Stick.weight=30
    Stick.length=10
    print Stick
    
    print author.street
    author.street='Voskenslaan'
    print author.street
    print author.address.street
    author.address.street = 'Wiemersdreef'
    print author.address.street

    author = P.Prop(tag='author',name='John Doe',address={'city': 'London', 'street': 'Downing Street 10', 'country': 'United Kingdom'})
    print author

    for p in P.getProp(rec=[0,2]):
        print p.name

    for p in P.getProp(tag=['author']):
        print p.name

    for p in P.getProp(attr=['name']):
        print p.nr


    exit()
    Mat = MaterialDB(GD.cfg['pyformexdir']+'/examples/materials.db')
    Sec = SectionDB(GD.cfg['pyformexdir']+'/examples/sections.db')
    P.setMaterialDB(Mat)
    P.setSectionDB(Sec)

    P1 = [ 1.0,1.0,1.0, 0.0,0.0,0.0 ]
    P2 = [ 0.0 ] * 3 + [ 1.0 ] * 3 
    B1 = [ 1 ] + [ 0 ] * 5
    CYL = CoordSystem('cylindrical',[0,0,0,0,0,1])
    # node property on single node
    P.nodeProp(1,cload=[5,0,-75,0,0,0])
    # node property on nodes 2 and 3
    P.nodeProp(set=[2,3],bound='pinned')
    # node property on ALL nodes
    P.nodeProp(cload=P1,bound=B1,csys=CYL)
    # node property whose set will be reused
    nset1 = P.nodeProp(tag='step1',set=[2,3,4],cload=P1).nr
    # node properties with an already named set
    P.nodeProp(tag='step2',set=Nset(nset1),cload=P2)

    print 'nodeproperties'
    print P.nprop

    print 'all nodeproperties'
    print P.getProp('n')
    
    print "properties 0 and 2"
    for p in P.getProp('n',rec=[0,2]):
        print p
        
    print "tags 1 and step1"
    for p in P.getProp('n',tag=[1,'step1']):
        print p

    print "cload attributes"
    for p in P.getProp('n',attr=['cload']):
        print p

    # materials and sections
    ElemSection.matDB = Mat
    ElemSection.secDB = Sec
    
    vert = ElemSection('IPEA100', 'steel')
    hor = ElemSection({'name':'IPEM800','A':951247,'I':CDict({'Ix':1542,'Iy':6251,'Ixy':352})}, {'name':'S400','E':210,'fy':400})
    circ = ElemSection({'name':'circle','radius':10,'sectiontype':'circ'},'steel')

    print "Materials"
    for m in Mat:
        print Mat[m]
        
    print "Sections"
    for s in Sec:
        print Sec[s]


    times = arange(10)
    values = square(times)
    amp = Amplitude(column_stack([times,values]))

    q1 = ElemLoad('PZ',2.5)
    q2 = ElemLoad('PY',3.14,amplitude=amp)


    top = P.elemProp(set=[0,1,2],eltype='B22',section=hor,dload=q1)
    column = P.elemProp(eltype='B22',section=vert)
    diagonal = P.elemProp(eltype='B22',section=hor)
    bottom = P.elemProp(section=hor,dload=q2)


    print 'elemproperties'
    for p in P.eprop:
        print p
    
    print "section properties"
    for p in P.getProp('e',attr=['section']):
        print p.nr


# End

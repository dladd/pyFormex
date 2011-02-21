#!/usr/bin/env pyformex

"""Exporting finite element models in code Aster mesh file format (.mail).

"""

from plugins.fe_abq import fmtData
from plugins.properties import *
from plugins.fe import *
from mydict import Dict,CDict
import pyformex as pf
from datetime import datetime
import os,sys



def astInputNames(job,extension='mail'):
    """Returns corresponding Code Aster input filename.

    job should be a jobname, with or without directory part, but without extension
    
    The extension can be mail or comm.
    
    The jobname is the basename without the extension and the directory part.
    The filename is the abspath of the job with extension.
    """
    jobname = os.path.basename(job)
    filename = os.path.abspath(job)
    if extension in ['mail','comm']:
        filename += '.%s' % extension
    else:
        raise ValueError,"Extension should be mail or comm"
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


def writeNodes(fil,nodes,type,name=None):
    """Write nodal coordinates.
    
    Type can be 2D or 3D.
    """
    if not type in ['2D','3D']:
        raise ValueError,"Type should be 2D or 3D"
    out = 'COOR_%s' % type
    if name is not None:
        out += ' nom = %s' % name
    fil.write('%s\n'% out)
    if type == '2D':
        nodes = nodes[:,:2]        
    nn = nodes.shape[1]
    fmt = 'N%d' + nn*' %14.6e' + '\n'
    for i,n in enumerate(nodes):
        fil.write(fmt % ((i,)+tuple(n)))
    fil.write('FINSF\n')
    fil.write('%\n')


def writeElems(fil,elems,type,name=None,eid=None,eofs=0,nofs=0):
    """Write element group of given type.

    elems is the list with the element node numbers.
    The elements are added to the named element set. 
    The eofs and nofs specify offsets for element and node numbers.
    If eid is specified, it contains the element numbers increased with eofs.
    """
    out = type
    if name is not None:
        out += ' nom = %s' % name
    fil.write('%s\n'% out)
    nn = elems.shape[1]
    if nn < 5:
        fmt = 'M%d' + nn*' N%d' + '\n'
    else:
        fl = nn/4
        fmt = 'M%d' + fl*(4*' N%d' + '\n')
        if nn%4 != 0:
            fmt += (nn%4)*' N%d' + '\n'
    
    if eid is None:
        eid = arange(elems.shape[0])
    else:
        eid = asarray(eid)
    for i,e in zip(eid+eofs,elems+nofs):
        fil.write(fmt % ((i,)+tuple(e)))

    fil.write('FINSF\n')
    fil.write('%\n')


def writeSet(fil,type,name,set):
    """Write a named set of nodes or elements (type=NSET|ELSET)

    `set` is a list of node/element numbers,
    in which case the `ofs` value will be added to them.
    """
    if type == 'NSET':
        fil.write('GROUP_NO nom = %s\n' % name)
        cap = 'N'
    elif type == 'ELSET':
        fil.write('GROUP_MA nom = %s\n' % name)
        cap = 'M'
    else:
        raise ValueError,"Type should be NSET or ELSET"
        
    for i in set:
        fil.write('%s%d\n' % (cap,i))
    fil.write('FINSF\n')
    fil.write('%\n')


def fmtHeadingMesh(text=''):
    """Format the heading of the Code Aster mesh file (.mail)."""
    out = """TITRE
Code Aster mail file created by %s (%s)
%s
FINSF
""" % (pf.Version,pf.Url,text)
    return out


def fmtHeadingComm(text=''):
    """Format the heading of the Code Aster command file (.comm)."""
    out = """#
# Code Aster command file created by %s (%s)
# %s
""" % (pf.Version,pf.Url,text)
    return out


class AstData(object):
    """Contains all data required to write the Code Aster mesh (.mail) and command (.comm) files.
        
    - `model` : a :class:`Model` instance.
    - `prop` : the `Property` database.
    - `steps` : a list of `Step` instances.
    - `res` : a list of `Result` instances.
    - `out` : a list of `Output` instances.
    - `bound` : a tag or alist of the initial boundary conditions.
      The default is to apply ALL boundary conditions initially.
      Specify a (possibly non-existing) tag to override the default.

    """
    
    def __init__(self,model,prop,nprop=None,eprop=None,steps=[],res=[],out=[],bound=None,type='3D'):
        """Create new AstData."""
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
        self.type = type


    def writeMesh(self,jobname=None,group_by_eset=True,group_by_group=False,header=''):
        """Write a Code Aster mesh file (.mail).
        """
        
        # Create the Code Aster mesh file
        if jobname is None:
            jobname,filename = 'Test',None
            fil = sys.stdout
        else:
            jobname,filename = astInputNames(jobname,extension='mail')
            fil = file(filename,'w')
            pf.message("Writing mesh to file %s" % (filename))
        
        fil.write(fmtHeadingMesh("""Model: %s     Date: %s      Created by pyFormex
Script: %s 
%s
""" % (jobname, datetime.now(), pf.scriptName, header)))

        # write coords
        nnod = self.model.nnodes()
        pf.message("Writing %s nodes" % nnod)
        writeNodes(fil,self.model.coords,self.type)


        # write elements
        pf.message("Writing elements")
        telems = self.model.celems[-1]
        nelems = 0
        for p in self.prop.getProp('e',attr=['eltype']):
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
            
            print('Elements of type %s: %s' % (p.eltype,set))
            
            ## TO CHECK !!!!
            ## names given to sets are now automically replaced by Eset_grp ...
            
            setname = esetName(p)
            gl,gr = self.model.splitElems(set)
            elems = self.model.getElems(gr)
            for i,elnrs,els in zip(range(len(gl)),gl,elems):
                grpname = Eset('grp',i)
                subsetname = Eset(p.nr,'grp',i,)
                nels = len(els)
                if nels > 0:
                    pf.message("Writing %s elements from group %s" % (nels,i))
                    writeElems(fil,els,p.eltype,name=subsetname,eid=elnrs)
                    nelems += nels
                    if group_by_eset:
                        writeSet(fil,'ELSET',setname,[subsetname])
                    if group_by_group:
                        writeSet(fil,'ELSET',grpname,[subsetname])
                    
        pf.message("Total number of elements: %s" % telems)
        if nelems != telems:
            pf.message("!! Number of elements written: %s !!" % nelems)


        # write node sets
        pf.message("Writing node sets")
        for p in self.prop.getProp('n',attr=['set']):
            if p.set is not None:
                # set is directly specified
                set = p.set
            elif p.prop is not None:
                # set is specified by nprop nrs
                if self.nprop is None:
                    raise ValueError,"nodeProp has a 'prop' field but no 'nprop'was specified"
                set = where(self.nprop == p.prop)[0]
            else:
                # default is all nodes
                set = range(self.model.nnodes())
                
            setname = nsetName(p)
            writeSet(fil,'NSET',setname,set)


        # write element sets
        pf.message("Writing element sets")
        for p in self.prop.getProp('e',attr=['set'],noattr=['eltype']):
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

            setname = nsetName(p)
            writeSet(fil,'ELSET',setname,set)
            
        
        fil.write('FIN')
            
        if filename is not None:
            fil.close()
        pf.message("Wrote Code Aster mesh file (.mail) %s" % filename)
    

    def writeComm(self,jobname=None,header=''):

        # Create the Code Aster command file
        if jobname is None:
            jobname,filename = 'Test',None
            fil = sys.stdout
        else:
            jobname,filename = astInputNames(jobname,extension='comm')
            fil = file(filename,'w')
            pf.message("Writing command to file %s" % (filename))
        
        fil.write(fmtHeadingComm("""Model: %s     Date: %s      Created by pyFormex
# Script: %s 
# %s
#
""" % (jobname, datetime.now(), pf.scriptName, header)))


        ## TO DO!!!


        if filename is not None:
            fil.close()
        pf.message("Wrote Code Aster command file (.comm) %s" % filename)
  
# End

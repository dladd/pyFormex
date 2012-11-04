# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Interface with tetgen

A collection of functions to read/write tetgen files and to run the
tetgen program

tetgen is a quality tetrahedral mesh generator and a 3D Delaunay triangulator.
See http://tetgen.org
"""
from __future__ import print_function
from coords import *
from connectivity import Connectivity
from mesh import Mesh
from filewrite import *
import utils

import os

filetypes = [ 'poly', 'smesh', 'ele', 'face', 'node' ]

def readNodeFile(fn):
    """Read a tetgen .node file.

    Returns a tuple as described in readNodesBlock.
    """
    fil = open(fn,'r')
    line = skipComments(fil)
    npts,ndim,nattr,nbmark = getInts(line,4)
    return readNodesBlock(fil,npts,ndim,nattr,nbmark)


def readEleFile(fn):
    """Read a tetgen .ele file.

    Returns a tuple as described in readElemsBlock.
    """
    fil = open(fn,'r')
    line = skipComments(fil)
    nelems,nplex,nattr = getInts(line,3)
    return readElemsBlock(fil,nelems,nplex,nattr)


def readFaceFile(fn):
    """Read a tetgen .face file.

    Returns a tuple as described in readFacesBlock.
    """
    fil = open(fn,'r')
    line = skipComments(fil)
    nelems,nbmark = getInts(line,2)
    return readFacesBlock(fil,nelems,nbmark)


def readSmeshFile(fn):
    """Read a tetgen .smesh file.

    Returns an array of triangle elements.
    """
    fil = open(fn,'r')

    # node section.
    line = skipComments(fil)
    npts,ndim,nattr,nbmark = getInts(line,4)
    if npts > 0:
        nodeInfo = readNodesBlock(fil,npts,ndim,nattr,nbmark)
    else:
        # corresponding .node file
        nodeInfo = readNodeFile(utils.changeExt(fn,'.node'))

    # facet section
    line = skipComments(fil)
    nelems,nbmark = getInts(line,2)
    facetInfo = readSmeshFacetsBlock(fil,nelems,nbmark)

    nodenrs = nodeInfo[1]
    if nodenrs.min() == 1 and nodenrs.max()==nodenrs.size:
        elems = facetInfo[0]
        for e in elems:
            elems[e] -= 1

    # We currently do not read the holes and attributes
    
    return nodeInfo[0],facetInfo[0]


def readPolyFile(fn):
    """Read a tetgen .poly file.

    Returns an array of triangle elements.
    """
    fil = open(fn,'r')

    # node section.
    line = skipComments(fil)
    npts,ndim,nattr,nbmark = getInts(line,4)
    if npts > 0:
        nodeInfo = readNodesBlock(fil,npts,ndim,nattr,nbmark)
    else:
        # corresponding .node file
        nodeInfo = readNodeFile(utils.changeExt(fn,'.node'))

    # facet section
    line = skipComments(fil)
    nelems,nbmark = getInts(line,2)
    facetInfo = readFacetsBlock(fil,nelems,nbmark)
    print("NEXT LINE:")
    print(line)
    return nodeInfo[0],facetinfo[0]

    ## s = line.strip('\n').split()
    ## nelems = int(s[0])
    ## elems = fromfile(fil,sep=' ',dtype=int32, count=4*nelems)
    ## elems = elems.reshape((-1,4))
    ## return elems[:,1:]


def readSurface(fn):
    """Read a tetgen surface from a .node/.face file pair.

    The given filename is either the .node or .face file.
    Returns a tuple of (nodes,elems).
    """
    nodeInfo = readNodeFile(utils.changeExt(fn,'.node'))
    nodes = nodeInfo[0]
    print("Read %s nodes" % nodes.shape[0])
    elemInfo = readFaceFile(utils.changeExt(fn,'.face'))
    elems = elemInfo[0]
    print("Read %s elems" % elems.shape[0])
    #if numbers[0] == 1:
    #    elems -= 1 
    return nodes,elems


######### Support functions ####################################


def skipComments(fil):
    """Skip comments and blank lines on a tetgen file.

    Reads from a file until the first non-comment and non-empty line.
    Then returns the non-empty, non-comment line, stripped from possible
    trailing comments.
    Returns None if end of file is reached.
    """
    while True:
        line = fil.readline()
        if len(line) == 0:
            return None      # EOF
        line = stripLine(line)
        if len(line) > 0:
            return line         # non-comment line found


def stripLine(line):
    """Strip blanks, newline and comments from a line of text.

    """
    nc = line.find('#')
    if nc >= 0:
        line = line[:nc] # strip comments
    return line.strip()  # strips blanks and end of line


def getInts(line,nint):
    """Read a number of ints from a line, adding zero for omitted values.

    line is a string with b;anks separated integer values.
    Returns a list of nint integers. The trailing ones are set to zero
    if the strings contains less values.
    """
    s = map(int,line.split())
    if len(s) < nint:
        s.extend([0]*(nint-len(s)))
    return s


def addElem(elems,nrs,e,n,nplex):
    """Add an element to a collection."""
    if nplex not in elems:
        elems[nplex] = []
        nrs[nplex] = []
    elems[nplex].append(e)
    nrs[nplex].append(n)
    

def readNodesBlock(fil,npts,ndim,nattr,nbmark):
    """Read a tetgen nodes block.

    Returns a tuple with:

    - coords: Coords array with nodal coordinates
    - nrs: node numbers
    - attr: node attributes
    - bmrk: node boundary marker

    The last two may be None.
    """
    ndata = 1 + ndim + nattr + nbmark
    data = fromfile(fil,sep=' ',dtype=Float,count=npts*(ndata)).reshape(npts,ndata)
    nrs = data[:,0].astype(int32)
    coords = Coords(data[:,1:ndim+1])
    if nattr > 0:
        attr = data[:,1+ndim:1+ndim+nattr].astype(int32)
    else:
        attr = None
    if nbmark == 1:
        bmark = data[:,-1].astype(int32)
    else:
        bmark = None
    return coords,nrs,attr,bmark


def readElemsBlock(fil,nelems,nplex,nattr):
    """Read a tetgen elems block.

    Returns a tuple with:

    - elems: Connectivity of type 'tet4' or 'tet10'
    - nrs: the element numbers
    - attr: the element attributes

    The last can be None.
    """
    ndata = 1 + nplex + nattr
    data = fromfile(fil,sep=' ',dtype=int32,count=ndata*nelems).reshape(nelems,ndata)
    nrs = data[:,0]
    elems = data[:,1:1+nplex]
    if nattr > 0:
        attr = data[:,1+nplex:]
    else:
        attr = None
    if nplex == 4:
        eltype = 'tet4'
    elif nplex == 10:
        eltype= 'tet10'
    else:
        raise ValueError,"Unknown tetgen .ele plexitude %s" % nplex
    return Connectivity(elems,eltype=eltype),nrs,attr


def readFacesBlock(fil,nelems,nbmark):
    """Read a tetgen faces block.

    Returns a a tuple with:

    - elems: Connectivity of type 'tri3'
    - nrs: face numbers
    - bmrk: face boundary marker

    The last can be None.
    """
    ndata = 1 + 3 + nbmark
    data = fromfile(fil,sep=' ',dtype=int32, count=ndata*nelems).reshape(nelems,ndata)
    nrs = data[:,0]
    elems = data[:,1:4]
    if nbmark == 1:
        bmark = data[:,-1]
    else:
        bmark = None
    return Connectivity(elems,eltype='tri3'),nrs,bmark


def readSmeshFacetsBlock(fil,nfacets,nbmark):
    """Read a tetgen .smesh facets bock.

    Returns a tuple of dictionaries with plexitudes as keys:

    - elems: for each plexitude a Connectivity array
    - nrs: for each plexitude a list of element numbers in corresponding elems
    
    """
    elems = {}
    nrs = {}
    for i in range(nfacets):
        line = fil.readline()
        line = line.strip()
        if len(line) > 0:
            data = fromstring(line,sep=' ',dtype=int32)
            nplex = data[0]
            if nplex > 0:
                e = data[1:1+nplex]
                # bmark currently not read
                addElem(elems,nrs,e,i,nplex)
            else:
                raise ValueError,"Invalid data line:\n%s" % line
            
    for np in elems:
        if np == 3:
            eltype= 'tri3'
        elif np == 4:
            eltype = 'quad4'
        else:
            eltype = None
        elems[np] = Connectivity(elems[np],eltype=eltype)
        nrs[np] = array(nrs[np])
    return elems,nrs


def readNeigh(fn):
    """Read a tetgen .neigh file.

    Returns an arrays containing the tetrahedra neighbours:
    """
    fil = open(fn,'r')
    line = fil.readline()
    s = line.strip('\n').split()
    nelems, nneigh = map(int,s)
    elems = fromfile(fil,sep=' ',dtype=int32).reshape((nelems,nneigh+1))
    return elems[:,1:]


def writeNodes(fn,coords,offset=0):
    """Write a tetgen .node file."""
    coords = asarray(coords).reshape((-1,3))
    fil = open(fn,'w')
    fil.write("%d %d 0 0\n" % coords.shape)
    writeIData(coords,fil,"%f ",ind=offset)
    fil.close()


def writeSmesh(fn,facets,coords=None,holes=None,regions=None):
    """Write a tetgen .smesh file.

    Currently it only writes the facets of a triangular surface mesh.
    Coords should be written independently to a .node file.
    """
    fil = open(fn,'w')
    fil.write("# part 1: node list.\n")
    if coords is None:
        fil.write("0  3  0  0  # coords are found in %s.node.\n")
    fil.write("# part 2: facet list.\n")
    fil.write("%s 0\n" % facets.shape[0])
    for i,n in enumerate(facets):
        # adding comments breaks fast readback
        # fil.write("3 %s %s %s # %s\n" % (n[0],n[1],n[2],i))
        fil.write("3 %s %s %s\n" % (n[0],n[1],n[2]))
    fil.write("# part 3: hole list.\n")
    if holes is None:
        fil.write("0\n")
    fil.write("# part 4: region list.\n")
    if regions is None:
        fil.write("0\n")
    fil.write("# Generated by pyFormex\n")


def writeSurface(fn,coords,elems):
    """Write a tetgen surface model to .node and .smesh files.

    The provided file name is the .node or the .smesh filename.
    """
    writeNodes(utils.changeExt(fn,'.node'),coords)
    writeSmesh(utils.changeExt(fn,'.smesh'),elems)



def nextFilename(fn):
    """Returns the next file name in a family of tetgen file names."""
    m = re.compile("(?P<base>.*)\.(?P<id>\d*)\.(?P<ext>.*)").match(fn)
    if m:
        return "%s.%s.%s" % (m.group('base'),int(m.group('id'))+1,m.group('ext'))
    else:
        return '.1'.join(os.path.splitext(fn))


def runTetgen(fn,options=''):
    """Run tetgen mesher on the specified file.

    The input file is a closed triangulated surface.
    tetgen will generate a volume tetraeder mesh inside the surface,
    and create a new approximation of the surface as a by-product.
    """
    if not utils.hasExternal('tetgen'):
        utils.warn("no_tetgen")
        return

    if os.path.exists(fn) and utils.hasExternal('tetgen'):
        sta,out = utils.runCommand('tetgen -z%s %s' % (options,fn))


def readTetgen(fn):
    """Read and draw a tetgen file.

    This is an experimental function for the geometry import menu.
    """
    res = {}
    base,ext = os.path.splitext(fn)
    if ext == '.node':
        nodes = readNodeFile(fn)[0]
        res['tetgen'+ext] = nodes
    elif ext in [ '.ele', '.face' ]:
        nodes,nodenrs = readNodeFile(utils.changeExt(fn,'.node'))[:2]
        if ext == '.ele':
            elems = readEleFile(fn)[0]
        elif ext == '.face':
            elems = readFaceFile(fn)[0]
        if nodenrs.min() == 1 and nodenrs.max()==nodenrs.size:
            elems = elems-1
        M = Mesh(nodes,elems,eltype=elems.eltype)
        res['tetgen'+ext] = M
        
    elif ext == '.smesh':
        nodes,elems = readSmeshFile(fn)
        ML = [ Mesh(nodes,elems[e]) for e in elems ]
        res = dict([('Mesh-%s'%M.nplex(),M) for M in ML])
        
    elif ext == '.poly':
        nodes,elems = readPolyFile(fn)
        ML = [ Mesh(nodes,elems[e]) for e in elems ]
        res = dict([('Mesh-%s'%M.nplex(),M) for M in ML])

    return res


#################################################################
## Extra TriSurface Methods ##

def checkSelfIntersectionsWithTetgen(self,verbose=False):
    """check self intersections using tetgen

    Returns couples of intersecting triangles
    """
    from plugins.tetgen import writeSurface
    cmd = 'tetgen -d '
    tmp = tempfile.mktemp('')
    print(tmp)
    pf.message("Writing temp file %s" % tmp)
    writeSurface(tmp,self.coords, self.elems)
    if verbose:
        cmd += '-V '
    cmd=cmd+ tmp
    pf.message("Checking with command\n %s" % cmd)
    sta,out = utils.runCommand(cmd)
    if sta:
        pf.message('Tetgen got an error')
        return sta
    try:
        os.remove(tmp+'.node')
        os.remove(tmp+'.smesh')
        os.remove(tmp+'.1.face')
        os.remove(tmp+'.1.node')
    except:
        pass
    if sta or verbose:
        pf.message(out)
    return asarray( [int(l.split(' ')[0]) for l in out.split('acet #')[1:]]).reshape(-1, 2)-1


def meshInsideSurface(self,quality=False):
    d = utils.tempDir()
    fn = os.path.join(d,'surface.off')
    self.write(fn)
    if quality:
        options='-q'
    else:
        options=''
    runTetgen(fn,options)
    fn = os.path.join(d,'surface.1.ele')
    res = readTetgen(fn)
    utils.removeTree(d)
    return res['tetgen.ele']

def install_more_trisurface_methods():
    from plugins.trisurface import TriSurface
    TriSurface.tetmesh = meshInsideSurface
    TriSurface.checkSelfIntersectionsWithTetgen = checkSelfIntersectionsWithTetgen


# End

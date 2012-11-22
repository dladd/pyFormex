#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
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

"""
Interface with Calculix FE input files (.inp).


"""
from __future__ import print_function

#
# DEVS: Try to do this without importing from pyformex
#

import re
import numpy as np
from pyformex.arraytools import *

re_eltypeB = re.compile("^(?P<type>B)(?P<ndim>[23])(?P<degree>\d)?(?P<mod>(OS)?H*)$")
re_eltype = re.compile("^(?P<type>.*?)(?P<ndim>[23]D)?(?P<nplex>\d+)?(?P<mod>[HIMRSW]*)$")

#
# List of known Abaqus/Calculix element type
# 
#
    
abq_elems = [
    'SPRINGA', 
    'CONN3D2','CONN2D2',
    'FRAME3D','FRAME2D',
    'T2D2','T2D2H','T2D3','T2D3H',
    'T3D2','T3D2H','T3D3','T3D3H',
    'B21', 'B21H','B22','B22H','B23','B23H',
    'B31', 'B31H','B32','B32H','B33','B33H',
    'M3D3',
    'M3D4','M3D4R',
    'M3D6','M3D8',
    'M3D8R',
    'M3D9','M3D9R',
    'CPS3',
    'CPS4','CPS4I','CPS4R',
    'CPS6','CPS6M',
    'CPS8','CPS8R','CPS8M',
    'CPE3','CPE3H',
    'CPE4','CPE4H','CPE4I','CPE4IH','CPE4R','CPE4RH',
    'CPE6','CPE6H','CPE6M','CPE6MH',
    'CPE8','CPE8H','CPE8R','CPE8RH',
    'CPEG3','CPEG3H',
    'CPEG4','CPEG4H','CPEG4I','CPEG4IH','CPEG4R','CPEG4RH',
    'CPEG6','CPEG6H','CPEG6M','CPEG6MH',
    'CPEG8','CPEG8H','CPEG8R','CPEG8RH',
    'CAX6','CAX8','CAX8R',
    'S3','S3R', 'S3RS',
    'S4','S4R', 'S4RS','S4RSW','S4R5',
    'S8R','S8R5',
    'S9R5',
    'STRI3',
    'STRI65', 
    'SC8R',
    'SFM3D3',
    'SFM3D4','SFM3D4R',
    'SFM3D6',
    'SFM3D8','SFM3D8R',
    'C3D4','C3D4H',
    'C3D6','C3D6H',
    'C3D8','C3D8I','C3D8H','C3D8R','C3D8RH','C3D10',
    'C3D10H','C3D10M','C3D10MH',
    'C3D15','C3D15H',
    'C3D20','C3D20H','C3D20R','C3D20RH',
    'R2D2','RB2D2','RB3D2','RAX2','R3D3','R3D4',
    ]


def abq_eltype(eltype):
    """Analyze an Abaqus element type and return eltype characteristics.

    Returns a dictionary with:

    - type: the element base type
    - ndim: the dimensionality of the element
    - nplex: the plexitude (number of nodes)
    - mod: a modifier string

    Currently, all these fields are returned as strings. We should probably
    change ndim and nplex to an int.
    """
    if eltype.startswith('B'):
        m = re_eltypeB.match(eltype)
    else:
        m = re_eltype.match(eltype)
    if m:
        d = m.groupdict()
        try:
            nplex = int(d['nplex'])
        except:
            if 'degree' in d:
                degree = int(m.group('degree'))
                if degree == 2:
                    nplex = 3
                else:
                    nplex = 2
            elif d['type'] == 'FRAME':
                nplex = 2
            elif d['type'] == 'SPRINGA':
                nplex = 1
            else:
                nplex = 0
        d['nplex'] = nplex
        if 'ndim' not in d or d['ndim'] is None:
            if d['type'][:2] in ['CP','CA'] :
                d['ndim'] = '2'
        if d['type'] in ['R'] and d['nplex']==4:
            d['ndim'] = '2'
        try:
            ndim = int(d['ndim'][0])
        except:
            ndim = 3
        d['ndim'] = ndim
        if 'mod' not in d or d['mod'] is None:
            d['mod'] = ''
        d['avail'] = 'A'   # Available in Abaqus
        d['pyf'] = pyf_eltype(d)
    else:
        d = {}
    return d


known_eltypes = {
    1: { 'point': [ 'SPRINGA', ] },
    2: { 'line2': [ 'CONN', 'FRAME', 'T', 'B', 'RB', 'RAX',] },
    3: { 'line3': [ 'B', ],
         'tri3':  [ 'M', 'CPS', 'CPE', 'CPEG', 'S', 'SFM', 'R', ] },
    4: { 'quad4': [ 'M', 'CPS', 'CPE', 'CPEG', 'S', 'SFM', 'R', ],
         'tet4':  [ 'C', ] },
    6: { '':  [ 'M', 'CPS', 'CPE', 'CPEG', 'SFM', ],
         'wedge6':[ 'C', ] },
    8: { 'quad8': [ 'M', 'CPS', 'CPE', 'CPEG', 'CAX', 'S', 'SFM', ],
         'hex8':  [ 'C', ] },
    9: { 'quad9': [ 'M', 'S' ] },
    10:{ 'tet10': [ 'C', ] },
    15:{ '': [ 'C', ] },
    20:{ 'hex20': [ 'C', ] },
    }

pyf_eltypes = {
    1:  'point',
    2:  'line2',
    3:  { 2: 'line3', 3: 'tri3'   },
    4:  { 2: 'quad4', 3: 'tet4'   },
    6:  { 2: ''     , 3: 'wedge6' },
    8:  { 2: 'quad8', 3: 'hex8'   },
    9:  'quad9',
    10: 'tet10',
    15: '',
    20: 'hex20',
    }


def pyf_eltype(d):
    """Return the best matching pyFormex element type for an abq/ccx element

    d is an element groupdict obtained by scanning the element name.
    """
    eltype = pyf_eltypes.get(d['nplex'],'')
    if isinstance(eltype,dict):
        eltype = eltype[d['ndim']]
    return eltype
    

def print_catalog():
    for el in abq_elems:
        d = abq_eltype(el)
        if d:
            print("Eltype %s = Type %s, ndim %s, nplex %s, mod %s, pyf_type %s" % (el,d['type'],d['ndim'],d['nplex'],d['mod'],d['pyf']))
        else:
            print("No match: %s" % el)

    
print_catalog()
#
#  TODO: S... and RAX elements are still scanned wrongly
#

class InpModel(object):
    pass

model = None
system = None
skip_unknown_eltype = False
log = None
part = None


def startPart(name):
    """Start a new part."""
    global part
    print("Start part %s" % name)
    model.parts.append({'name':name})
    part = model.parts[-1]


def readCommand(line):
    """Read a command line, return the command and a dict with options"""
    if line[0] == '*':
        line = line[1:]
    s = line.split(',')
    s = [si.strip() for si in s]
    cmd = s[0]
    opts = {}
    for si in s[1:]:
        kv = si.split('=')
        k = kv[0]
        if len(kv) > 1:
            v = kv[1]
        else:
            v = True
        opts[k] = v
    return cmd,opts


def do_HEADING(opts,data):
    """Read the nodal data"""
    model.heading = '\n'.join(data)


def do_PART(opts,data):
    """Set the part name"""
    startPart(opts['NAME'])
    

def do_SYSTEM(opts,data):
    """Read the system data"""
    global system
    if len(data) == 0:
        system = None
        return
    
    s = data[0].split(',')
    A = map(float,s[:3])
    try:
        B = map(float,s[3:])
    except:
        B,C = None,None
    if len(data) > 1:
        C = map(float,data[1].split(''))
    else:
        B[2] = 0.
        C = [ -B[1], B[0], 0. ]
    t = array(A)
    if B is None:
        r = None
    else:
        r = rotmat(array([A,B,C]))
    system = (t,r)
    

def do_NODE(opts,data):
    """Read the nodal data"""
    nnodes = len(data)
    print("Read %s nodes" % nnodes)
    ndata = 4
    data = ','.join(data)
    with open('%s-NODE.data'%part['name'],'w') as f:
        f.write(data)

    x = np.fromstring(data,dtype=np.float32,count=ndata*nnodes,sep=',').reshape(-1,ndata)
    nodid = x[:,0].astype(np.int32)
    coords = x[:,1:]

    if system:
        t,r = system
        if r is not None:
            coords = dot(coords,r)
        coords += t

    if 'coords' in part:
        part['nodid'] = concatenate([part['nodid'],nodid])
        part['coords'] = concatenate([part['coords'],coords],axis=0)
    else:
        part['nodid'] = nodid
        part['coords'] = coords


def do_ELEMENT(opts,data):
    """Read element data"""
    d = abq_eltype(opts['TYPE'])
    eltype = d['pyf']
    if not eltype:
        if skip_unknown_eltype:
            return
        else:
            raise ValueError,"Element type '%s' can not yet be imported" % opts['TYPE']
    nplex = d['nplex']
    nelems = len(data)
    print("Read %s elements of type %s, plexitude %s" % (nelems,eltype,nplex))
    ndata = nplex+1
    data = ','.join(data)
    with open('%s-ELEMENT.data'%part['name'],'w') as f:
        f.write(data)
    e = np.fromstring(data,dtype=np.int32,count=ndata*nelems,sep=',').reshape(-1,ndata)
    elid = e[:,0]
    elems = e[:,1:]
    if not 'elems' in part:
        part['elems'] = []
    if not 'elid' in part:
        part['elid'] = []
    part['elems'].append((eltype,elems))
    part['elid'].append(elid)


def endCommand(cmd,opts,data):
    global log
    func = 'do_%s' % cmd
    if func in globals():
        globals()[func](opts,data)
    else:
        #print("Data %s" % data)
        log.write("Don't know how to handle keyword '%s'\n" % cmd)


def readInput(fn):
    """Read an input file (.inp)"""
    global line,part,log,model
    model = InpModel()
    model.parts = []
    startPart('DEFAULT')
    cmd = ''
    logname = fn.replace('.inp','ccxinp.log')
    with open(logname,'w') as log:
        with open(fn) as fil:
            for line in fil:
                if len(line) == 0:
                    break
                line = line.upper()
                if line.startswith('*'):
                    if cmd:
                        endCommand(cmd,opts,data)
                        cmd = ''
                    if line[1] != '*':
                        data = []
                        cmd,opts = readCommand(line[1:])
                        log.write("Keyword %s; Options %s\n" % (cmd,opts))
                        data_cont = False
                else:
                    line = line.strip()
                    if data_cont:
                        data[-1] += line
                    else:
                        data.append(line)
                    data_cont = line.endswith(',')

    print("Number of parts in model: %s" % len(model.parts))
    return model


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        readInput(a)
        print(model)

# End

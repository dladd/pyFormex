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
"""Operations on triangulated surfaces.

A triangulated surface is a surface consisting solely of triangles.
Any surface in space, no matter how complex, can be approximated with
a triangulated surface.
"""
from __future__ import print_function

import pyformex as pf

## from formex import *
## from connectivity import Connectivity,connectedLineElems,adjacencyArrays
## from mesh import Mesh
## import mesh_ext  # load the extended Mesh functions

## import geomtools
## import inertia
## import fileread,filewrite
## from gui.drawable import interpolateNormals

## import os,tempfile
## import tempfile

import utils

from olist import List

class WebGL(List):
    """_A 3D geometry model for export to WebGL.

    UNFINISHED! DO NOT USE
    """
    def __init__(self,data=[]):
        List.__init__(self,data)
        self.script = "http://get.goXTK.com/xtk_edge.js"

    def format_object(self,obj):
        if hasattr(obj,'name'):
            name = obj.name
            s = "var %s = new X.mesh();\n" % name
        else:
            return ''
        if hasattr(obj,'file'):
            s += "%s.file = '%s';\n" % (name,obj.file)
        if hasattr(obj,'caption'):
            s += "%s.capion = '%s';\n" % (name,obj.caption)
        if hasattr(obj,'magicmode'):
            s += "%s.magicmode = '%s';\n" % (name,str(bool(obj.magicmode)))
        s += "r.add(%s);\n" % name
        return s

    def export(self,name,title=None):
        if title is None:
            title = name
        title = '%s --- Created by pyFormex' % title

        jstext = """window.onload = function() {
var r = new X.renderer3D();
r.init();
""" + '\n'.join([self.format_object(o) for o in self ]) + """
r.render();
};
"""
        jsname = utils.changeExt(name,'.js')
        with open(jsname,'w') as jsfile:
            jsfile.write(jstext)

        htmltext = """<html>
<head>
<title>%s</title>
<script type="text/javascript" src="%s"></script>
<script type="text/javascript" src="%s"></script>
</head>
<body>
</body>
</html>
""" % (title,self.script,jsname)
        htmlname = utils.changeExt(jsname,'.html')
        with open(htmlname,'w') as htmlfile:
            htmlfile.write(htmltext)

    
def stl2webgl(stlname,caption=None,magicmode=True,script="http://get.goXTK.com/xtk_edge.js"):
    """Create a WebGL model from an STL file

    - `stlname`: name of an (ascii or binary) STL file
    - `caption`: text to use as caption
    """
    if caption is None:
        caption = stlname
    caption = '%s --- Created by pyFormex' % caption
    
    jstext = """window.onload = function() {
var r = new X.renderer3D();
r.init();
var m = new X.mesh();
m.file = '%s';
m.magicmode = %s;
m.caption = '%s';
r.add(m);
r.render();
};
""" % (stlname,str(bool(magicmode)).lower(),caption)
    jsname = utils.changeExt(stlname,'.js')
    with open(jsname,'w') as jsfile:
        jsfile.write(jstext)
        jsfile.close()
    
    htmltext = """<html>
<head>
<title>%s</title>
<script type="text/javascript" src="%s"></script>
<script type="text/javascript" src="%s"></script>
</head>
<body>
</body>
</html>
""" % (caption,script,jsname)
    htmlname = utils.changeExt(jsname,'.html')
    with open(htmlname,'w') as htmlfile:
        htmlfile.write(htmltext)


def surface2webgl(S,name,caption=None):
    """Create a WebGL model of a surface

    - `S`: TriSurface
    - `name`: basename of the output files
    - `caption`: text to use as caption
    """
    stlname = utils.changeExt(name,'.stl')
    scale = 50./S.sizes().max()
    S.scale(scale).write(stlname,'stlb')
    stl2webgl(stlname,caption)



# End

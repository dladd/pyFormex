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
"""View and manipulate 3D models in your browser.

This module defines some classes and function to help with the creation
of WebGL models. A WebGL model can be viewed directly from a compatible
browser (see http://en.wikipedia.org/wiki/WebGL).

A WebGL model typically consists out of an HTML file and a Javascript file,
possibly also some geometry data files. The HTML file is loaded in the
browser and starts the Javascript program, responsible for rendering the
WebGL scene. 
"""
from __future__ import print_function

import pyformex as pf
import utils
from olist import List
from mydict import Dict
    

class WebGL(List):
    """A 3D geometry model for export to WebGL.

    The WebGL class provides a limited model to be easily exported as
    a complete WebGL model, including the required HTML, Javascript
    and data files.

    Currently the following features are included:

    - create a new WebGL model
    - add Geometry to the model (including color and transparency)
    - set the camera position
    - export the model

    An example of its usage can be found in the WebGL example.

    The create model uses the XTK toolkit from http://www.goXTK.com.
    """
    
    def __init__(self):
        """Create a new (empty) WebGL model."""
        List.__init__(self)
        self.script = "http://get.goXTK.com/xtk_edge.js"
        self.camera = None
        

    def add(self,**kargs):
        """Add a geometry object to the model.

        Currently, two types of objects can be added: pyFormex Geometry
        objects and file names. Geometry objects should be convertible
        to TriSurface (using their toSurface method). Geometry files
        should be in STL format.

        The following keyword parameters are available and all optional:

        - `obj=`: specify a pyFormex Geometry object
        - `file=`: specify a geometry data file (STL). If no `obj` is
          specified, the file should exist. If an `obj` file is specified,
          this is the name that will be used to export the object.
        - `name=`: specify a name for the object. The name will be used
          as a variable in the Javascript script and as filename for for
          export if an `obj` was specified but no `file` was given.
          It should only contain alphanumeric characters and not start with
          a digit.
        - `caption=`: specify a caption to be used as a tooltip when the
          mouse hovers over the object.
        - `color=`: specify a color to be sued for the object. The color
          should be a list of 3 values in the range 0..1 (OpenGL color).
        - `opacity=`: specify a value for the opacity of the object (the
          'alpha' value in pyFormex terms).
        - `magicmode=`: specify True or False. If magicmode is True, colors
          will be set from the normals of the object. This is incompatible
          with `color=`.
        """
        if not 'name' in kargs:
            kargs['name'] = 'm%s' % len(self)
        if 'obj' in kargs:
            try:
                obj = kargs['obj'].toSurface()
            except:
                return
            if obj:
                if not 'file' in kargs:
                    kargs['file'] = '%s.stl' % kargs['name'] 
                obj.write(kargs['file'],'stlb')
        if 'file' in kargs:
            self.append(Dict(kargs))
        else:
            print("Not added because no file:",kargs)

    def view(self,**kargs):
        """Set the camera position and direction.

        This takes two (optional) keyword parameters:

        - `position=`: specify a list of 3 coordinates. The camera will
          be positioned at that place, and be looking at the origin.
          This should be set to a proper distance from the scene to get
          a decent result on first display.
        - `upvector=': specify a list of 3 components of a vector indicating
          the upwards direction of the camera. The default is [0.,1.,0.].
        """
        self.camera = Dict(kargs)

    def format_object(self,obj):
        """Export an object in XTK Javascript format"""
        if hasattr(obj,'name'):
            name = obj.name
            s = "var %s = new X.mesh();\n" % name
        else:
            return ''
        if hasattr(obj,'file'):
            s += "%s.file = '%s';\n" % (name,obj.file)
        if hasattr(obj,'caption'):
            s += "%s.caption = '%s';\n" % (name,obj.caption)
        if hasattr(obj,'color'):
            s += "%s.color = %s;\n" % (name,list(obj.color))
        if hasattr(obj,'alpha'):
            s += "%s.opacity = %s;\n" % (name,obj.alpha)
        if hasattr(obj,'magicmode'):
            s += "%s.magicmode = '%s';\n" % (name,str(bool(obj.magicmode)))
        s += "r.add(%s);\n" % name
        return s

    def export(self,name,title=None,description=None,keywords=None,author=None):
        """Export the WebGL scene.

        Parameters:

        - `name`: a string that will be used for the filenames of the
          HTML, JS and STL files.
        - `title`: an optional title to be set in the .html file. If not
          specified, the `name` is used.

        You can also set the meta tags 'description', 'keywords' and
        'author' to be included in the .html file. The first two have
        defaults if not specified.
        """
        if title is None:
            title = '%s WebGL example, created by pyFormex' % name
        if description is None:
            description = title
        if keywords is None:
            keywords = "pyFormex, WebGL, XTK, HTML, JavaScript"

        s = """window.onload = function() {
var r = new X.renderer3D();
r.init();

"""
        s += '\n'.join([self.format_object(o) for o in self ])

        if self.camera:
            if 'position' in self.camera:
                s +=  "r.camera.position = %s;\n" % list(self.camera.position)
            if 'up' in self.camera:
                s +=  "r.camera.up = %s;\n" % list(self.camera.up)

        s += """
r.render();
};
"""
        jsname = utils.changeExt(name,'.js')
        with open(jsname,'w') as jsfile:
            jsfile.write(s)

        # TODO: setting DOCTYTPE makes browser initial view not good
        # s = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
        s = """<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<meta name="generator" content="%s">
<meta name="description" content="%s">
<meta name="keywords" content="%s">
""" % (pf.FullVersion(),description,keywords)
        if author:
            s += '<meta name="author" content="%s">\n' % author
        s += """<title>%s</title>
<script type="text/javascript" src="%s"></script>
<script type="text/javascript" src="%s"></script>
</head>
<body>
</body>
</html>
""" % (title,self.script,jsname)
        htmlname = utils.changeExt(jsname,'.html')
        with open(htmlname,'w') as htmlfile:
            htmlfile.write(s)


def surface2webgl(S,name,caption=None):
    """Create a WebGL model of a surface

    - `S`: TriSurface
    - `name`: basename of the output files
    - `caption`: text to use as caption
    """
    W = WebGL()
    W.add(obj=S)
    s = S.dsize()
    W.view(position=[0.,0.,s])
    W.export(name,caption)


# End

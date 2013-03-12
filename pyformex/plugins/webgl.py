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
from gui import colors
import utils
from olist import List, intersection
from mydict import Dict
import os
from arraytools import checkFloat,checkArray,checkInt


# Formatting a controller for an attribute
#   %N will be replaced with name of object,
#   %A will be replaced with name of attribute,

controller_format = {
    'visible': "add(%N,'%A')",
    'opacity': "add(%N,'%A',0,1)",
    'color': "addColor(%N,'%A')"
}


def saneSettings(k):
    """Sanitize sloppy settings for JavaScript output"""
    ok = {}
    try:
        ok['color'] = checkArray(k['color'],(3,),'f')
    except:
        try:
            c = checkInt(k['color'][0])
            print("COLOR INDEX %s" % c)
            colormap = pf.canvas.settings.colormap
            ok['color'] = colormap[c % len(colormap)]
        except:
            print("Unexpected color: %s" % k['color'])

    try:
        ok['alpha'] = checkFloat(k['alpha'],0.,1.)
    except:
        pass
    try:
        ok['caption'] = str(k['caption'])
    except:
        pass
    try:
        ok['control'] = intersection(k['control'],controller_format.keys())
    except:
        pass
    return ok


def properties(o):
    """Return properties of an object

    properties are public attributes (not starting with an '_') that
    are not callable.
    """
    keys = [ k for k in sorted(dir(o)) if not k.startswith('_') and not callable(getattr(o,k)) ]
    return utils.selectDict(o.__dict__,keys)



class WebGL(List):
    """A 3D geometry model for export to WebGL.

    The WebGL class provides a limited model to be easily exported as
    a complete WebGL model, including the required HTML, Javascript
    and data files.

    Currently the following features are included:

    - create a new WebGL model
    - add the current scene to the model
    - add Geometry to the model (including color and transparency)
    - set the camera position
    - export the model

    An example of its usage can be found in the WebGL example.

    The create model uses the XTK toolkit from http://www.goXTK.com.
    """

    def __init__(self,name='Scene1'):
        """Create a new (empty) WebGL model."""
        List.__init__(self)
        self._camera = None
        if pf.cfg['webgl/devel']:
            self.scripts = [
                os.path.join(pf.cfg['webgl/devpath'],'lib/closure-library/closure/goog/base.js'),
                os.path.join(pf.cfg['webgl/devpath'],'xtk-deps.js')
                ]
        else:
            self.scripts = [ pf.cfg['webgl/script'] ]
        self.gui = []
        self.name = str(name)


    def objdict(self,clas=None):
        """Return a dict with the objects in this model.

        Returns a dict with the name:object pairs in the model. Objects
        that have no name are disregarded.
        """
        obj = [ o for o in self if hasattr(o,'name') ]
        if clas:
            obj = [ o for o in obj if isinstance(o,clas) ]
        print("OBJDICT: %s" % len(obj))
        print([type(o) for o in obj])
        print(obj)
        return obj


    def addScene(self):
        """Add the current OpenGL scene to the WebGL model.

        This method add all the geometry in the current viewport to
        the WebGL model.
        """
        cv = pf.canvas
        print("Exporting %s actors from current scene" % len(cv.actors))
        for i,a in enumerate(cv.actors):
            o = a.object
            #print("OBJDICT = %s" % sorted(dir(o)))
            atype = type(a).__name__
            otype = type(o).__name__
            print("Actor %s: %s %s Shape=(%s,%s) Color=%s"% (i,atype,otype,o.nelems(),o.nplex(),a.color))
            kargs = properties(o)
            kargs.update(properties(a))
            kargs = saneSettings(kargs)
            print("  Exporting with settings %s" % kargs)
            self.add(obj=o,**kargs)
        ca = cv.camera
        self.camera(focus=[0.,0.,0.],position=ca.eye-ca.focus,up=ca.upVector())


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
        - `control=`: a list of attributes that get a gui controller
        """
        if not 'name' in kargs:
            kargs['name'] = 'm%s' % len(self)
        if 'obj' in kargs:
            # A pyFormex object.
            try:
                obj = kargs['obj']
                obj = obj.toMesh()
                print("LEVEL:%s" % obj.level())
                if obj.level() == 3:
                    print("TAKING BORDER")
                    obj = obj.getBorderMesh()
                obj = obj.toSurface()
            except:
                print("Not added because not convertible to TriSurface : %s",obj)
                return
            if obj:
                if not 'file' in kargs:
                    kargs['file'] = '%s_%s.stl' % (self.name,kargs['name'])
                obj.write(kargs['file'],'stlb')
        elif 'file' in kargs:
            # The name of an STL file
            fn = kargs['file']
            if fn.endswith('.stl') and os.path.exists(fn):
                # We should copy to current directory!
                pass
            else:
                return
        # OK, we can add it
        self.append(Dict(kargs))
        if 'control' in kargs:
            # Move the 'control' parameters to gui
            self.gui.append((kargs['name'],kargs.get('caption',''),kargs['control']))
            del kargs['control']
        elif pf.cfg['webgl/autogui']:
            # Add autogui
            self.gui.append((kargs['name'],kargs.get('caption',''),controller_format.keys()))


    def camera(self,**kargs):
        """Set the camera position and direction.

        This takes two (optional) keyword parameters:

        - `position=`: specify a list of 3 coordinates. The camera will
          be positioned at that place, and be looking at the origin.
          This should be set to a proper distance from the scene to get
          a decent result on first display.
        - `upvector=': specify a list of 3 components of a vector indicating
          the upwards direction of the camera. The default is [0.,1.,0.].
        """
        self._camera = Dict(kargs)


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


    def format_gui_controller(self,name,attr):
        """Format a single controller"""
        if attr in controller_format:
            return controller_format[attr].replace('%N',name).replace('%A',attr)
        else:
            raise ValueError,"Controller for attribute '%s' not implemented"


    def format_gui(self):
        """Create the controller GUI script"""
        s = """
r.onShowtime = function() {
var gui = new dat.GUI();
"""
        for name,caption,attrs in self.gui:
            guiname = "gui_%s" % name
            if not caption:
                caption = name
            s += "var %s = gui.addFolder('%s');\n" % (guiname,caption)
            for attr in attrs:
                cname = "%s_%s" % (guiname,attr)
                s += "var %s = %s.%s;\n" % (cname,guiname,self.format_gui_controller(name,attr))
            #s += "%s.open();\n" % guiname


        # add camera gui
        guiname = "gui_camera"
        s += """
var %s = gui.addFolder('Camera');
var %s_reset = %s.add(r.camera,'reset');
""".replace('%s',guiname)


        s += "}\n\n"
        return s


    def exportPGF(self,fn,sep=''):
        """Export the current scene to a pgf file"""
        from plugins.geometry_menu import writeGeometry
        res = writeGeometry(self.objdict(),fn,sep=sep)
        return res


    def export(self,name=None,title=None,description=None,keywords=None,author=None,createdby=False):
        """Export the WebGL scene.

        Parameters:

        - `name`: a string that will be used for the filenames of the
          HTML, JS and STL files.
        - `title`: an optional title to be set in the .html file. If not
          specified, the `name` is used.

        You can also set the meta tags 'description', 'keywords' and
        'author' to be included in the .html file. The first two have
        defaults if not specified.

        Returns the name of the exported htmlfile.
        """
        if name is None:
            name = self.name
        if title is None:
            title = '%s WebGL example, created by pyFormex' % name
        if description is None:
            description = title
        if keywords is None:
            keywords = "pyFormex, WebGL, XTK, HTML, JavaScript"

        s = """// Script generated by %s

window.onload = function() {
var r = new X.renderer3D();
r.init();

""" % pf.fullVersion()
        s += '\n'.join([self.format_object(o) for o in self ])
        if self.gui:
            s += self.format_gui()
        if self._camera:
            if 'position' in self._camera:
                s +=  "r.camera.position = %s;\n" % list(self._camera.position)
            if 'focus' in self._camera:
                s +=  "r.camera.focus = %s;\n" % list(self._camera.focus)
            if 'up' in self._camera:
                s +=  "r.camera.up = %s;\n" % list(self._camera.up)
        s += """
r.render();
};
"""
        jsname = utils.changeExt(name,'.js')
        with open(jsname,'w') as jsfile:
            jsfile.write(s)
        print("Exported WebGL script to %s" % os.path.abspath(jsname))

        # TODO: setting DOCTYTPE makes browser initial view not good
        # s = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
        s = """<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<meta name="generator" content="%s">
<meta name="description" content="%s">
<meta name="keywords" content="%s">
""" % (pf.fullVersion(),description,keywords)
        if author:
            s += '<meta name="author" content="%s">\n' % author
        s += "<title>%s</title>\n" % title

        if self.gui:
            self.scripts.append(pf.cfg['webgl/guiscript'])
        self.scripts.append(jsname)

        for scr in self.scripts:
            s += '<script type="text/javascript" src="%s"></script>\n' % scr

        s += """
</head>
<body>"""
        if createdby:
            if type(createdby) is int:
                width = ' width="%s%%"' % createdby
            else:
                width = ''
            s += """<div id='pyformex' style='position:absolute;top:10px;left:10px;'>
<a href='http://pyformex.org' target=_blank><img src='http://pyformex.org/images/pyformex_createdby.png' border=0%s></a>
</div>""" % width
        s += """</body>
</html>
"""
        htmlname = utils.changeExt(jsname,'.html')
        with open(htmlname,'w') as htmlfile:
            htmlfile.write(s)
        print("Exported WebGL model to %s" % os.path.abspath(htmlname))

        return htmlname


def surface2webgl(S,name,caption=None):
    """Create a WebGL model of a surface

    - `S`: TriSurface
    - `name`: basename of the output files
    - `caption`: text to use as caption
    """
    W = WebGL()
    W.add(obj=S,file=name)
    s = S.dsize()
    W.view(position=[0.,0.,s])
    W.export(name,caption)


# End
